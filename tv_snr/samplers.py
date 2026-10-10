import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.integrate import solve_ivp
from torch import nn
from tqdm import tqdm

from .base import ReverseDiffusion
from .constants import class_labels_key, image_key
from .functional import _check_shapes, sample_noise_like
from .sdes import RevSDE
from .snr import Scale_SNR_SDE
from .time_schedules import TimeSchedule

logger = logging.getLogger(__name__)


class Sampler:
    """
    Base class for for sampling or denoising from diffusion models.
    """

    def __init__(
        self,
        reverse_process: ReverseDiffusion,
        denoiser: Union[nn.Module, str],
        noise_pred_key: str = "eps_pred",
        std_key: str = "sigma",
        conditional: bool = False,
        guidance_strength: float = 0.0,
        cond_key: Optional[str] = None,
        cutoff: float = 5.0,
        recompute_neighbors: bool = False,
        additional_keys: List[str] = [],
        save_progress: bool = False,
        progress_stride: int = 1,
        results_on_cpu: bool = True,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        """
        Args:
            reverse_process: The reverse diffusion process to sample from.
            denoiser: The denoiser to use for the reverse process.
            noise_pred_key: The key for the noise prediction in the model output.
            std_key: The key for the standard deviation in the model input.
            cutoff: The cutoff for the neighbor list.
            recompute_neighbors: Whether to recompute the neighbors at each step.
            save_progress: Whether to save the progress of the reverse process.
            progress_stride: The stride to save the progress.
            results_on_cpu: Whether to save the results on the CPU.
            device: The device to use for the sampling.
        """
        self.reverse_process = reverse_process
        self.denoiser = denoiser
        self.noise_pred_key = noise_pred_key
        self.std_key = std_key
        self.conditional = conditional
        self.guidance_strength = guidance_strength
        self.cond_key = cond_key
        self.cutoff = cutoff
        self.save_progress = save_progress
        self.progress_stride = progress_stride
        self.recompute_neighbors = recompute_neighbors
        self.results_on_cpu = results_on_cpu
        self.additional_keys = additional_keys

        if self.conditional and self.cond_key is None:
            raise ValueError(
                "The conditional key must be provided for conditional models."
            )

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if isinstance(self.denoiser, str):
            self.denoiser = torch.load(
                self.denoiser, map_location=self.device, weights_only=False
            ).eval()
        elif self.denoiser is not None:
            self.denoiser = self.denoiser.to(self.device).eval()

        if self.get_T() is None:
            raise ValueError("The schedule must be descritised during sampling.")

        # To save the progress of the reverse process (i.e. the reverse trajectory)
        self._trajs = {}

    def update_model(self, model: nn.Module):
        """
        Updates the denoiser model.

        Args:
            model: the new denoiser model.
        """
        self.denoiser = model

    def sample_prior(
        self,
        inputs: Dict[str, torch.Tensor],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Samples the prior p(x_t) for the reverse diffusion process.
        It uses the forward diffusion process to diffuse the input data if t not None,
        otherwise it samples from the tractable prior p(x_T).

        Args:
            inputs: input data with x_0 for each target property.
            t: the start time step of the reverse process,
                starting at 0 for diffusion step 1 until T-1.
            **kwargs: additional arguments to pass to the reverse process.
        """

        raise ValueError("Use latents from edm for sampling.")

        # x_t = self.reverse_process.sample_prior(
        #     x_0, inputs[properties.idx_m], **kwargs
        # )

        # outputs = {image_key: x_t.to(device=self.device)}

        # return outputs

    def _prepare_inputs(
        self, inputs: Dict[str, torch.Tensor], start: Optional[int] = None
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """Prepare input data for processing."""
        # Default is t=T
        if start is None:
            start = self.get_T()

        if not (isinstance(start, int) and 1 <= start <= self.get_T()):
            raise ValueError(
                "t must be one int between 1 and T that indicates the starting step."
                "Sampling using different starting steps is not supported yet for DDPM."
            )

        # copy inputs to avoid inplace operations
        batch = {prop: val.clone().to(self.device) for prop, val in inputs.items()}

        self._trajs = {}

        return batch, start

    def _save_trajectory(self, batch: Dict[str, torch.Tensor], i: int):
        """Save the reverse trajectory progress if required."""
        if self.save_progress and (i % self.progress_stride == 0):
            if not self._trajs:
                self._trajs[image_key] = [batch[image_key].cpu().float().clone()]
            else:
                self._trajs[image_key].append(batch[image_key].cpu().float().clone())

    def _prepare_outputs(
        self, batch: Dict[str, torch.Tensor], start: int
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """Prepare final output after denoising."""
        x_0 = {
            image_key: (
                batch[image_key].cpu() if self.results_on_cpu else batch[image_key]
            )
        }

        num_steps = torch.full_like(
            batch[image_key], start, dtype=torch.long, device="cpu"
        )

        trajs = {
            k: torch.cat([elem.unsqueeze(-1) for elem in elems], dim=-1)
            for k, elems in self._trajs.items()
        }

        return x_0, num_steps, trajs

    @abstractmethod
    def get_T(self) -> int:
        """
        Returns the number of steps of the descritised reverse process.
        """
        raise NotImplementedError

    @abstractmethod
    def get_sigma(
        self, inputs: Dict[str, torch.Tensor], curr_steps: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns the standard deviation for the given reverse/time step.
        """
        raise NotImplementedError

    @abstractmethod
    def get_sigmas(self) -> torch.Tensor:
        """
        Returns the standard deviations for all reverse/time steps.
        """
        raise NotImplementedError

    @torch.no_grad()
    def apply_guidance(
        self,
        inputs: Dict[str, torch.Tensor],
        cond_noise: torch.Tensor,
    ):
        if (
            self.conditional
            and self.guidance_strength > 0.0
            and inputs[self.cond_key].sum() > 0.0  # type: ignore
        ):
            # set unconditional state
            inputs[self.cond_key] = inputs[self.cond_key] * 0  # type: ignore

            # cast input to float for the denoiser
            inputs = {
                key: val.float() if val.dtype == torch.float64 else val
                for key, val in inputs.items()
            }

            uncond_noise = self.denoiser(inputs)[self.noise_pred_key]  # type: ignore
            noise_pred = (
                1 + self.guidance_strength
            ) * cond_noise - self.guidance_strength * uncond_noise
        else:
            noise_pred = cond_noise

        return noise_pred

    def model_inputs(self, inputs: Dict[str, torch.Tensor], curr_steps: torch.Tensor):
        """
        Update the model inputs before inference.
        """
        # cast input to float for the denoiser
        # inputs = {
        #     key: val.float() if val.dtype == torch.float64 else val
        #     for key, val in inputs.items()
        # }

        # get the std of the current marginal p_t and and add it to the model input
        inputs[self.std_key] = self.get_sigma(inputs, curr_steps)

        return inputs

    @abstractmethod
    def iter(
        self, batch: Dict[str, torch.Tensor], i: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one step of the reverse process.

        Args:
            batch: the input data for the reverse process.
        """
        raise NotImplementedError

    def denoise(
        self,
        inputs: Dict[str, torch.Tensor],
        start: Optional[int] = None,
        progress_bar: bool = True,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Denoise the input data using the reverse process.

        Args:
            inputs: input data for denoising.
            start: The time step to start denoising from. Default is the last step.
        """
        inputs = {k: v.to(torch.float64) for (k, v) in inputs.items()}
        batch, start = self._prepare_inputs(inputs, start)

        for i in tqdm(reversed(range(start)), disable=not progress_bar):
            # perform one reverse step
            x_t_next, _ = self.iter(batch, i)

            # save history if required. Before updating the batch with the new state.
            self._save_trajectory(batch, start - (i + 1))

            batch[image_key] = x_t_next

        return self._prepare_outputs(batch, start)

    def __call__(
        self,
        *args,
        **kwargs,
    ) -> Any:
        """
        Defines the default call method.
        Currently equivalent to calling ``self.denoise``.
        """
        return self.denoise(*args, **kwargs)


class Euler(Sampler):
    """
    Uses 1st order Euler to integrate the SDE/ODE.
    local error: O(dt^2)
    """

    def __init__(
        self,
        reverse_process: RevSDE,
        denoiser: Union[str, nn.Module],
        time_schedule: TimeSchedule,
        max_stoch_std: float = torch.inf,
        min_stoch_std: float = 0.0,
        clip_stoch_std: bool = False,
        selected_stoch: bool = False,
        **kwargs,
    ):
        """
        Args:
            reverse_process: SDE of the reverse diffusion process.
            denoiser: Denoiser or path to denoiser to use for the reverse process.
            time_schedule: The time schedule to use for the reverse SDE.
            std_key: Key to save the standard deviation in the model input.
            noise_pred_key: Key for the predicted noise in model output.
        """
        self.time_schedule = time_schedule
        super().__init__(reverse_process, denoiser, **kwargs)
        self.reverse_process = reverse_process

        self.max_stoch_std = max_stoch_std
        self.min_stoch_std = min_stoch_std
        self.clip_stoch_std = clip_stoch_std
        self.selected_stoch = selected_stoch

        # Euler defualt to only first order integration
        self._second_order = False

    @torch.no_grad()
    def inference_step(
        self, inputs: Dict[str, torch.Tensor], i: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One inference step for the model to get the score prediction.

        Args:
            inputs: input data for noise prediction.
            curr_steps: the current iteration of the reverse process.
        """
        # broadcast the current step to the batch
        curr_steps = torch.full_like(
            inputs[image_key],
            fill_value=i,
            dtype=torch.long,
            device=self.device,
        )

        # prepare the model inputs
        mod_inputs = self.model_inputs(inputs, curr_steps)

        gamma = self.snr_sch(self.curr_t(curr_steps)).to(torch.float64).mean()
        tau = self.scale_sch(self.curr_t(curr_steps)).to(torch.float64).mean()
        sigma = torch.sqrt(1 / gamma)

        # our gamma is gamma**2 in the paper
        # our tau is tau**2 in the paper
        scaler = 1 / torch.sqrt((tau * gamma) / (1 + gamma))

        class_labels = (
            mod_inputs[class_labels_key] if class_labels_key in mod_inputs else None
        )
        model_out = self.denoiser(mod_inputs[image_key] * scaler, sigma, class_labels).to(torch.float64)  # type: ignore

        # fetch the noise prediction
        noise_pred = model_out  # [self.noise_pred_key]

        # guidance if required
        noise_pred = self.apply_guidance(mod_inputs, noise_pred)

        return noise_pred, curr_steps

    def get_T(self) -> int:
        """
        Returns the number of steps of the descritised reverse process.
        """
        return self.time_schedule.T

    def get_sigma(
        self, inputs: Dict[str, torch.Tensor], curr_steps: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the standard deviation of the perturbation kernel for the current step.

        Args:
            curr_steps: The current iteration of the reverse process.
        """
        return self.time_schedule.get_sigma(curr_steps)

    def get_sigmas(self) -> torch.Tensor:
        """
        Returns the standard deviations for all reverse/time steps.
        """
        if not self.time_schedule.discretize or self.time_schedule.sigmas is None:
            raise ValueError(
                "Returning all sigmas require a discretized time schedule."
                "Please set the sigmas in the time schedule."
            )
        else:
            return self.time_schedule.sigmas

    def _infer_score(
        self, denoised_x: torch.Tensor, x, std: torch.Tensor, snr, tau
    ) -> torch.Tensor:
        """
        Computes the score for the reverse process from the noise.

        Args:
            noise: The noise tensor.
            std: The standard deviation tensor of the perturbation kernel.
        """

        # TODO: is this the correct way to get the score?
        # note that we predict D(x,sigma) here, not the score!
        # return (denoised_x - x) / std**2

        # sigma = torch.sqrt(1/snr)
        # s = std / sigma
        # prev = (denoised_x - x / s) / (sigma**2)

        # this is Tweedie's formula for the score
        # x_0 (predicted) is already on data manifold and doesn't need to be scaled
        # a = torch.sqrt(torch.pow(snr / (1 + snr), tau))

        # I think std=b here and gamma (code) = gamma^2 (paper)
        a = torch.sqrt(snr * (std**2))
        return (denoised_x * a - x) / (std**2)

        # if len(std.shape) != len(noise.shape):
        #     std = std.unsqueeze(-1)
        # return -1.0 * noise / std

    def get_score(
        self, batch: Dict[str, torch.Tensor], i: int, x_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the score at step i from the denoiser output.
        """
        noise, curr_steps = self.inference_step(batch, i)

        # infer the score for the reverse process
        std = self.get_sigma(batch, curr_steps)
        snr = self.snr_sch(self.curr_t(curr_steps))
        tau = self.scale_sch(self.curr_t(curr_steps))
        score = self._infer_score(noise, x_t, std, snr, tau)

        return score, curr_steps

    def _get_sde_time(
        self, curr_steps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get SDE time information for the current and next step.

        Args:
            curr_steps: The current iteration of the reverse process.
        """
        t = self.time_schedule(curr_steps)
        t_next = (
            self.time_schedule(curr_steps - 1)
            if curr_steps.unique().item() > 0
            else t * 0.0
        )
        return t, t_next

    def _euler_step(self, x_t, drift, dt) -> torch.Tensor:
        """
        Perform one step of the Euler integration.

        Args:
            x_t: Current state tensor.
            drift: The drift term.
            dt: The time step difference.
        """
        # apply the Euler step
        return x_t + drift * dt

    def _euler_maruyama_step(self, x_t_next, diffusion, dt, idx_m) -> torch.Tensor:
        """current_step
        Inject noise as in the Euler-Maruyama integration.

        Args:
            x_t_next: Next state tensor.
            diffusion: The diffusion term.
            dt: The time step difference.
            idx_m: Index of molecule in flattened batch.
        """
        # the std of the added stochasticity (noise)
        noise_std = diffusion * torch.sqrt(torch.abs(dt))

        # clip the std of the added stochasticity (noise).
        if self.clip_stoch_std:
            noise_std = torch.clamp(
                noise_std, min=self.min_stoch_std, max=self.max_stoch_std
            )

        noise = sample_noise_like(
            x_t_next,
            self.reverse_process.invariant,
            idx_m,
        )

        return x_t_next + noise_std * noise

    def _heun_correction_step(self, *args, **kwargs) -> torch.Tensor:
        """
        Apply Heun's second-order correction.
        """
        raise ValueError(
            "Heun's second order correction is not implemented for Euler. "
            "Use Heun class instead."
        )

    def rsde_coefficients(
        self, x_t: torch.Tensor, score: torch.Tensor, t: torch.Tensor, dt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the drift and diffusion coefficients of the reverse SDE.
        """
        drift, diffusion = self.reverse_process.coefficients(x_t, score, t, t + dt)

        # alternate between deterministic and stochastic coefficients if specified
        if self.reverse_process.stochastic and self.selected_stoch:
            # check if the noise std is within the desired range
            noise_std = diffusion * torch.sqrt(torch.abs(dt))
            noise_std = noise_std.unique().item()
            if (noise_std > self.max_stoch_std) or (noise_std < self.min_stoch_std):
                self.reverse_process.stochastic = False
                drift, diffusion = self.reverse_process.coefficients(
                    x_t, score, t, t + dt
                )
                self.reverse_process.stochastic = True

        return drift, diffusion

    def iter(
        self, batch: Dict[str, torch.Tensor], i: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one iteration of the reverse process.

        Args:
            batch: Batch of inputs.
            curr_steps: the current iteration of the reverse process.
        """
        x_t = batch[image_key]

        # predict the score from using the denoiser prediction
        score, curr_steps = self.get_score(batch, i, x_t)

        # get diffusion SDE time
        t, t_next = self._get_sde_time(curr_steps)
        x_t, t, t_next = _check_shapes(x_t, t, t_next)
        dt = t_next - t

        # note that this is the drift WITHOUT dt (i.e. dx/dt)
        drift, diffusion = self.rsde_coefficients(x_t, score, t, dt)

        # apply one Euler step to get the next step
        x_t_next = self._euler_step(x_t, drift, dt)

        # inject noise if stochastic (reduces to Euler-Maruyama method)
        if self.reverse_process.stochastic:
            x_t_next = self._euler_maruyama_step(x_t_next, diffusion, dt, None)

        # Heun's second order correction
        if self._second_order:
            x_t_next = self._heun_correction_step(
                batch, i, curr_steps, x_t, x_t_next, drift, t_next, dt
            )

        return x_t_next, curr_steps


class SNREuler(Euler):
    """
    Uses 1st order Euler to integrate the SDE/ODE.
    local error: O(dt^2)
    """

    def __init__(
        self,
        reverse_process: RevSDE,
        denoiser: Union[str, nn.Module],
        out_var_scaler: float,
        time_schedule: TimeSchedule,
        T=None,
        scale_input: bool = True,
        snr_key: str = "gamma",
        **kwargs,
    ):
        """
        Args:
            reverse_process: SDE of the reverse diffusion process.
            denoiser: Denoiser or path to denoiser to use for the reverse process.
            time_schedule: The time schedule to use for the reverse SDE.
            std_key: Key to save the standard deviation in the model input.
            noise_pred_key: Key for the predicted noise in model output.
        """
        assert (T is not None) or (
            time_schedule is not None
        ), "Either T or time_schedule must be provided."
        self.T = T
        self.reverse_process = reverse_process
        self.snr_key = snr_key
        self.out_var_scaler = out_var_scaler

        super().__init__(reverse_process, denoiser, time_schedule, **kwargs)

        if not isinstance(reverse_process.forward_process, Scale_SNR_SDE):
            raise ValueError("SNREuler requires a Scale_SNR_SDE forward process.")
        else:
            self.snr_sch = reverse_process.forward_process.snr_sch
            self.scale_sch = reverse_process.forward_process.scale_sch
            self.sde = reverse_process.forward_process

        self.scale_input = scale_input

    def get_T(self) -> int:
        """
        Returns the number of steps of the descritised reverse process.
        """
        if self.time_schedule is not None:
            return self.time_schedule.T

        return self.T

    def curr_t(self, curr_steps: torch.Tensor) -> torch.Tensor:
        """
        Get the time for the current step.
        """
        # t = (curr_steps.to(self.snr_sch.dtype)+1) / (self.T)
        # t = self.snr_sch.clip_t(t)
        # return t

        if self.time_schedule is not None:
            return self.time_schedule(curr_steps)

        return (curr_steps.to(self.snr_sch.dtype) + 1) / (self.T)

    def get_sigma(
        self, inputs: Dict[str, torch.Tensor], curr_steps: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the standard deviation of the perturbation kernel for the current step.

        Args:
            curr_steps: The current iteration of the reverse process.
        """
        _, std = self.reverse_process.forward_process.perturbation_kernel(
            inputs[image_key], self.curr_t(curr_steps)
        )

        return std

    def get_sigmas(self) -> torch.Tensor:
        """
        Returns the standard deviations for all reverse/time steps.
        """
        raise ValueError("SNREuler does not support returning all sigmas.")

    def _get_sde_time(
        self, curr_steps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get SDE time information for the current and next step.

        Args:
            curr_steps: The current iteration of the reverse process.
        """
        t = self.curr_t(curr_steps)
        if curr_steps.unique().item() > 0:
            t_next = self.curr_t(curr_steps - 1)
        else:
            t_next = t * 0.0

        return t, t_next

    def model_inputs(self, inputs: Dict[str, torch.Tensor], curr_steps: torch.Tensor):
        """
        Update the model inputs before inference.
        """
        # copy and cast input to float for the denoiser
        # inputs = {
        #     key: val.float() if val.dtype == torch.float64 else val
        #     for key, val in inputs.items()
        # }

        # get the current SNR as model input
        # inputs[self.snr_key] = self.snr_sch(self.curr_t(curr_steps)).to(torch.float64)

        # TODO: is this the correct way to get sigma?
        # assuming that we have a schedule that also has a sigma
        # this uses the SDE perturbation kernel to get sigma, which is good
        inputs[self.std_key] = self.get_sigma(inputs, curr_steps)

        # this is worse than above
        # inputs[self.std_key] = torch.pow(inputs[self.snr_key], -2)

        if self.scale_input:
            inputs[image_key] = self.sde.scale_input(
                inputs[image_key], self.curr_t(curr_steps)
            ).float()

        return inputs

    def _prepare_outputs(
        self, batch: Dict[str, torch.Tensor], start: int
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """Prepare final output after denoising."""
        x_0 = {
            image_key: (
                batch[image_key].cpu() if self.results_on_cpu else batch[image_key]
            )
        }

        num_steps = torch.full_like(
            batch[image_key], start, dtype=torch.long, device="cpu"
        )

        trajs = {
            k: torch.cat([elem.unsqueeze(-1) for elem in elems], dim=-1)
            for k, elems in self._trajs.items()
        }

        # scale output back to original data variance
        x_0[image_key] = x_0[image_key] * self.out_var_scaler**0.5
        if image_key in trajs:
            trajs[image_key] = trajs[image_key] * self.out_var_scaler**0.5

        return x_0, num_steps, trajs


class SNRHeun(SNREuler):
    """
    Uses 2nd order Heun to integrate the SDE/ODE.
    local error: O(dt^3)
    """

    def __init__(self, reverse_process: RevSDE, *args, **kwargs):
        if reverse_process.stochastic:
            raise ValueError(
                "Heun second order solver does not yet support stochasticity."
            )

        # The only difference between the two classes is the second_order flag.
        super().__init__(reverse_process, *args, **kwargs)
        self._second_order = True

        # if self.reverse_process.forward_process.disc_type != "forward":
        #     raise ValueError(
        #         "Heun second order solver requires forward discretization."
        #     )
        # TODO: Why?

    def _heun_correction_step(
        self,
        batch: Dict[str, torch.Tensor],
        i: int,
        curr_steps: torch.Tensor,
        x_t: torch.Tensor,
        x_t_next: torch.Tensor,
        drift: torch.Tensor,
        t_next: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply Heun's second-order correction.

        Args:
            batch: Batch of inputs.
            x_t: Current state tensor.
            x_t_next: Predicted next state tensor.
            drift: The drift term.
            dt: The time step difference.
        """
        # Not defined at the end of the reverse process t=0.
        if i <= 0:
            return x_t_next

        # get model inference for the next step t_next
        batch[image_key] = x_t_next
        score, next_steps = self.get_score(batch, i - 1, x_t_next)

        # get the the drift d = dx/dt for the next step t_next (not t)
        drift_t_next, _ = self.reverse_process.coefficients(
            x_t_next, score, t_next, t_next
        )

        # correct the next step prediction with the average of the drifts
        return x_t + 0.5 * (drift + drift_t_next) * dt

    # def denoise(
    #     self,
    #     inputs: Dict[str, torch.Tensor],
    #     start: Optional[int] = None,
    # ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[Dict[str, torch.Tensor]]]:
    #     x_0, num_steps, trajs = super(Heun, self).denoise(inputs, start)
    #     num_steps = (num_steps * 2) - 1
    #     return x_0, num_steps, trajs


class SNRRK(SNREuler):
    """
    Using the Runge-Kutta methods to integrate the SDE/ODE.
    """

    def __init__(
        self,
        method="RK45",
        **kwargs,
    ):
        """
        Args:
            method: The Runge-Kutta method to use, default is "RK45".
        """
        super().__init__(
            **kwargs,
        )
        self.method = method
        if self.reverse_process.stochastic:
            raise ValueError("RK does not yet support stochasticity.")
        # if self.sde.disc_type != "forward":
        #     raise ValueError("RK supports only forward discretization.")

    def curr_t(self, curr_steps: torch.Tensor) -> torch.Tensor:
        """
        Get the time for the current step.
        """
        return curr_steps.to(self.snr_sch.dtype)

    @torch.no_grad()
    def inference_step(
        self, inputs: Dict[str, torch.Tensor], i: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One inference step for the model to get the score prediction.

        Args:
            inputs: input data for noise prediction.
            curr_steps: the current iteration of the reverse process.
        """
        # broadcast the current step to the batch
        t = torch.full_like(
            inputs[image_key],
            fill_value=i,
            dtype=torch.float,
            device=self.device,
        )

        # prepare the model inputs
        mod_inputs = self.model_inputs(inputs, t)
        gamma = self.snr_sch(t).to(torch.float64).mean()
        tau = self.scale_sch(t).to(torch.float64).mean()
        sigma = torch.sqrt(1 / gamma)

        scaler = 1 / torch.sqrt((tau * gamma) / (1 + gamma))
        class_labels = (
            mod_inputs[class_labels_key] if class_labels_key in mod_inputs else None
        )
        model_out = self.denoiser(mod_inputs[image_key] * scaler, sigma, class_labels).to(torch.float64)  # type: ignore

        # fetch the noise prediction
        noise_pred = model_out  # [self.noise_pred_key]

        # guidance if required
        noise_pred = self.apply_guidance(mod_inputs, noise_pred)

        return noise_pred, t

    def _ode(
        self, _t: float, _x_t: np.ndarray, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Wrapper that defines the ODE for black box solver.
        """
        x_t = torch.tensor(_x_t, device=self.device).reshape(*batch[image_key].shape)
        batch[image_key] = x_t

        # predict the score from using the denoiser prediction
        noise, t = self.inference_step(batch, _t)

        # infer the score for the reverse process
        std = self.get_sigma(batch, t)
        snr = self.snr_sch(t)
        tau = self.scale_sch(t)
        score = self._infer_score(noise, x_t, std, snr, tau)

        # get diffusion SDE time
        x_t, t = _check_shapes(x_t, t)

        drift, _ = self.reverse_process.coefficients(x_t, score, t, t)

        return drift.flatten().cpu().numpy()

    def iter(
        self, batch: Dict[str, torch.Tensor], i: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Not needed for RK.
        """
        raise NotImplementedError("RK does not support per iteration inference.")

    def denoise(
        self,
        inputs: Dict[str, torch.Tensor],
        start: Optional[int] = None,
        eps: float = 1e-3,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Denoise the input data using the reverse process.

        Args:
            inputs: input data for denoising.
            start: The time step to start denoising from. Default is the last step.
        """
        batch, start = self._prepare_inputs(inputs, start)

        t_s = start * 1.0 / self.T
        t_e = eps

        x_T = batch[image_key].flatten().cpu().numpy()

        solution = solve_ivp(
            self._ode,
            (t_s, t_e),
            x_T,
            rtol=rtol,
            atol=atol,
            method=self.method,
            args=(batch,),
        )

        x_0 = {
            image_key: torch.Tensor(
                solution.y[:, -1], device="cpu" if self.results_on_cpu else self.device
            ).reshape(*batch[image_key].shape)
        }

        # scale output back to original data variance
        x_0[image_key] = x_0[image_key] * self.out_var_scaler**0.5

        return x_0, solution.nfev, None


class DPMSolver(SNREuler):
    """
    DPM-solver proposed in https://arxiv.org/abs/2206.00927.
    Code adapted from their source code.
    """

    def __init__(
        self,
        **kwargs,
    ):
        """
        Args:
        """
        super().__init__(
            **kwargs,
        )
        if self.reverse_process.stochastic:
            raise ValueError("DPM solver does not yet support stochasticity.")
        if self.sde.disc_type != "forward":
            raise ValueError("DPM solver supports only forward discretization.")

    def get_orders_and_timesteps_for_singlestep_solver(self, steps, order, t_T, t_0):
        """
        Get the order of each step for sampling by the singlestep DPM-Solver.

        We combine both DPM-Solver-1,2,3 to use all the function evaluations, which is named as "DPM-Solver-fast".
        Given a fixed number of function evaluations by `steps`, the sampling procedure by DPM-Solver-fast is:
            - If order == 1:
                We take `steps` of DPM-Solver-1 (i.e. DDIM).
            - If order == 2:
                - Denote K = (steps // 2). We take K or (K + 1) intermediate time steps for sampling.
                - If steps % 2 == 0, we use K steps of DPM-Solver-2.
                - If steps % 2 == 1, we use K steps of DPM-Solver-2 and 1 step of DPM-Solver-1.
            - If order == 3:
                - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
                - If steps % 3 == 0, we use (K - 2) steps of DPM-Solver-3, and 1 step of DPM-Solver-2 and 1 step of DPM-Solver-1.
                - If steps % 3 == 1, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-1.
                - If steps % 3 == 2, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-2.

        ============================================
        Args:
            order: A `int`. The max order for the solver (2 or 3).
            steps: A `int`. The total number of function evaluations (NFE).
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            device: A torch device.
        Returns:
            orders: A list of the solver order of each step.
        """
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [
                    3,
                ] * (
                    K - 2
                ) + [2, 1]
            elif steps % 3 == 1:
                orders = [
                    3,
                ] * (
                    K - 1
                ) + [1]
            else:
                orders = [
                    3,
                ] * (
                    K - 1
                ) + [2]
        elif order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [
                    2,
                ] * K
            else:
                K = steps // 2 + 1
                orders = [
                    2,
                ] * (
                    K - 1
                ) + [1]
        elif order == 1:
            K = steps
            orders = [
                1,
            ] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")

        timesteps = torch.linspace(t_T, t_0, steps + 1).long()
        timesteps_outer = timesteps[
            torch.cumsum(
                torch.tensor(
                    [
                        0,
                    ]
                    + orders
                ),
                0,
            )
        ].to(self.device)
        return timesteps_outer, orders

    def get_lambda(self, curr_step: torch.Tensor) -> torch.Tensor:
        """
        Get the lambda for the current step. (winnie: i.e. log snr)
        """
        return 0.5 * self.snr_sch(self.curr_t(curr_step)).log()

    def time_from_lambda(self, lambda_: torch.Tensor) -> torch.Tensor:
        """
        Get the time from lambda.
        """
        gamma = torch.exp(2 * lambda_)
        t = self.snr_sch.inverse(gamma)
        curr_step = t * self.T
        curr_step = torch.round(curr_step).long()
        return curr_step

    def get_alpha_sigma(self, curr_steps: torch.Tensor):
        """
        Get the alpha and sigma for the current step.
        """
        t = self.curr_t(curr_steps)
        alpha2, sigma2 = self.sde.get_a2_b2(t)

        return alpha2**0.5, sigma2**0.5

    @torch.no_grad()
    def inference_step(self, inputs: Dict[str, torch.Tensor], i: float):

        inf_out, curr_steps = super().inference_step(inputs, i)
        std = self.get_sigma(inputs, curr_steps)
        snr = self.snr_sch(self.curr_t(curr_steps))
        tau = self.scale_sch(self.curr_t(curr_steps))
        x_t = inputs[image_key]

        score = self._infer_score(inf_out, x_t, std, snr, tau)
        return -score * std, curr_steps

    def dpm_solver_first_update(self, batch, s, t):
        """
        DPM-Solver-1 (equivalent to DDIM) from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        x_t = batch[image_key]

        lambda_s, lambda_t = self.get_lambda(s), self.get_lambda(t)
        h = lambda_t - lambda_s

        alpha_s, sigma_s = self.get_alpha_sigma(s)
        alpha_t, sigma_t = self.get_alpha_sigma(t)

        phi_1 = torch.expm1(h)

        model_s, _ = self.inference_step(batch, s.item())

        x_t = (alpha_t / alpha_s) * x_t - (sigma_t * phi_1) * model_s

        return x_t

    def singlestep_dpm_solver_second_update(
        self,
        batch,
        s,
        t,
        r1=0.5,
    ):
        """
        Singlestep solver DPM-Solver-2 from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            r1: A `float`. The hyperparameter of the second-order solver.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if r1 is None:
            r1 = 0.5

        x_t = batch[image_key]

        lambda_s, lambda_t = self.get_lambda(s), self.get_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h

        s1 = self.time_from_lambda(lambda_s1)

        alpha_s, sigma_s = self.get_alpha_sigma(s)
        alpha_s1, sigma_s1 = self.get_alpha_sigma(s1)
        alpha_t, sigma_t = self.get_alpha_sigma(t)

        phi_11 = torch.expm1(r1 * h)
        phi_1 = torch.expm1(h)

        model_s, _ = self.inference_step(batch, s.item())

        x_s1 = (alpha_s1 / alpha_s) * x_t - (sigma_s1 * phi_11) * model_s

        batch[image_key] = x_s1
        model_s1, _ = self.inference_step(batch, int(s1.item()))

        x_t = (
            (alpha_t / alpha_s) * x_t
            - (sigma_t * phi_1) * model_s
            - (0.5 / r1) * (sigma_t * phi_1) * (model_s1 - model_s)
        )

        return x_t

    def singlestep_dpm_solver_third_update(
        self,
        batch,
        s,
        t,
        r1=1.0 / 3.0,
        r2=2.0 / 3.0,
    ):
        """
        Singlestep solver DPM-Solver-3 from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            r1: A `float`. The hyperparameter of the third-order solver.
            r2: A `float`. The hyperparameter of the third-order solver.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if r1 is None:
            r1 = 1.0 / 3.0
        if r2 is None:
            r2 = 2.0 / 3.0

        x_t = batch[image_key]

        lambda_s, lambda_t = self.get_lambda(s), self.get_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        lambda_s2 = lambda_s + r2 * h

        s1 = self.time_from_lambda(lambda_s1)
        s2 = self.time_from_lambda(lambda_s2)

        # print(s, t, s1, s2)

        alpha_s, sigma_s = self.get_alpha_sigma(s)
        alpha_s1, sigma_s1 = self.get_alpha_sigma(s1)
        alpha_s2, sigma_s2 = self.get_alpha_sigma(s2)
        alpha_t, sigma_t = self.get_alpha_sigma(t)

        phi_11 = torch.expm1(r1 * h)
        phi_12 = torch.expm1(r2 * h)
        phi_1 = torch.expm1(h)
        phi_22 = torch.expm1(r2 * h) / (r2 * h) - 1.0
        phi_2 = phi_1 / h - 1.0

        model_s, _ = self.inference_step(batch, s.item())

        x_s1 = (alpha_s1 / alpha_s) * x_t - (sigma_s1 * phi_11) * model_s
        batch[image_key] = x_s1
        model_s1, _ = self.inference_step(batch, int(s1.item()))

        x_s2 = (
            (alpha_s2 / alpha_s) * x_t
            - (sigma_s2 * phi_12) * model_s
            - r2 / r1 * (sigma_s2 * phi_22) * (model_s1 - model_s)
        )
        batch[image_key] = x_s2
        model_s2, _ = self.inference_step(batch, int(s2.item()))

        x_t = (
            (alpha_t / alpha_s) * x_t
            - (sigma_t * phi_1) * model_s
            - (1.0 / r2) * (sigma_t * phi_2) * (model_s2 - model_s)
        )

        return x_t

    def curr_t(self, curr_steps: torch.Tensor) -> torch.Tensor:
        """
        Get the time for the current step.
        """

        # TODO: check this
        if self.time_schedule is not None:
            # return self.time_schedule(curr_steps - 1)
            raise ValueError("Time schedule not supported for DPM solver.")

        return (curr_steps.to(self.snr_sch.dtype)) / (self.T)

    def singlestep_dpm_solver_update(
        self, batch, s, t, order, r1=None, r2=None
    ) -> torch.Tensor:
        if order == 1:
            return self.dpm_solver_first_update(batch, s, t)
        elif order == 2:
            return self.singlestep_dpm_solver_second_update(batch, s, t, r1=r1)
        elif order == 3:
            return self.singlestep_dpm_solver_third_update(batch, s, t, r1=r1, r2=r2)
        else:
            raise ValueError("DPM solver supports only order 1, 2, or 3.")

    def data_prediction_fn(self, batch, t):
        """
        Return the data prediction model (with corrector).
        """
        x_t = batch[image_key]

        noise, _ = self.inference_step(batch, t.item())

        alpha_t, sigma_t = self.get_alpha_sigma(t)
        x0 = (x_t - sigma_t * noise) / alpha_t

        return x0

    @torch.no_grad()
    def denoise(
        self,
        inputs: Dict[str, torch.Tensor],
        start: Optional[int] = None,
        order: int = 3,
        denoise_to_zero=False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the sample at time `t_end` by DPM-Solver, given the initial `x` at time `t_start`.

        Args:
            x: A pytorch tensor. The initial value at time `t_start`
                e.g. if `t_start` == T, then `x` is a sample from the standard normal distribution.
            steps: A `int`. The total number of function evaluations (NFE).
            t_start: A `float`. The starting time of the sampling.
                If `T` is None, we use self.noise_schedule.T (default is 1.0).
            t_end: A `float`. The ending time of the sampling.
                If `t_end` is None, we use 1. / self.noise_schedule.total_N.
                e.g. if total_N == 1000, we have `t_end` == 1e-3.
                For discrete-time DPMs:
                    - We recommend `t_end` == 1. / self.noise_schedule.total_N.
                For continuous-time DPMs:
                    - We recommend `t_end` == 1e-3 when `steps` <= 15; and `t_end` == 1e-4 when `steps` > 15.
            order: A `int`. The order of DPM-Solver.
            method: A `str`. The method for sampling. 'singlestep' or 'multistep' or 'singlestep_fixed' or 'adaptive'.
            denoise_to_zero: A `bool`. Whether to denoise to time 0 at the final step.
                Default is `False`. If `denoise_to_zero` is `True`, the total NFE is (`steps` + 1).

                This trick is firstly proposed by DDPM (https://arxiv.org/abs/2006.11239) and
                score_sde (https://arxiv.org/abs/2011.13456). Such trick can improve the FID
                for diffusion models sampling by diffusion SDEs for low-resolutional images
                (such as CIFAR-10). However, we observed that such trick does not matter for
                high-resolutional images. As it needs an additional NFE, we do not recommend
                it for high-resolutional images.
        """
        batch, start = self._prepare_inputs(inputs, start)
        t_0 = 0.0
        t_T = start
        steps = self.get_T()

        timesteps_outer, orders = self.get_orders_and_timesteps_for_singlestep_solver(
            steps=steps,
            order=order,
            t_T=t_T,
            t_0=t_0,
        )

        # print(orders)

        mols = []

        for step, order in enumerate(orders):
            s, t = timesteps_outer[step], timesteps_outer[step + 1]
            timesteps_inner = (
                torch.linspace(s.item(), t.item(), order + 1).long().to(self.device)
            )

            # lambda = lambda(a/b) = 0.5*log(SNR)
            lambda_inner = self.get_lambda(timesteps_inner)

            h = lambda_inner[-1] - lambda_inner[0]
            r1 = None if order <= 1 else (lambda_inner[1] - lambda_inner[0]) / h
            r2 = None if order <= 2 else (lambda_inner[2] - lambda_inner[0]) / h

            x_t_next = self.singlestep_dpm_solver_update(
                batch, s, t, order=order, r1=r1, r2=r2
            )

            batch[image_key] = x_t_next

            mols.append(x_t_next.clone())

        if denoise_to_zero:
            # one step from t_0 = eps to t=0
            t0 = torch.zeros((1,), dtype=torch.long).to(self.device)
            x_t_next = self.data_prediction_fn(batch, t0)
            batch[image_key] = x_t_next

        x_0 = {
            image_key: x_t_next.to(device="cpu" if self.results_on_cpu else self.device)
        }

        # scale output back to original data variance
        x_0[image_key] = x_0[image_key] * self.out_var_scaler**0.5

        return x_0, steps, mols


class ParamScoreMixin:
    """
    For a denoiser that directly returns the score s(x, t), e.g. an analytic score.
    """

    def get_score(
        self, batch: Dict[str, torch.Tensor], i: int, x_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        curr_steps = torch.full_like(
            x_t,
            fill_value=i,
            dtype=torch.long,
            device=self.device,
        )
        return self.denoiser(x_t, self.curr_t(curr_steps)), curr_steps


class SNREulerParamScore(ParamScoreMixin, SNREuler):
    pass


class SNRHeunParamScore(ParamScoreMixin, SNRHeun):
    pass
