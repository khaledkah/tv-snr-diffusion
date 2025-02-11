import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from ase import Atoms
from torch import nn
from tqdm import tqdm

from .base import ReverseDiffusion

from typing import Dict, Tuple, Union

import torch
from torch import nn

from .sdes import RevSDE
from .snr import SNR_SDE, Scale_SNR_SDE


logger = logging.getLogger(__name__)
from typing import Dict, Tuple, Union

import torch
from torch import nn

from .functional import _check_shapes, sample_noise_like
from .sdes import RevSDE
from .time_schedules import TimeSchedule
from .sampler import *

from .constants import image_key


class EulerAdaptiveScale(Sampler):
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
        sigma = torch.sqrt(1/gamma)

        # our gamma is gamma**2 in the paper
        # our tau is tau**2 in the paper
        scaler = 1/torch.sqrt((tau*gamma)/(1+gamma))
        model_out = self.denoiser(mod_inputs[image_key] * scaler, sigma).to(torch.float64)  # type: ignore

        # fetch the noise prediction
        noise_pred = model_out # [self.noise_pred_key]

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

    def _infer_score(self, denoised_x: torch.Tensor, x, std: torch.Tensor, snr, tau) -> torch.Tensor:
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
        a = torch.sqrt(snr*(std**2))
        return (denoised_x * a - x) / (std**2)

        # if len(std.shape) != len(noise.shape):
        #     std = std.unsqueeze(-1)
        # return -1.0 * noise / std

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
        noise, curr_steps = self.inference_step(batch, i)

        # infer the score for the reverse process
        std = self.get_sigma(batch, curr_steps)
        snr = self.snr_sch(self.curr_t(curr_steps))
        tau = self.scale_sch(self.curr_t(curr_steps))
        #std = torch.sqrt(1/snr.mean())
        score = self._infer_score(noise, x_t, std, snr, tau)

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
            x_t_next = self._euler_maruyama_step(
                x_t_next, diffusion, dt, batch[image_key]
            )

        # Heun's second order correction
        if self._second_order:
            x_t_next = self._heun_correction_step(
                batch, i, curr_steps, x_t, x_t_next, drift, t_next, dt
            )

        return x_t_next, curr_steps


class SNREulerAdaptiveScale(EulerAdaptiveScale):
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
        T = None,
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
        assert (T is not None) or (time_schedule is not None), "Either T or time_schedule must be provided."
        self.T = T
        self.reverse_process = reverse_process
        self.snr_key = snr_key
        self.out_var_scaler = out_var_scaler

        super().__init__(reverse_process, denoiser, time_schedule, **kwargs)

        if not isinstance(reverse_process.forward_process, Scale_SNR_SDE):
            raise ValueError("SNREuler requires an SNR_SDE forward process.")
        else:
            self.snr_sch = reverse_process.forward_process.snr_sch
            self.scale_sch = reverse_process.forward_process.scale_sch

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
        #t = (curr_steps.to(self.snr_sch.dtype)+1) / (self.T)
        # t = self.snr_sch.clip_t(t)
        #return t

        if self.time_schedule is not None:
            return self.time_schedule(curr_steps)
        
        return (curr_steps.to(self.snr_sch.dtype)+1) / (self.T)

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
                batch[image_key].cpu()
                if self.results_on_cpu
                else batch[image_key]
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


class SNRHeunAdaptiveScale(SNREulerAdaptiveScale):
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
        noise, next_steps = self.inference_step(batch, i - 1)

        std = self.get_sigma(batch, next_steps)
        snr = self.snr_sch(self.curr_t(next_steps))
        tau = self.scale_sch(self.curr_t(next_steps))
        #std = torch.sqrt(1/snr.mean())
        score = self._infer_score(noise, x_t_next, std, snr, tau)

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
