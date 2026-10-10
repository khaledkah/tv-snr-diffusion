from typing import Dict, Tuple, Union

import torch
from torch import nn

from schnetpack import properties
from tv_snr.functional import _check_shapes, sample_noise_like
from tv_snr.sdes import RevSDE
from tvsnr_mol.sampling.base import Sampler
from tv_snr.time_schedules import TimeSchedule


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
        snr_key: str = "gamma",
        scale_input: bool = False,
        out_var_scaler: float = 1.0,
        data_var: float = 2.0,
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
        Sampler.__init__(self, reverse_process, denoiser, **kwargs)
        self.reverse_process = reverse_process

        self.snr_key = snr_key
        self.out_var_scaler = out_var_scaler
        self.scale_input = scale_input
        self.data_var = data_var
        self.max_stoch_std = max_stoch_std
        self.min_stoch_std = min_stoch_std
        self.clip_stoch_std = clip_stoch_std
        self.selected_stoch = selected_stoch

        # Euler defualt to only first order integration
        self._second_order = False

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

    def get_gamma(
        self, inputs: Dict[str, torch.Tensor], curr_steps: torch.Tensor
    ) -> torch.Tensor:
        if hasattr(self.time_schedule, "noise_sch"):
            beta_bar = self.get_sigma(inputs, curr_steps) ** 2
            return (1 - beta_bar) / beta_bar
        else:
            return self.get_sigma(inputs, curr_steps) ** -2

    def input_scaler(
        self, inputs: Dict[str, torch.Tensor], std: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Scale the input to keep a unit input variance to the denoiser.
        """
        if not hasattr(self.time_schedule, "noise_sch"):
            inputs[properties.R] = inputs[properties.R] / torch.sqrt(
                std**2 + 1
            ).unsqueeze(-1)

        return inputs

    def model_inputs(self, inputs: Dict[str, torch.Tensor], curr_steps: torch.Tensor):
        """
        Update the model inputs before inference.
        """
        # cast input to float for the denoiser
        inputs = {
            key: val.float() if val.dtype == torch.float64 else val
            for key, val in inputs.items()
        }

        # get the std of the current marginal p_t and and add it to the model input
        inputs[self.std_key] = self.get_sigma(inputs, curr_steps).float()

        inputs[self.snr_key] = self.get_gamma(inputs, curr_steps).float()

        if self.scale_input:
            inputs = self.input_scaler(inputs, inputs[self.std_key])

        return inputs

    def _prepare_outputs(
        self, batch: Dict[str, torch.Tensor], start: int
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """Prepare final output after denoising."""
        x_0 = {
            properties.R: (
                batch[properties.R].cpu()
                if self.results_on_cpu
                else batch[properties.R]
            )
        }

        num_steps = torch.full_like(
            batch[properties.n_atoms], start, dtype=torch.long, device="cpu"
        )

        trajs = {
            k: torch.cat([elem.unsqueeze(-1) for elem in elems], dim=-1)
            for k, elems in self._trajs.items()
        }

        # scale output back to original data variance
        x_0[properties.R] = x_0[properties.R] * self.out_var_scaler**0.5
        if properties.R in trajs:
            trajs[properties.R] = trajs[properties.R] * self.out_var_scaler**0.5

        return x_0, num_steps, trajs

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

    def _infer_score(self, noise: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        Computes the score for the reverse process from the noise.

        Args:
            noise: The noise tensor.
            std: The standard deviation tensor of the perturbation kernel.
        """
        if len(std.shape) != len(noise.shape):
            std = std.unsqueeze(-1)
        return -1.0 * noise / std

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
        x_t = batch[properties.R]

        # predict the score from using the denoiser prediction
        noise, curr_steps = self.inference_step(batch, i)

        # infer the score for the reverse process
        std = self.get_sigma(batch, curr_steps)
        score = self._infer_score(noise, std)

        # get diffusion SDE time
        t, t_next = self._get_sde_time(curr_steps)
        x_t, t, t_next = _check_shapes(x_t, t, t_next)
        dt = t_next - t

        drift, diffusion = self.rsde_coefficients(x_t, score, t, dt)

        # apply one Euler step to get the next step
        x_t_next = self._euler_step(x_t, drift, dt)

        # inject noise if stochastic (reduces to Euler-Maruyama method)
        if self.reverse_process.stochastic:
            x_t_next = self._euler_maruyama_step(
                x_t_next, diffusion, dt, batch[properties.idx_m]
            )

        # Heun's second order correction
        if self._second_order:
            x_t_next = self._heun_correction_step(
                batch, i, curr_steps, x_t, x_t_next, drift, t_next, dt
            )

        return x_t_next, curr_steps


class Heun(Euler):
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
        Euler.__init__(self, reverse_process, *args, **kwargs)
        self._second_order = True

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
        batch[properties.R] = x_t_next.float()
        noise, next_steps = self.inference_step(batch, i - 1)

        std = self.get_sigma(batch, next_steps)
        score = self._infer_score(noise, std)

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
