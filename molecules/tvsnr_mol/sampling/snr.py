from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from scipy.integrate import solve_ivp
from torch import nn
from tqdm import tqdm

from schnetpack import properties
from tv_snr.functional import _check_shapes
from tv_snr.sdes import RevSDE
from tv_snr.snr import Scale_SNR_SDE
from tvsnr_mol.sampling.sde import Euler


class SNREuler(Euler):
    """
    Uses 1st order Euler to integrate the SDE/ODE.
    local error: O(dt^2)
    """

    def __init__(
        self,
        reverse_process: RevSDE,
        denoiser: Union[str, nn.Module],
        T: int,
        out_var_scaler: float,
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
        self.T = T
        Euler.__init__(self, reverse_process, denoiser, None, **kwargs)
        self.reverse_process = reverse_process
        self.snr_key = snr_key
        self.out_var_scaler = out_var_scaler

        if not isinstance(reverse_process.forward_process, Scale_SNR_SDE):
            raise ValueError("SNREuler requires a Scale_SNR_SDE forward process.")
        else:
            self.sde = reverse_process.forward_process
            self.snr_sch = self.sde.snr_sch

        self.scale_input = scale_input

    def get_T(self) -> int:
        """
        Returns the number of steps of the descritised reverse process.
        """
        return self.T

    def curr_t(self, curr_steps: torch.Tensor) -> torch.Tensor:
        """
        Get the time for the current step.
        """
        t = (curr_steps.to(self.snr_sch.dtype) + 1) / (self.T)

        # t = self.snr_sch.clip_t(t)

        return t

    def get_sigma(
        self, inputs: Dict[str, torch.Tensor], curr_steps: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the standard deviation of the perturbation kernel for the current step.

        Args:
            curr_steps: The current iteration of the reverse process.
        """
        _, std = self.reverse_process.forward_process.perturbation_kernel(
            inputs[properties.R], self.curr_t(curr_steps)
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

        if curr_steps.unique().item() >= 0:
            t_next = self.curr_t(curr_steps - 1)
        else:
            t_next = t * 0.0

        return t, t_next

    def model_inputs(self, inputs: Dict[str, torch.Tensor], curr_steps: torch.Tensor):
        """
        Update the model inputs before inference.
        """
        # copy and cast input to float for the denoiser
        inputs = {
            key: val.float() if val.dtype == torch.float64 else val
            for key, val in inputs.items()
        }

        # get the current SNR as model input
        inputs[self.snr_key] = self.snr_sch(self.curr_t(curr_steps)).float()

        if self.scale_input:
            inputs[properties.R] = self.sde.scale_input(
                inputs[properties.R], self.curr_t(curr_steps)
            ).float()

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
        SNREuler.__init__(self, reverse_process, *args, **kwargs)
        self._second_order = True

        # if self.reverse_process.forward_process.disc_type != "forward":
        #     raise ValueError(
        #         "Heun second order solver requires forward discretization."
        #     )

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


class SNRRK(SNREuler):
    """
    Runge-Kutta methods to integrate the SDE/ODE.
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
        if self.sde.disc_type != "forward":
            raise ValueError("RK supports only forward discretization.")

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
            inputs[properties.idx_m],
            fill_value=i,
            dtype=torch.float,
            device=self.device,
        )

        # prepare the model inputs
        mod_inputs = self.model_inputs(inputs, t)

        # forward pass through the denoiser
        model_out = self.denoiser(mod_inputs)  # type: ignore

        # fetch the noise prediction
        noise_pred = model_out[self.noise_pred_key]

        # guidance if required
        noise_pred = self.apply_guidance(mod_inputs, noise_pred)

        return noise_pred, t

    def _ode(
        self, _t: float, _x_t: np.ndarray, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Wrapper that defines the ODE for black box solver.
        """
        x_t = torch.tensor(_x_t, device=self.device).reshape(-1, 3)
        batch[properties.R] = x_t

        # predict the score from using the denoiser prediction
        noise, t = self.inference_step(batch, _t)

        # infer the score for the reverse process
        std = self.get_sigma(batch, t)
        score = self._infer_score(noise, std)

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
        eps: float = 0.0,
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

        t_T = start * 1.0 / self.T
        t_0 = eps

        x_T = batch[properties.R].flatten().cpu().numpy()

        solution = solve_ivp(
            self._ode,
            (t_T, t_0),
            x_T,
            rtol=rtol,
            atol=atol,
            method=self.method,
            args=(batch,),
        )

        x_0 = {
            properties.R: torch.Tensor(
                solution.y[:, -1], device="cpu" if self.results_on_cpu else self.device
            ).reshape(-1, 3)
        }
        num_steps = torch.full_like(
            batch[properties.n_atoms], solution.nfev, dtype=torch.long, device="cpu"
        )

        # scale output back to original data variance
        x_0[properties.R] = x_0[properties.R] * self.out_var_scaler**0.5

        return x_0, num_steps, None


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
        Get the lambda for the current step.
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
        x_t = batch[properties.R]

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

        x_t = batch[properties.R]

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

        batch[properties.R] = x_s1
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

        x_t = batch[properties.R]

        lambda_s, lambda_t = self.get_lambda(s), self.get_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        lambda_s2 = lambda_s + r2 * h

        s1 = self.time_from_lambda(lambda_s1)
        s2 = self.time_from_lambda(lambda_s2)

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
        batch[properties.R] = x_s1
        model_s1, _ = self.inference_step(batch, int(s1.item()))

        x_s2 = (
            (alpha_s2 / alpha_s) * x_t
            - (sigma_s2 * phi_12) * model_s
            - r2 / r1 * (sigma_s2 * phi_22) * (model_s1 - model_s)
        )
        batch[properties.R] = x_s2
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
        t = curr_steps.to(self.snr_sch.dtype) / (self.T)

        # t = self.snr_sch.clip_t(t)

        return t

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
        x_t = batch[properties.R]

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
        **kwargs,
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

        mols = []

        for step, order in tqdm(enumerate(orders)):
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

            batch[properties.R] = x_t_next

            mols.append(x_t_next.clone())

        if denoise_to_zero:
            # one step from t_0 = eps to t=0
            t0 = torch.zeros((1,), dtype=torch.long).to(self.device)
            x_t_next = self.data_prediction_fn(batch, t0)
            batch[properties.R] = x_t_next

        x_0 = {
            properties.R: x_t_next.to(
                device="cpu" if self.results_on_cpu else self.device
            )
        }
        num_steps = torch.full_like(
            batch[properties.n_atoms], steps, dtype=torch.long, device="cpu"
        )

        # scale output back to original data variance
        x_0[properties.R] = x_0[properties.R] * self.out_var_scaler**0.5

        return x_0, num_steps, mols
