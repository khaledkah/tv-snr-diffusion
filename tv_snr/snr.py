from typing import Tuple, Optional

import torch
import numpy as np

from .sdes import SDE
from .snr_schedules import SNRSchedule
from .functional import _check_shapes
from .scale_schedule import ScaleSchedule


class SNR_SDE(SDE):
    """
    Signal-to-noise ratio (SNR) based SDE diffusion process.
    """

    def __init__(
        self,
        snr_sch: SNRSchedule,
        tau: float,
        disc_type: str = "forward",
        log_deriv: Optional[bool] = False,
        **kwargs,
    ):
        """
        Args:
            snr_sch: SNR schedule.
            kwargs: additional arguments to pass to GaussianDiffusion.__init__.
        """
        super().__init__(**kwargs)
        self.snr_sch = snr_sch
        self.tau = tau
        self.log_deriv = log_deriv

        if disc_type not in ["forward", "midpoint", "avg", "difference"]:
            raise ValueError(
                f"Discretization type {disc_type} not supported."
                "Choose from ['forward', 'midpoint', 'avg']."
            )

        self.disc_type = disc_type

    def prior(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the mean and std of the prior distribution p(x_T) = N(mean,std).

        Args:
            x: input tensor, e.g. to infer shape.
        """
        mean = torch.zeros_like(x, dtype=self.dtype)

        log_std = (0.5 * self.tau - 0.5) * self.snr_sch.log_gamma_min
        std = torch.ones_like(x, dtype=self.dtype) * np.exp(log_std)

        return mean, std

    def scale_input(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Scale the input for the SNR to keep a unit input variance to the denoiser.
        """
        gamma = self.snr_sch(t)
        scaler = gamma / (1 + gamma)
        scaler = scaler ** (0.5 - 0.5 * self.tau)
        x = (scaler.unsqueeze(-1) * x)
        return x

    def coefficients(
        self, x_t: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the drift mu(x,t) and diffusion coefficient sigma(t) of the forward SDE
        dx = mu(x,t) dt + sigma(t) dW.

        Args:
            x_t: input tensor x_t ~ p_t to be diffused.
            t: time steps.
        """
        t = t.to(self.dtype)

        if not self.log_deriv:
            # compute SNR and its derivative with respect to time using autograd
            t.requires_grad = True
            gamma = self.snr_sch(t)
            gamma_dot = torch.autograd.grad(gamma.sum(), t, create_graph=False)[0]

            if self.disc_type == "midpoint":
                dt = t_next - t
                t_dt = t + dt / 2
                gamma_dt = self.snr_sch(t_dt)
                gamma_dot = torch.autograd.grad(
                    gamma_dt.sum(), t_dt, create_graph=False
                )[0]
            elif self.disc_type == "avg":
                t_next.requires_grad = True
                gamma_next = self.snr_sch(t_next)
                gamma_dot_next = torch.autograd.grad(
                    gamma_next.sum(), t_next, create_graph=False
                )[0]
                t_next.requires_grad = False
                gamma_dot = 0.5 * (gamma_dot + gamma_dot_next)
            elif self.disc_type == "difference":
                dt = (t_next - t)

                if dt.abs().max() < 1e-10:
                    # no timestep, do nothing
                    # TODO: this is a hack to avoid numerical issues
                    return torch.zeros_like(x_t), torch.zeros_like(x_t)

                gamma_next = self.snr_sch(t_next)
                gamma_dot = (gamma_next - gamma) / dt

            t.requires_grad = False
            gamma = gamma.detach()
            gamma_dot = gamma_dot.detach()

            # compute drift and diffusion coefficients
            drift = x_t * (self.tau * gamma_dot) / (2 * gamma * (1 + gamma))
            diffusion = -(gamma_dot * gamma ** (self.tau - 2)) / (1 + gamma) ** self.tau

        else:
            # compute SNR and its derivative with respect to time using autograd
            t.requires_grad = True
            gamma = self.snr_sch(t)
            log_gamma = torch.log(gamma)
            log_gamma_dot = torch.autograd.grad(log_gamma.sum(), t, create_graph=False)[
                0
            ]

            if self.disc_type == "midpoint":
                dt = t_next - t
                t_dt = t + dt / 2
                log_gamma_dt = torch.log(self.snr_sch(t_dt))
                log_gamma_dot = torch.autograd.grad(
                    log_gamma_dt.sum(), t_dt, create_graph=False
                )[0]
            elif self.disc_type == "avg":
                t_next.requires_grad = True
                log_gamma_next = torch.log(self.snr_sch(t_next))
                log_gamma_dot_next = torch.autograd.grad(
                    log_gamma_next.sum(), t_next, create_graph=False
                )[0]
                t_next.requires_grad = False
                log_gamma_dot = 0.5 * (log_gamma_dot + log_gamma_dot_next)
            elif self.disc_type == "difference":
                log_gamma_next = torch.log(self.snr_sch(t_next))
                log_gamma_dot = (log_gamma_next - log_gamma) / (t_next - t)

            t.requires_grad = False
            gamma = gamma.detach()
            log_gamma = log_gamma.detach()
            log_gamma_dot = log_gamma_dot.detach()

            # compute drift and diffusion coefficients
            drift = x_t * (log_gamma_dot * self.tau) / (2 * (1 + gamma))
            diffusion = (
                -(log_gamma_dot * gamma ** (self.tau - 1)) / (1 + gamma) ** self.tau
            )

        diffusion = torch.sqrt(diffusion)

        return drift, diffusion

    def perturbation_kernel(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the mean and std of the Gaussian perturbation kernel
        p(x_t|x_0) = N(mean(x_0,t),std(t)) to diffuse x_0 ~ p_data.

        Args:
            x_0: input tensor x_0 ~ p_data to be diffused.
            t: time steps.
        """
        t = t.to(self.dtype)

        x_0, t = _check_shapes(x_0, t)

        gamma = self.snr_sch(t)
        a_2 = (gamma / (1 + gamma)) ** self.tau
        b_2 = a_2 / gamma

        mean = x_0.to(self.dtype) * a_2**0.5
        std = torch.ones_like(x_0, dtype=self.dtype) * b_2**0.5

        return mean, std

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Computes the signal-to-noise ratio (SNR) at time t.

        Args:
            t: time steps.

        Returns:
            SNR at time t.
        """
        return self.snr_sch(t)


class Scale_SNR_SDE(SDE):
    """
    Scale/Signal-to-noise ratio (SNR) based SDE diffusion process.
    """
    def __init__(
        self,
        snr_sch: SNRSchedule,
        scale_sch: ScaleSchedule,
        disc_type: str = "forward",
        log_deriv: Optional[bool] = False,
        **kwargs,
    ):
        """
        Args:
            snr_sch: SNR schedule.
            kwargs: additional arguments to pass to GaussianDiffusion.__init__.
        """
        super().__init__(**kwargs)
        self.snr_sch = snr_sch
        self.scale_sch = scale_sch
        self.log_deriv = log_deriv
        if disc_type not in ["forward", "midpoint", "avg", "difference"]:
            raise ValueError(
                f"Discretization type {disc_type} not supported."
                "Choose from ['forward', 'midpoint', 'avg']."
            )
        self.disc_type = disc_type

    def prior(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the mean and std of the prior distribution p(x_T) = N(mean,std).
        Args:
            x: input tensor, e.g. to infer shape.
        """
        mean = torch.zeros_like(x, dtype=self.dtype)
        std = self.scale_sch.max_scale**0.5 * torch.ones_like(x, dtype=self.dtype)
        return mean, std

    def scale_input(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Scale the input for the SNR to keep a unit input variance to the denoiser.
        """
        tau = self.scale_sch(t)**0.5
        x = (x / tau).float()
        return x

    def coefficients(
        self, x_t: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the drift mu(x,t) and diffusion coefficient sigma(t) of the forward SDE
        dx = mu(x,t) dt + sigma(t) dW.
        Args:
            x_t: input tensor x_t ~ p_t to be diffused.
            t: time steps.
        """
        t = t.to(self.dtype)
        if not self.log_deriv:
            # compute SNR and its derivative with respect to time using autograd
            t.requires_grad = True
            sqrt_gamma = self.snr_sch(t) ** 0.5
            sqrt_gamma_dot = torch.autograd.grad(
                sqrt_gamma.sum(), t, create_graph=False
            )[0]
            tau = self.scale_sch(t) ** 0.5
            tau_dot = torch.autograd.grad(tau.sum(), t, create_graph=False)[0]
            if self.disc_type == "midpoint":
                dt = t_next - t
                t_dt = t + dt / 2
                sqrt_gamma_dt = self.snr_sch(t_dt) ** 0.5
                sqrt_gamma_dot = torch.autograd.grad(
                    sqrt_gamma_dt.sum(), t_dt, create_graph=False
                )[0]
                tau_dt = self.scale_sch(t_dt) ** 0.5
                tau_dot = torch.autograd.grad(tau_dt.sum(), t_dt, create_graph=False)[0]
            elif self.disc_type == "avg":
                t_next.requires_grad = True
                sqrt_gamma_next = self.snr_sch(t_next) ** 0.5
                sqrt_gamma_dot_next = torch.autograd.grad(
                    sqrt_gamma_next.sum(), t_next, create_graph=False
                )[0]
                sqrt_gamma_dot = 0.5 * (sqrt_gamma_dot + sqrt_gamma_dot_next)
                tau_next = self.scale_sch(t_next) ** 0.5
                tau_dot_next = torch.autograd.grad(
                    tau_next.sum(), t_next, create_graph=False
                )[0]
                tau_dot = 0.5 * (tau_dot + tau_dot_next)
                t_next.requires_grad = False
            elif self.disc_type == "difference":
                sqrt_gamma_next = self.snr_sch(t_next) ** 0.5
                sqrt_gamma_dot = (sqrt_gamma_next - sqrt_gamma) / (t_next - t)
                tau_next = self.scale_sch(t_next) ** 0.5
                tau_dot = (tau_next - tau) / (t_next - t)
            t.requires_grad = False
            sqrt_gamma = sqrt_gamma.detach()
            sqrt_gamma_dot = sqrt_gamma_dot.detach()
            tau = tau.detach()
            tau_dot = tau_dot.detach()
            # compute drift and diffusion coefficients
            const_scale_drift = sqrt_gamma_dot / (sqrt_gamma * (1 + sqrt_gamma**2.0))
            drift = x_t * ((tau_dot / tau) + const_scale_drift)
            diffusion = -(2 * tau**2.0 * const_scale_drift)
        else:
            # compute SNR and its derivative with respect to time using autograd
            t.requires_grad = True
            gamma = self.snr_sch(t)
            log_sqrt_gamma = 0.5 * torch.log(gamma)
            log_sqrt_gamma_dot = torch.autograd.grad(
                log_sqrt_gamma.sum(), t, create_graph=False
            )[0]
            tau_2 = self.scale_sch(t)
            log_tau = 0.5 * torch.log(tau_2)
            log_tau_dot = torch.autograd.grad(log_tau.sum(), t, create_graph=False)[0]
            if self.disc_type == "midpoint":
                dt = t_next - t
                t_dt = t + dt / 2
                log_sqrt_gamma_dt = 0.5 * torch.log(self.snr_sch(t_dt))
                log_sqrt_gamma_dot = torch.autograd.grad(
                    log_sqrt_gamma_dt.sum(), t_dt, create_graph=False
                )[0]
                log_tau_dt = 0.5 * torch.log(self.scale_sch(t_dt))
                log_tau_dot = torch.autograd.grad(
                    log_tau_dt.sum(), t_dt, create_graph=False
                )[0]
            elif self.disc_type == "avg":
                t_next.requires_grad = True
                log_sqrt_gamma_next = 0.5 * torch.log(self.snr_sch(t_next))
                log_sqrt_gamma_dot_next = torch.autograd.grad(
                    log_sqrt_gamma_next.sum(), t_next, create_graph=False
                )[0]
                log_sqrt_gamma_dot = 0.5 * (
                    log_sqrt_gamma_dot + log_sqrt_gamma_dot_next
                )
                log_tau_next = 0.5 * torch.log(self.scale_sch(t_next))
                log_tau_dot_next = torch.autograd.grad(
                    log_tau_next.sum(), t_next, create_graph=False
                )[0]
                log_tau_dot = 0.5 * (log_tau_dot + log_tau_dot_next)
                t_next.requires_grad = False
            elif self.disc_type == "difference":
                log_sqrt_gamma_next = 0.5 * torch.log(self.snr_sch(t_next))
                log_sqrt_gamma_dot = (log_sqrt_gamma_next - log_sqrt_gamma) / (
                    t_next - t
                )
                log_tau_next = 0.5 * torch.log(self.scale_sch(t_next))
                log_tau_dot = (log_tau_next - log_tau) / (t_next - t)
            t.requires_grad = False
            gamma = gamma.detach()
            log_sqrt_gamma = log_sqrt_gamma.detach()
            log_sqrt_gamma_dot = log_sqrt_gamma_dot.detach()
            tau_2 = tau_2.detach()
            log_tau = log_tau.detach()
            log_tau_dot = log_tau_dot.detach()
            # compute drift and diffusion coefficients
            const_scale_drift = log_sqrt_gamma_dot / (1 + gamma)
            drift = x_t * (log_tau_dot + const_scale_drift)
            diffusion = -(2 * tau_2 * const_scale_drift)
        diffusion = torch.sqrt(diffusion)
        return drift, diffusion

    def perturbation_kernel(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the mean and std of the Gaussian perturbation kernel
        p(x_t|x_0) = N(mean(x_0,t),std(t)) to diffuse x_0 ~ p_data.
        Args:
            x_0: input tensor x_0 ~ p_data to be diffused.
            t: time steps.
        """
        t = t.to(self.dtype)
        #x_0, t = _check_shapes(x_0, t)
        gamma = self.snr_sch(t)
        tau = self.scale_sch(t)
        a_2 = (tau * gamma) / (1 + gamma)
        b_2 = a_2 / gamma
        mean = x_0.to(self.dtype) * a_2**0.5
        std = torch.ones_like(x_0, dtype=self.dtype) * b_2**0.5
        return mean, std
        
    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Computes the signal-to-noise ratio (SNR) at time t.
        Args:
            t: time steps.
        Returns:
            SNR at time t.
        """
        return self.snr_sch(t)
