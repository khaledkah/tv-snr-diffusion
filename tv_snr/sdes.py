from typing import Tuple

import torch

from .noise_schedules import NoiseSchedule

from functools import partial

from abc import abstractmethod
from typing import Optional, Tuple

import torch

from .base import GaussianDiffusion
from .base import ReverseDiffusion
from .functional import _check_shapes, sample_noise_like


class SDE(GaussianDiffusion):
    """
    Abstarct base class for diffusion using stochastic differential equations (SDEs)
    as a general framework.
    First introduced by Song et al. 2021 (https://arxiv.org/abs/2011.13456).
    We follow the formulation of Karras et al. 2022 (https://arxiv.org/abs/2206.00364).
    """

    def __init__(
        self,
        **kwargs,
    ):
        """
        Args:
            kwargs: additional arguments to pass to GaussianDiffusion.__init__.
        """
        super().__init__(**kwargs)

    @abstractmethod
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
        raise NotImplementedError

    def step(
        self,
        x_t: torch.Tensor,
        idx_m: Optional[torch.Tensor],
        t: torch.Tensor,
        t_next: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulates the forward SDE for one step, i.e. sample from p(x_t+1|x_t).
        Discretizes SDE: x_t+1 = x_t + mu(x_t, t) dt + sigma(x_t, t) * noise * sqrt(dt).

        Args:
            x_t: input tensor x_t ~ p_t to be diffused.
            idx_m: same as ``proporties.idx_m`` to map each row of x to its system.
                    Set to None if one system or no invariance needed.
            t: current time steps.
            t_next: next time steps.
            noise: the Gaussian noise. If None, a new noise is sampled.
            kwargs: additional keyword arguments.
        """
        # convert to correct dtype
        x_t = x_t.to(self.dtype)
        t = t.to(self.dtype)
        t_next = t_next.to(self.dtype)

        x_t, t, t_next = _check_shapes(x_t, t, t_next)

        # get the time step size
        dt = t_next - t

        # get the drift and diffusion coefficients of the forward SDE
        drift, diffusion = self.coefficients(x_t, t, t_next)

        # sample noise
        if noise is None:
            noise = sample_noise_like(
                x_t, invariant=self.invariant, idx_m=idx_m, **kwargs
            )
        # sample from discretized SDE
        return x_t + drift * dt + diffusion * noise * torch.sqrt(dt), noise

    def reverse(self, stochastic: bool = True, **kwargs) -> "RevSDE":
        """
        Reverses the forward SDE to get the reverse SDE/ODE.

        Args:
            stochastic: if True, use the stochastic reverse SDE.
                        Otherwise use the deterministic reverse ODE.
            kwargs: additional keyword arguments.
        """
        # return the reverse SDE/ODE
        return RevSDE(self, stochastic=stochastic, invariant=self.invariant, **kwargs)


class RevSDE(ReverseDiffusion):
    """
    The reverse SDE/ODE of the forward SDE.
    """

    def __init__(
        self,
        forward_process: SDE,
        stochastic: bool = True,
        **kwargs,
    ):
        """
        Args:
            forward_process: the forward SDE.
            stochastic: if True, use the stochastic reverse SDE.
                        Otherwise use the deterministic reverse ODE.
            kwargs: additional arguments to pass to ReverseDiffusion.__init__.
        """
        super().__init__(forward_process, **kwargs)
        self.stochastic = stochastic
        self.forward_process = forward_process

    def coefficients(
        self,
        x_t: torch.Tensor,
        score: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the drift mu(x,t) and diffusion coefficient sigma(t) of the reverse SDE/ODE
        dx = [mu(x,t) - sigma(t)**2 * score(x_t)]dt + sigma(t)dW.
        Multiplies sigma(t)**2 with 0.5 if ODE is used, i.e. stochastic=False.

        Args:
            x_t: input tensor x_t ~ p_t to be denoised.
            t: time steps.
        """
        # convert to correct dtype
        x_t = x_t.to(self.dtype)
        score = score.to(self.dtype)
        t = t.to(self.dtype)

        # get the drift and diffusion coefficients of the forward SDE
        drift, diffusion = self.forward_process.coefficients(x_t, t, t_next)

        # compute the drift and diffusion coefficients of the reverse SDE/ODE
        score = diffusion**2 * score

        # use the deterministic reverse ODE cefficients if not stochastic
        if not self.stochastic:
            score *= 0.5
            diffusion = torch.zeros_like(diffusion, dtype=self.forward_process.dtype)

        return drift - score, diffusion

    def step(
        self,
        x_t: torch.Tensor,
        idx_m: Optional[torch.Tensor],
        score: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Samples from p(x_t-i|x_t), 0<i using the discretized reverse SDE/ODE:
        x_t-i = x_t + [mu(x_t, t) - sigma(t)**2 * score(x_t)]dt + sigma(x_t, t) * noise * sqrt(|dt|).
        Multiplies sigma(t)**2 with 0.5 if ODE is used, i.e. stochastic=False.

        Args:
            x_t: input tensor x_t to be denoised.
            idx_m: should be proporties.idx_m: to map each row of x_t to its system.
            t: current time steps.
            t_next: next time steps.
            kwargs: additional keyword arguments.
        """
        # convert to correct dtype and device
        x_t = x_t.to(self.device, self.dtype)
        score = score.to(self.device, self.dtype)
        t = t.to(self.device, self.dtype)
        t_next = t_next.to(self.device, self.dtype)

        x_t, t, t_next = _check_shapes(x_t, t, t_next)

        # get the time step size
        dt = t_next - t

        # get the drift and diffusion coefficients of the reverse SDE/ODE
        drift, diffusion = self.coefficients(x_t, score, t, t_next)

        # sample noise
        noise = sample_noise_like(x_t, invariant=self.invariant, idx_m=idx_m, **kwargs)

        # sample from discretized reverse SDE/ODE
        x_t_next = x_t + drift * dt + diffusion * noise * torch.sqrt(torch.abs(dt))

        return x_t_next


class KVeSDE(SDE):
    """
    SDE using variance exploding formulation from Karras et al. 2022
    (https://arxiv.org/abs/2206.00364)
    """

    def __init__(
        self,
        sigma_min: float,
        sigma_max: float,
        **kwargs,
    ):
        """
        Args:
            sigma_min: minimum std.
            sigma_max: maximum std.
            kwargs: additional arguments to pass to SDE.__init__.
        """
        super().__init__(**kwargs)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def prior(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the mean and std of the prior distribution p(x_T) = N(mean,std).

        Args:
            x: input tensor, e.g. to infer shape.
        """
        mean = torch.zeros_like(x, dtype=self.dtype)
        std = torch.ones_like(x, dtype=self.dtype) * self.sigma_max

        return mean, std

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

        drift = torch.zeros_like(x_t, dtype=self.dtype)
        diffusion = torch.ones_like(x_t, dtype=self.dtype) * (2.0 * t) ** 0.5

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
        mean = x_0.to(self.dtype)
        std = torch.ones_like(x_0, dtype=self.dtype) * t

        return mean, std


class VeSDE(SDE):
    """
    SDE using original variance exploding from Song et al. 2021
    https://arxiv.org/abs/2011.13456.
    """

    def __init__(
        self,
        sigma_min: float,
        sigma_max: float,
        **kwargs,
    ):
        """
        Args:
            sigma_min: minimum std.
            sigma_max: maximum std.
            kwargs: additional arguments to pass to SDE.__init__.
        """
        super().__init__(**kwargs)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def prior(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the mean and std of the prior distribution p(x_T) = N(mean,std).

        Args:
            x: input tensor, e.g. to infer shape.
        """
        mean = torch.zeros_like(x, dtype=self.dtype)
        std = torch.ones_like(x, dtype=self.dtype) * self.sigma_max

        return mean, std

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
        drift = torch.zeros_like(x_t, dtype=self.dtype)
        diffusion = torch.ones_like(x_t, dtype=self.dtype)

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

        mean = x_0.to(self.dtype)
        std = torch.ones_like(x_0, dtype=self.dtype) * (t**0.5)

        return mean, std


class MVpSDE(SDE):
    """
    Recover SDE for variance preserving from Markov formulation.
    """

    def __init__(
        self,
        noise_sch: NoiseSchedule,
        **kwargs,
    ):
        """
        Args:
            noise_sch: continious noise schedule for alphas_bar.
            kwargs: additional arguments to pass to SDE.__init__.
        """
        super().__init__(**kwargs)
        self._alpha_bar_fn = partial(noise_sch.alphas_bar_fn, discretize=False)
        self._inverse_alpha_bar_fn = noise_sch.inverse
        self._derivative_alpha_bar_fn = noise_sch.derivative

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """
        Gets the beta from the noise schedule.

        Args:
            t: time steps.
        """
        t = t.to(self.dtype)

        return -1 * self._derivative_alpha_bar_fn(t) / self._alpha_bar_fn(t)

    def prior(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the mean and std of the prior distribution p(x_T) = N(mean,std).

        Args:
            x: input tensor, e.g. to infer shape.
        """
        mean = torch.zeros_like(x, dtype=self.dtype)
        std = torch.ones_like(x, dtype=self.dtype)

        return mean, std

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

        beta = self.beta(t)
        drift = -0.5 * beta * x_t
        diffusion = torch.sqrt(beta)

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

        alpha_bar = self._alpha_bar_fn(t)
        mean = torch.sqrt(alpha_bar) * x_0
        std = torch.sqrt(1 - alpha_bar)

        return mean, std
