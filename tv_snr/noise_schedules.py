import logging
from functools import partial
from typing import Callable, Dict

import torch
from torch import nn

logger = logging.getLogger(__name__)


def clip_noise_schedule(
    alphas_bar: torch.Tensor, clip_min: float = 0.0, clip_max: float = 1.0
) -> torch.Tensor:
    """
    For a noise schedule given by alpha_bar, this clips alpha_t / alpha_t-1.
    This may help improve stability during sampling.

    Args:
        alphas_bar: noise schedule.
        clip_min: minimum value to clip to.
        clip_max: maximum value to clip to.
    """
    # get alphas from alphas_bar
    alphas = alphas_bar[1:] / alphas_bar[:-1]

    # clip for more stable noise schedule
    alphas = torch.clip(alphas, min=clip_min, max=clip_max)

    # recompute alphas_bar
    alphas_bar = torch.cumprod(alphas, dim=0)

    return alphas_bar


def polynomial_decay(
    timesteps: torch.Tensor,
    discretize: bool,
    s: float = 1e-5,
    power: float = 2.0,
    clip_value: float = 0.001,
) -> torch.Tensor:
    """
    A noise schedule based on a simple polynomial equation from
    https://arxiv.org/abs/2203.17003 to approximate the cosine schedule.

    Args:
        timesteps: tesnor with input timesteps in [0.,1.].
        discretize: if True, use the discretized schedule.
        s: precision parameter.
        power: power of the polynomial.
        clip_value: minimum value to clip to.
    """
    # compute alphas_bar
    alphas_bar = (1 - torch.pow(timesteps, power)) ** 2

    if discretize:
        # clip for more stable noise schedule
        alphas_bar = clip_noise_schedule(alphas_bar, clip_min=clip_value)

    # add precision
    precision = 1 - 2 * s
    alphas_bar = precision * alphas_bar + s

    return alphas_bar


def inverse_polynomial_decay(
    sqrt_beta_bar: torch.Tensor, s: float = 1e-5, power: float = 2.0
):
    """
    Inverse the polynomial schedule to recover the time step from the std,
    i.e. t = f^-1(sqrt(beta_bar)).

    Args:
        sqrt_beta_bar: the standard deviation of the diffusion perturbation kernel.
        s: precision parameter.
        power: power of the polynomial.
    """
    alpha_bar = 1 - sqrt_beta_bar.to(torch.float64) ** 2
    alpha_bar = (alpha_bar - s) / (1 - 2 * s)

    t = torch.pow(1 - torch.sqrt(torch.clip(alpha_bar, min=0.0)), 1.0 / power)

    return t.to(sqrt_beta_bar.dtype)


def derivative_polynomial_decay(
    timesteps: torch.Tensor, s: float = 1e-5, power: float = 2.0
):
    """
    Compute the derivative of the polynomial schedule at timestep t.

    Args:
        timesteps: tesnor with input timesteps in [0.,1.].
        s: precision parameter.
        power: power of the polynomial.
    """
    return (
        -2
        * power
        * (1 - 2 * s)
        * torch.pow(timesteps, power - 1.0)
        * (1 - torch.pow(timesteps, power))
    )


def cosine_decay(
    timesteps: torch.Tensor,
    discretize: bool,
    s: float = 0.008,
    v: float = 1.0,
    clip_value: float = 0.001,
) -> torch.Tensor:
    """
    Cosine schedule with clipping from https://arxiv.org/abs/2102.09672.

    Args:
        timesteps: tesnor with input timesteps in [0.,1.].
        discretize: if True, use the discretized schedule.
        s: precision parameter.
        v: decay parameter.
        clip_value: minimum value to clip to.
    """

    # compute alphas_bar
    def _decay_fn(t):
        return torch.cos((t**v + s) / (1 + s) * torch.pi * 0.5) ** 2

    f_t = _decay_fn(timesteps)
    f_0 = _decay_fn(torch.tensor(0.0))

    alphas_bar = f_t / f_0

    if discretize:
        # clip for more stable noise schedule
        alphas_bar = clip_noise_schedule(alphas_bar, clip_min=clip_value)

    return alphas_bar


def inverse_cosine_decay(sqrt_beta_bar: torch.Tensor, s: float = 0.008, v: float = 1.0):
    """
    Inverse the cosine schedule to recover the time step from the std,
    i.e. t = f^-1(sqrt(beta_bar)).

    Args:
        sqrt_beta_bar: the standard deviation of the diffusion perturbation kernel.
        s: precision parameter.
        v: decay parameter.
    """
    alpha_bar = 1 - sqrt_beta_bar.to(torch.float64) ** 2
    f_0 = torch.cos((torch.tensor(0) ** v + s) / (1 + s) * torch.pi * 0.5) ** 2
    t = torch.acos(torch.sqrt(alpha_bar * f_0)) * 2 * (1 + s) / torch.pi - s

    return t.to(sqrt_beta_bar.dtype)


def linear_decay(
    timesteps: torch.Tensor,
    discretize: bool,
    beta_start: float = 0.1,
    beta_end: float = 20.0,
) -> torch.Tensor:
    """
    Linear schedule from https://arxiv.org/pdf/2006.11239.pdf
    and reformulated to continuous time in https://arxiv.org/abs/2011.13456.

    Args:
        timesteps: tesnor with input timesteps in [0.,1.].
        beta_start: starting value of beta_t, i.e. t=0.
        beta_end: ending value of beta_t, i.e. t=T.
        discretize: if True, use the discretized schedule.
    """
    # compute alphas_bar
    alphas_bar = torch.exp(
        -0.5 * timesteps**2 * (beta_end - beta_start) - timesteps * beta_start
    )

    if discretize:
        # the first index 0 is related to alpha_bar_1 and alpha_bar_0 = 1
        alphas_bar = alphas_bar[1:]

    return alphas_bar


def inverse_linear_decay(
    sqrt_beta_bar: torch.Tensor, beta_start: float = 0.1, beta_end: float = 20.0
):
    """
    Inverse the linear schedule to recover the time step from the std,
    i.e. t = f^-1(sqrt(beta_bar)).

    Args:
        sqrt_beta_bar: the standard deviation of the diffusion perturbation kernel.
        beta_start: starting value of beta_t, i.e. t=0.
        beta_end: ending value of beta_t, i.e. t=T.
    """
    alpha_bar = 1 - sqrt_beta_bar.to(torch.float64) ** 2

    offset = beta_end - beta_start
    t = torch.sqrt(beta_start**2 - 2 * offset * torch.log(alpha_bar))
    t = (t - beta_start) / offset

    return t.to(sqrt_beta_bar.dtype)


class NoiseSchedule(nn.Module):
    """
    Base class for noise schedules. To be used together with Markovian processes,
    i.e. inheriting from ``MarkovianDiffusion``.
    """

    def __init__(
        self,
        alpha_bar_fn: Callable,
        discretize: bool,
        T: int,
        variance_type: str = "lower_bound",
        dtype: torch.dtype = torch.float64,
    ):
        """
        Args:
            alpha_bar_fn: function to compute alpha_bar at any index i in [0.,1.].
            discretize: if True, discretize the time schedule to T steps.
            T: number of discretization steps. Not used if discretize=False.
            variance_type: use either 'lower_bound' or the 'upper_bound'.
            dtype: torch dtype to use for computation accuracy.
        """
        super().__init__()

        self.alphas_bar_fn = alpha_bar_fn
        self.T = T
        self.discretize = discretize
        self.variance_type = variance_type
        self.dtype = dtype

        if self.discretize and self.T is None:
            raise ValueError("T must be set when using discretization.")

        if isinstance(dtype, str):
            if dtype == "float64":
                self.dtype = torch.float64
            elif dtype == "float32":
                self.dtype = torch.float32
            else:
                raise ValueError(f"data type must be float32 or float64, got {dtype}")

        if self.variance_type == "upper_bound":
            logger.warning(
                "The upper bound for the posterior variance is not the exact one. "
                "This may affect the NLL estimation if used."
            )

        # pre-compute the parameters using double precision
        if self.discretize:
            self.pre_compute_statistics()

    def pre_compute_statistics(self):
        """
        Pre-compute the noise parameters based on the notation of Ho et al.
        """
        timesteps = torch.linspace(
            0.0, 1.0, self.T + 1, dtype=torch.float64, device="cpu"
        )

        # save to cpu to save gpu memory
        self.alphas_bar = (
            self.alphas_bar_fn(timesteps, discretize=True).to(torch.float64).to("cpu")
        )

        self.betas_bar = 1 - self.alphas_bar
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_betas_bar = torch.sqrt(
            self.betas_bar
        )  # different from 1-sqrt(alphas_bar) !

        # infer the different statistics
        self.alphas = self.alphas_bar[1:] / self.alphas_bar[:-1]
        self.alphas = torch.concatenate([self.alphas_bar[:1], self.alphas])
        self.betas = 1.0 - self.alphas
        self.betas_square = self.betas**2
        self.sqrt_betas = torch.sqrt(self.betas)
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.inv_sqrt_alphas = 1.0 / self.sqrt_alphas
        self.inv_sqrt_betas_bar = 1.0 / self.sqrt_betas_bar

        # Weither to use the true posterior variance which is the lower bound
        # or the upper bound formulation
        if self.variance_type == "lower_bound":
            self.sigmas_square = self.betas[1:] * (
                self.betas_bar[:-1] / self.betas_bar[1:]
            )

            # lower bound sigma_1 = 0 because beta_bar_0 = 1 - alpha_bar_0 = 1 - 1 = 0
            # We clip to avoid inf when computing decoder loglikelihood or vlb weights
            self.sigmas_square = torch.concatenate(
                [self.sigmas_square[:1], self.sigmas_square]
            )

        elif self.variance_type == "upper_bound":
            self.sigmas_square = self.betas.clone()

            # we always replace the first value by the true posterior variacne
            # to have a better likelihood of L_0
            # see https://arxiv.org/abs/2102.09672
            self.sigmas_square[0] = self.betas[1] * (
                self.betas_bar[0] / self.betas_bar[1]
            )

        else:
            raise ValueError(
                "variance_type must be either 'lower_bound' or 'upper_bound'"
            )

        self.sigmas = torch.sqrt(self.sigmas_square)

        self.vlb_weights = self.betas**2 / (
            2 * self.sigmas_square * self.alphas * self.betas_bar
        )

    def forward(self, t: torch.Tensor, keys: list = []) -> Dict[str, torch.Tensor]:
        """
        Query the noise parameters at timestep t.

        Args:
            t: the query timestep, starting at 0 for diffusion step 1 until T-1.
        """
        if not isinstance(t, torch.Tensor):
            raise ValueError("t must be a torch.Tensor.")

        device = t.device

        if len(t.shape) == 0:
            t = t.reshape(1)

        if not self.discretize:
            if (
                t.dtype not in [torch.float, torch.double]
                or (t < 0.0).any()
                or (t > 1.0).any()
            ):
                raise ValueError(
                    "t must be a float or double in [0.,1.] if continous schedule used."
                )

            alpha_bar = self.alphas_bar_fn(t.to(torch.float64), discretize=False)

            # query the noise parameters
            key_to_attr_mapping = {
                "alpha_bar": alpha_bar,
                "beta_bar": 1 - alpha_bar,
                "sqrt_alpha_bar": torch.sqrt(alpha_bar),
                "sqrt_beta_bar": torch.sqrt(1 - alpha_bar),
            }

        else:
            # convert to integer and numpy
            if t.dtype in [torch.float, torch.double]:
                t = torch.round(t.to(torch.float64) * (self.T - 1)).long()

            t = t.to("cpu")

            # check if out of bounds
            if torch.any(t < 0) or torch.any(t >= self.T):
                raise ValueError(
                    "t must be between 0 and T-1. This may be due to rounding errors."
                )

            # query the noise parameters
            key_to_attr_mapping = {
                "alpha_bar": self.alphas_bar[t],
                "beta_bar": self.betas_bar[t],
                "sqrt_alpha_bar": self.sqrt_alphas_bar[t],
                "sqrt_beta_bar": self.sqrt_betas_bar[t],
                "alpha": self.alphas[t],
                "beta": self.betas[t],
                "sqrt_alpha": self.sqrt_alphas[t],
                "sqrt_beta": self.sqrt_betas[t],
                "beta_square": self.betas_square[t],
                "sigma_square": self.sigmas_square[t],
                "sigma": self.sigmas[t],
                "inv_sqrt_alpha": self.inv_sqrt_alphas[t],
                "inv_sqrt_beta_bar": self.inv_sqrt_betas_bar[t],
                "vlb_weight": self.vlb_weights[t],
            }

        # return all keys if not specified
        if not keys:
            keys = key_to_attr_mapping.keys()  # type: ignore

        # fetch parameters and convert to torch tensors
        params = {}
        for key in keys:
            if key in key_to_attr_mapping:
                params[key] = key_to_attr_mapping[key].to(device=device).to(self.dtype)
            else:
                raise KeyError(
                    f"Key {key} not recognized. "
                    "If using continuous schedule, only few keys are available"
                    " because of the need of the next timestep for the others."
                )

        return params

    def inverse(self, sqrt_beta_bar: torch.Tensor) -> torch.Tensor:
        """
        Inverse the noise schedule to recover the time step from the std,
        i.e. t = f^-1(sqrt(beta_bar)).

        Args:
            sqrt_beta_bar: the standard deviation of the diffusion perturbation kernel.
        """
        raise NotImplementedError("Inverse method not implemented.")

    def derivative(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the derivative of the noise schedule at timestep t.

        Args:
            t: the query timestep, starting at 0 for diffusion step 1 until T-1.
        """
        raise NotImplementedError("Derivative method not implemented.")


class CosineSchedule(NoiseSchedule):
    """
    Cosine noise schedule.
    Subclasses ``NoiseSchedule``.
    """

    def __init__(
        self,
        s: float = 0.008,
        v: float = 1.0,
        clip_value: float = 0.001,
        **kwargs,
    ):
        """
        Args:
            s: precision parameter.
            v: decay parameter.
            clip_value: clip parameetrs for numerical stability.
            kwargs: additional keyword arguments.
        """
        self.s = s
        self.v = v
        self.clip_value = clip_value

        alphas_bar_fn = partial(
            cosine_decay, s=self.s, v=self.v, clip_value=self.clip_value
        )

        super().__init__(
            alpha_bar_fn=alphas_bar_fn,
            **kwargs,
        )

    def inverse(self, sqrt_beta_bar: torch.Tensor) -> torch.Tensor:
        """
        Inverse the cosine schedule to recover the time step from the std,
        i.e. t = f^-1(sqrt(beta_bar)).

        Args:
            sqrt_beta_bar: the standard deviation of the diffusion perturbation kernel.
        """
        return inverse_cosine_decay(sqrt_beta_bar, s=self.s, v=self.v)


class PolynomialSchedule(NoiseSchedule):
    """
    Polynomial noise schedule.
    Subclasses ``NoiseSchedule``.
    """

    def __init__(
        self,
        s: float = 1e-5,
        power: float = 2.0,
        clip_value: float = 0.001,
        **kwargs,
    ):
        """
        Args:
            s: precision parameter.
            power: power of the polynomial.
            clip_value: clip parameetrs for numerical stability.
            kwargs: additional keyword arguments.
        """
        self.s = s
        self.power = power
        self.clip_value = clip_value

        alphas_bar_fn = partial(
            polynomial_decay, s=self.s, power=self.power, clip_value=self.clip_value
        )

        super().__init__(
            alpha_bar_fn=alphas_bar_fn,
            **kwargs,
        )

    def inverse(self, sqrt_beta_bar: torch.Tensor) -> torch.Tensor:
        """
        Inverse the polynomial schedule to recover the time step from the std,
        i.e. t = f^-1(sqrt(beta_bar)).

        Args:
            sqrt_beta_bar: the standard deviation of the diffusion perturbation kernel.
        """
        return inverse_polynomial_decay(sqrt_beta_bar, s=self.s, power=self.power)

    def derivative(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the derivative of the noise schedule at timestep t.

        Args:
            t: the query timestep, starting at 0 for diffusion step 1 until T-1.
        """
        return derivative_polynomial_decay(t, s=self.s, power=self.power)


class LinearSchedule(NoiseSchedule):
    """
    Linear noise schedule.
    Subclasses ``NoiseSchedule``.
    """

    def __init__(
        self,
        beta_start: float = 0.1,
        beta_end: float = 20.0,
        **kwargs,
    ):
        """
        Args:
            beta_start: starting value of beta_t, i.e. t=0.
            beta_end: ending value of beta_t, i.e. t=T.
            kwargs: additional keyword arguments.
        """
        self.beta_start = beta_start
        self.beta_end = beta_end

        alphas_bar_fn = partial(
            linear_decay, beta_start=self.beta_start, beta_end=self.beta_end
        )

        super().__init__(
            alpha_bar_fn=alphas_bar_fn,
            **kwargs,
        )

    def inverse(self, sqrt_beta_bar: torch.Tensor) -> torch.Tensor:
        """
        Inverse the linear schedule to recover the time step from the std,
        i.e. t = f^-1(sqrt(beta_bar)).
        """
        return inverse_linear_decay(
            sqrt_beta_bar, beta_start=self.beta_start, beta_end=self.beta_end
        )
