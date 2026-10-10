import logging
from functools import partial
from typing import Callable

import torch
import numpy as np

logger = logging.getLogger(__name__)


def ve_schedule(
    i: torch.Tensor, sigma_min: float = 0.002, sigma_max: float = 30.0
) -> torch.Tensor:
    """
    Original Variance Exploding schedule from Song et al. 2021
    https://arxiv.org/abs/2011.13456.

    Args:
        i: index of time step in [0.,1.].
        sigma_min: minimum std.
        sigma_max: maximum std.
    """
    return sigma_min**2 * (sigma_max**2 / sigma_min**2) ** i


def inverse_ve_schedule(
    sigma_2: torch.Tensor, sigma_min: float = 0.002, sigma_max: float = 30.0
) -> torch.Tensor:
    """
    Inverse of the original Variance Exploding schedule from Song et al. 2021

    Args:
        sigma: std.
        sigma_min: minimum std.
        sigma_max: maximum std.
    """
    r = (sigma_max**2) / (sigma_min**2)
    return torch.log(sigma_2 / (sigma_min**2)) / np.log(r)


def kve_schedule(
    i: torch.Tensor, sigma_min: float = 0.002, sigma_max: float = 30.0, rho: float = 7.0
) -> torch.Tensor:
    """
    Variance Exploding schedule from Karras et al. 2022
    https://arxiv.org/abs/2206.00364.

    Args:
        i: index of time step in [0.,1.].
        sigma_min: minimum std.
        sigma_max: maximum std.
        rho: steepness of exponential variance.
    """
    return (
        sigma_max ** (1 / rho)
        + (1.0 - i) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho


def inverse_kve_schedule(
    sigma: torch.Tensor,
    sigma_min: float = 0.002,
    sigma_max: float = 30.0,
    rho: float = 7.0,
) -> torch.Tensor:
    A = sigma_max ** (1 / rho)
    B = sigma_min ** (1 / rho)
    return 1 - (sigma ** (1 / rho) - A) / (B - A)


class TimeSchedule:
    """
    Base class for time schedules used for SDE diffusion.
    Note: Not to be confused with the index of the current latent distribution!
    """

    def __init__(
        self,
        time_fn: Callable,
        discretize: bool,
        T: int,
        dtype: torch.dtype = torch.float64,
    ):
        """
        Args:
            time_fn: function to compute the time step at index i in [0.,1.].
            discretize: if True, discretize the time schedule to T steps.
            T: number of discretization steps. Not used if discretize=False.
            dtype: data type to use for computation accuracy.
        """
        self.time_fn = time_fn
        self.discretize = discretize
        self.T = T
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

        self.time_steps = None
        self.sigmas = None

        # pre-compute the parameters using double precision
        if self.discretize:
            self.pre_compute_schedule()

    def pre_compute_sigmas(self, time_steps: torch.Tensor) -> torch.Tensor:
        """
        Pre-compute sigmas from pre-computed time steps.
        """
        raise NotImplementedError

    def pre_compute_schedule(self):
        """
        Pre-compute the time schedule.
        """
        # compute time steps and save to cpu to save gpu memory
        t = torch.arange(self.T, dtype=torch.float64, device="cpu")
        self.time_steps = self.time_fn(t / (self.T - 1)).to(self.dtype).to("cpu")
        self.sigmas = self.pre_compute_sigmas(self.time_steps).to(self.dtype).to("cpu")

    def get_sigma(self, i: torch.Tensor) -> torch.Tensor:
        """
        Get the standard deviation at index i.

        Args:
            i: index of time step in [0.,1.].
                or integer index if discretized schedule used,
                starting at 0 for diffusion step 1 until T-1.
        """
        if not self.discretize:
            raise ValueError("Sigma only available for discretized schedules.")

        if self.sigmas is None:
            raise ValueError("Sigmas not pre-computed.")

        return self.sigmas[i.to("cpu")].to(i.device)

    def __call__(self, i: torch.Tensor) -> torch.Tensor:
        """
        Get the time step at index i.

        Args:
            i: index of time step in [0.,1.].
                or integer index if discretized schedule used,
                starting at 0 for diffusion step 1 until T-1.
        """
        if not isinstance(i, torch.Tensor):
            raise ValueError("i must be a torch.Tensor.")

        device = i.device

        if len(i.shape) == 0:
            i = i.reshape(1)

        # use continous schedule
        if not self.discretize:
            if (
                i.dtype not in [torch.float, torch.double]
                or (i < 0.0).any()
                or (i > 1.0).any()
            ):
                raise ValueError(
                    "i must be a float or double in [0.,1.] if continous schedule used."
                )
            return self.time_fn(i.to(torch.float64)).to(self.dtype)

        # query pre-computed discretized schedule
        else:
            # convert to integer and numpy
            if i.dtype in [torch.float, torch.double]:
                i = torch.round(i.to(torch.float64) * (self.T - 1)).long()

            # check if out of bounds
            if (i < 0).any() or (i >= self.T).any():
                raise ValueError(
                    f"i must be between 0 and T-1. This may be due to rounding errors."
                    f" Got {i.min()} and {i.max()}."
                )
            return self.time_steps[i.to("cpu")].to(device)  # type: ignore


class KVeSchedule(TimeSchedule):
    """
    Wrapper class for ``kve_schedule``.
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 30.0,
        rho: float = 7.0,
        **kwargs,
    ):
        """
        sigma_min: minimum std.
        sigma_max: maximum std.
        rho: steepness of exponential variance.
        """
        super().__init__(
            time_fn=partial(
                kve_schedule, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho
            ),
            **kwargs,
        )

    def pre_compute_sigmas(self, time_steps: torch.Tensor) -> torch.Tensor:
        """
        Pre-compute sigmas from pre-computed time steps.
        """
        return time_steps.clone()
