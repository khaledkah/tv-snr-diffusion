from abc import abstractmethod
from functools import partial
import torch

from .noise_schedules import NoiseSchedule
from .time_schedules import ve_schedule, kve_schedule


class SNRSchedule:
    """
    Base class for SNR schedules.
    """

    def __init__(
        self,
        log_gamma_max: float,
        log_gamma_min: float,        
        t_min: float = 0.0,
        t_max: float = 1.0,
        dtype: torch.dtype = torch.float64,
    ):
        self.log_gamma_max = log_gamma_max
        self.log_gamma_min = log_gamma_min
        self.t_min = t_min
        self.t_max = t_max

        if isinstance(dtype, str):
            if dtype == "float64":
                self.dtype = torch.float64
            elif dtype == "float32":
                self.dtype = torch.float32
            else:
                raise ValueError(f"data type must be float32 or float64, got {dtype}")
        else:
            self.dtype = dtype

    @abstractmethod
    def log_gamma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the log SNR at time t.

        Args:
            t: time.

        Returns:
            log_snr: log signal-to-noise ratio.
        """
        raise NotImplementedError

    def clip_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        Clip time t if necessary. Default is no clipping.

        Args:
            t: time.

        Returns:
            t: clipped time.
        """
        return t

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the SNR at time t.

        Args:
            t: time.

        Returns:
            snr: signal-to-noise ratio.
        """
        t = t.to(self.dtype)
        t = t * (self.t_max - self.t_min) + self.t_min
        return torch.exp(self.log_gamma(t))

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return self.forward(t)


def exp_inverse_sigmoid_slope(
    timesteps: torch.Tensor,
    slope: float,
    shift: float,
):
    return torch.log(1/timesteps - 1) * slope + shift

class InverseSigmoid(SNRSchedule):
    """
    Inverse sigmoid SNR schedule.
    """
    def __init__(
        self,
        slope: float,
        shift: float,
        t_min: float = 0.0,
        t_max: float = 1.0,
        **kwargs,
    ):
        self.slope = slope
        self.shift = shift
        log_gamma_min = self.log_gamma(torch.tensor(t_max)).item()
        log_gamma_max = self.log_gamma(torch.tensor(t_min)).item()
        super().__init__(
            log_gamma_max=log_gamma_max,
            log_gamma_min=log_gamma_min,
            t_min=t_min,
            t_max=t_max,
            **kwargs,
        )
    def log_gamma(self, t: torch.Tensor) -> torch.Tensor:
        return exp_inverse_sigmoid_slope(
            t, self.slope, self.shift
        )

class SigmaToSNRSchedule(SNRSchedule):
    """
    Noise std to SNR schedule.
    """

    def __init__(self, t_min: float = 0.0, t_max: float = 1.0, **kwargs):
        """
        Args:
            t_min: minimum time.
            t_max: maximum time.
            sigma_fn: function of the std schedule.
        """
        self.t_min = t_min
        self.t_max = t_max

        log_gamma_min = self.log_gamma(torch.tensor(t_max, dtype=torch.float64)).item()
        log_gamma_max = self.log_gamma(torch.tensor(t_min, dtype=torch.float64)).item()
        super().__init__(
            log_gamma_max=log_gamma_max,
            log_gamma_min=log_gamma_min,
            t_min=t_min,
            t_max=t_max,
            **kwargs,
        )

    @abstractmethod
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def clip_t(self, t: torch.Tensor) -> torch.Tensor:
        return torch.clamp(t, self.t_min, self.t_max)

    def log_gamma(self, t: torch.Tensor) -> torch.Tensor:
        # if (t < self.t_min).any() or (t > self.t_max).any():
        #     raise ValueError(
        #         f"Time t must be in [{self.t_min}, {self.t_max}], got {t}"
        #         f"You can clip the time with the clip_t() method beforehand."
        #     )

        return -2 * torch.log(self.sigma(t))


class NoiseToSNRSchedule(SNRSchedule):
    """
    Noise-to-SNR schedule.
    """

    def __init__(
        self,
        noise_schedule: NoiseSchedule,
        t_min: float = 1e-5,
        t_max: float = 1.0,
        **kwargs,
    ):
        """
        Args:
            gamma_max: maximum SNR at minimum time t_min.
            gamma_min: minimum SNR at maximum time t_max.
        """
        self.noise_schedule = noise_schedule
        self.t_min = t_min
        self.t_max = t_max

        log_gamma_min = self.log_gamma(torch.tensor(t_max)).item()
        log_gamma_max = self.log_gamma(torch.tensor(t_min)).item()

        super().__init__(
            log_gamma_max=log_gamma_max,
            log_gamma_min=log_gamma_min,
            t_min=t_min,
            t_max=t_max,
            **kwargs,
        )

    def clip_t(self, t: torch.Tensor) -> torch.Tensor:
        return torch.clamp(t, self.t_min, self.t_max)

    def log_gamma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the SNR at time t.

        Args:
            t: time.

        Returns:
            snr: signal-to-noise ratio.
        """
        # if (t < self.t_min).any() or (t > self.t_max).any():
        #     raise ValueError(
        #         f"Time t must be in [{self.t_min}, {self.t_max}], got {t}"
        #         f"You can clip the time with the clip_t() method beforehand."
        #     )

        alphas_bar = self.noise_schedule.alphas_bar_fn(t, False)
        log_gamma = torch.log(alphas_bar) - torch.log(1 - alphas_bar)
        return log_gamma


class LinearToSNRSchedule(SigmaToSNRSchedule):
    """
    Linear sigma to SNR schedule
    """

    def __init__(self, sigma_min=0.0, sigma_max=1.0, **kwargs):
        self.fn = partial(
            kve_schedule, sigma_min=sigma_min, sigma_max=sigma_max, rho=1
        )
        super().__init__(**kwargs)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.fn(t)


class StraightEstimatorToSNRSchedule(SigmaToSNRSchedule):
    """
    Linear sigma to SNR schedule
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return t / (1 - t)


class VeToSNRSchedule(SigmaToSNRSchedule):
    """
    SNR schedule from VE (Song et al. 2021).
    https://arxiv.org/abs/2011.13456.
    """

    def __init__(self, sigma_min: float = 0.002, sigma_max: float = 30.0, **kwargs):
        self.fn = partial(ve_schedule, sigma_min=sigma_min, sigma_max=sigma_max)
        super().__init__(**kwargs)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.fn(t) ** 0.5


class KveToSNRSchedule(SigmaToSNRSchedule):
    """
    SNR schedule from EDM-VE schedule (Karras et al. 2022).
    https://arxiv.org/abs/2206.00364.
    """

    def __init__(
        self, sigma_min: float = 0.002, sigma_max: float = 30.0, rho=7.0, **kwargs
    ):
        self.fn = partial(
            kve_schedule, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho
        )
        super().__init__(**kwargs)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.fn(t)
