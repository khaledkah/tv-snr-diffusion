from abc import abstractmethod

import torch
import numpy as np
from .snr_schedules import SNRSchedule, InverseSigmoid


class ScaleSchedule:
    """
    Base class for scale schedules.
    """

    def __init__(
        self,
        snr_schedule: SNRSchedule,
        dtype: torch.dtype = torch.float64,
    ):
        self.snr_schedule = snr_schedule
        self.t_min = snr_schedule.t_min
        self.t_max = snr_schedule.t_max
        self.max_scale = self.get_max_scale()

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
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the scale \tau at time t.

        Args:
            t: time.
        """
        raise NotImplementedError

    @abstractmethod
    def get_max_scale(self) -> float:
        """
        Compute the maximum scale.
        """
        raise NotImplementedError

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return self.forward(t)


class FMScale(ScaleSchedule):
    """
    Scale schedule for the Flow Matching model.
    """

    def __init__(
        self,
        snr_schedule: InverseSigmoid,
        eta = None,
        kappa = None,
        **kwargs,
    ):
        if eta is not None:
            # override the default value
            self.eta = eta
        else:
            self.eta = snr_schedule.slope

        if kappa is not None:
            # override the default value
            self.kappa = kappa
        else:
            self.kappa = snr_schedule.shift

        super(FMScale, self).__init__(snr_schedule, **kwargs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the scale \tau at time t.

        Args:
            t: time.
        """
        return (1 - t) ** self.eta + t**self.eta * np.exp(-self.kappa)

    def get_max_scale(self) -> float:
        return np.exp(-self.kappa)


class ConstantScale(ScaleSchedule):
    """
    Scale schedule for the Flow Matching model.
    """

    def __init__(
        self,
        snr_schedule: InverseSigmoid,
        **kwargs,
    ):
        super().__init__(snr_schedule, **kwargs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the scale \tau at time t.

        Args:
            t: time.
        """
        return torch.tensor(1) + 0 * t

    def get_max_scale(self) -> float:
        return torch.tensor(1)
