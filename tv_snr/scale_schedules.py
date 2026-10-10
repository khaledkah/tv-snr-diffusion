from abc import abstractmethod

import numpy as np
import torch

from .snr_schedules import InverseSigmoid, SNRSchedule


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
        orig_fm_scale: bool = False,
        **kwargs,
    ):
        if not isinstance(snr_schedule, InverseSigmoid):
            raise ValueError("FMScale requires an InverseSigmoid SNR schedule.")

        if orig_fm_scale:
            self.eta = 2.0
            self.kappa = 0.0
        else:
            self.eta = snr_schedule.slope
            self.kappa = snr_schedule.shift

        super(FMScale, self).__init__(snr_schedule, **kwargs)  # type: ignore

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the scale \tau at time t.

        Args:
            t: time.
        """
        return (1 - t) ** self.eta + t**self.eta * np.exp(-self.kappa)

    def get_max_scale(self) -> float:
        return np.exp(-self.kappa)


class ConstScale(ScaleSchedule):
    """
    Scale schedule for the Flow Matching model.
    """

    def __init__(
        self,
        snr_schedule: SNRSchedule,
        constant: float = 1.0,
        **kwargs,
    ):
        super(ConstScale, self).__init__(snr_schedule, **kwargs)
        self.constant = torch.tensor(constant).to(self.dtype)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the scale \tau at time t.

        Args:
            t: time.
        """
        return t * 0.0 + self.constant

    def get_max_scale(self) -> float:
        return self.constant.item()


class VeScale(ScaleSchedule):
    """
    Scale schedule for the Flow Matching model.
    """

    def __init__(
        self,
        snr_schedule: SNRSchedule,
        **kwargs,
    ):
        super(VeScale, self).__init__(snr_schedule, **kwargs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the scale \tau at time t.

        Args:
            t: time.
        """
        return 1.0 / self.snr_schedule(t) + 1

    def get_max_scale(self) -> float:
        return 1.0 / np.exp(self.snr_schedule.log_gamma_min)
