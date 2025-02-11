from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import torch

from .functional import (
    _check_shapes,
    sample_isotropic_Gaussian,
)


class ForwardDiffusion(ABC):
    """
    Abstract base class to define the forward diffusion processes.
    """

    def __init__(
        self,
        invariant: bool = False,
        dtype: torch.dtype = torch.float64,
    ):
        """
        Args:
            invariant: invariant: if True, apply invariance constraints for symmetries.
                        e.g. For atoms positions this would be to force a zero CoG.
            dtype: data type to use for computational accuracy.
        """
        self.invariant = invariant
        self.dtype = dtype

        if isinstance(dtype, str):
            if dtype == "float64":
                self.dtype = torch.float64
            elif dtype == "float32":
                self.dtype = torch.float32
            else:
                raise ValueError(f"data type must be float32 or float64, got {dtype}")

    @abstractmethod
    def sample_prior(
        self, x: torch.Tensor, idx_m: Optional[torch.Tensor], **kwargs
    ) -> torch.Tensor:
        """
        Samples from the prior distribution p(x_T).

        Args:
            x: input tensor, e.g. to infer shape.
            idx_m: same as ``proporties.idx_m`` to map each row to its system.
            **kargs: additional keyword arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def diffuse(
        self,
        x_0: torch.Tensor,
        idx_m: Optional[torch.Tensor],
        t: torch.Tensor,
        return_dict: bool = False,
        output_key: str = "x_t",
        **kwargs,
    ) -> Any:
        """
        Diffuses origin x_0 by t steps to sample x_t from p(x_t|x_0),
        given x_0 ~ p_data.

        Args:
            x_0: input tensor x_0 ~ p_data to be diffused.
            idx_m: same as ``proporties.idx_m`` to map each row to its system.
                Set to None if one system or no invariance needed.
            t: time steps.
            return_dict: if True, return results under a dictionary of tensors.
            output_key: key to store the diffused x_t.
            kwargs: additional keyword arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def reverse(self, *args, **kwargs) -> "ReverseDiffusion":
        """
        Reverses the forward diffusion process.
        """
        raise NotImplementedError

    def __call__(
        self,
        x_0: torch.Tensor,
        idx_m: Optional[torch.Tensor],
        t: torch.Tensor,
        **kwargs,
    ) -> Any:
        """
        Defines the default call method.
        Currently equivalent to calling ``self.diffuse``.

        Args:
            x_0: input tensor x_0 ~ p_data to be diffused.
            idx_m: same as ``proporties.idx_m`` to map each row of x to its system.
                Set to None if one system or no invariance needed.
            t: time steps.
            kwargs: additional keyword arguments.
        """
        return self.diffuse(x_0, idx_m, t, **kwargs)


class ReverseDiffusion(ABC):
    """
    Abstract base class to define the reverse diffusion processes.
    """

    def __init__(
        self,
        forward_process: ForwardDiffusion,
        invariant: bool = False,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            forward_process: the forward diffusion process.
            invariant: invariant: if True, apply invariance constraints for symmetries.
                        e.g. For atoms positions this would be to force a zero CoG.
            dtype: data type to use for computation accuracy.
            device: device to use for computation.

        """
        self.forward_process = forward_process
        self.invariant = invariant
        self.dtype = dtype

        # set default device
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if isinstance(self.dtype, str):
            if dtype == "float64":
                self.dtype = torch.float64
            elif dtype == "float32":
                self.dtype = torch.float32
            else:
                raise ValueError(f"data type must be float32 or float64, got {dtype}")

    @abstractmethod
    def step(
        self, x_t: torch.Tensor, idx_m: Optional[torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        """
        Performs one reverse diffusion step to sample x_t-i ~ p(x_t-i|x_t), for i > 0.

        Args:
            x_t: input tensor x_t ~ p_t at diffusion step t.
            idx_m: same as ``proporties.idx_m`` to map each row of x to its system.
                Set to None if one system or no invariance needed.
            *args: additional positional arguments for subclasses.
            **kwargs: additional keyword arguments for subclasses.
        """
        raise NotImplementedError

    def sample_prior(
        self, x: torch.Tensor, idx_m: Optional[torch.Tensor], **kwargs
    ) -> torch.Tensor:
        """
        Samples from the prior distribution p(x_T).

        Args:
            x: dummpy input tensor, e.g. to infer shape.
            idx_m: same as ``proporties.idx_m`` to map each row to its system.
            **kargs: additional keyword arguments.
        """
        return self.forward_process.sample_prior(x, idx_m, **kwargs).to(
            self.device, self.dtype
        )

    def __call__(
        self, x_t: torch.Tensor, idx_m: Optional[torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        """
        Defines the default call method.
        Currently equivalent to calling ``self.step``.

        Args:
            x_t: input tensor x_t ~ p_t at diffusion step t.
            idx_m: same as ``proporties.idx_m`` to map each row of x to its system.
                Set to None if one system or no invariance needed.
            *args: additional positional arguments.
            **kwargs: additional keyword arguments.
        """
        return self.step(x_t, idx_m, *args, **kwargs)


class GaussianDiffusion(ForwardDiffusion):
    """
    Base class for diffusion models using Gaussian diffusion kernels.
    """

    def __init__(
        self,
        noise_key: str = "eps",
        **kwargs,
    ):
        """
        Args:
            noise_key: key to store the Gaussian noise.
            kwargs: additional arguments to be passed to ForwardDiffusion.__init__.
        """
        super().__init__(**kwargs)
        self.noise_key = noise_key

    @abstractmethod
    def perturbation_kernel(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        get the mean and std of the Gaussian perturbation kernel
        p(x_t|x_0) = N(mean(x_0,t),std(t)).

        Args:
            x_0: input tensor x_0 ~ p_data(x_0) to be diffused.
            t: time step.
        """
        raise NotImplementedError

    @abstractmethod
    def prior(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the mean and std of the prior distribution p(x_T) = N(mean,std).

        Args:
            x: dummy input tensor, e.g. to infer shape.
            **kargs: additional keyword arguments.
        """
        raise NotImplementedError

    def sample_prior(
        self, x: torch.Tensor, idx_m: Optional[torch.Tensor], **kwargs
    ) -> torch.Tensor:
        """
        Samples from the prior distribution p(x_T) = N(mean,std).

        Args:
            x: dummy input tensor, e.g. to infer shape.
            idx_m: same as ``proporties.idx_m`` to map each row to its system.
            **kargs: additional keyword arguments.
        """
        x = x.to(self.dtype)

        # get the mean and std of the prior.
        mean, std = self.prior(x, **kwargs)

        # sample from the prior.
        x_T, _ = sample_isotropic_Gaussian(
            mean, std, invariant=self.invariant, idx_m=idx_m, **kwargs
        )

        return x_T

    def diffuse(
        self,
        x_0: torch.Tensor,
        idx_m: Optional[torch.Tensor],
        t: torch.Tensor,
        return_dict: bool = False,
        output_key: str = "x_t",
        std_key: str = "sigma",
        **kwargs,
    ) -> Union[
        Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """
        Diffuses origin x_0 by t steps to sample x_t from p(x_t|x_0),
        given x_0 ~ p_data. Return tuple of tensors x_t and noise,
        or Dict of tensors with x_t and other quantities of interest.

        Args:
            x_0: input tensor x_0 ~ p_data to be diffused.
            idx_m: same as ``proporties.idx_m`` to map each row to its system.
                Set to None if one system or no invariance needed.
            t: time steps.
            return_dict: if True, return results under a dictionary of tensors.
            output_key: key to store the diffused x_t.
            std_key: key to store the standard deviation of the diffusion kernel.
            kwargs: additional keyword arguments.
        """
        # convert to correct data type.
        x_0 = x_0.to(self.dtype)

        x_0, t = _check_shapes(x_0, t)

        # query noise parameters.
        mean, std = self.perturbation_kernel(x_0, t)

        # sample by Gaussian diffusion.
        x_t, noise = sample_isotropic_Gaussian(
            mean, std, invariant=self.invariant, idx_m=idx_m, **kwargs
        )

        if return_dict:
            return {self.noise_key: noise, output_key: x_t, std_key: std}
        else:
            return x_t, noise, std
