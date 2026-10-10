import logging
from typing import Dict, Optional, Tuple

import torch

import schnetpack.transform as trn
from schnetpack import properties
from schnetpack.transform.neighborlist import NeighborListTransform
from tv_snr.base import ForwardDiffusion
from tv_snr.functional import batch_center_systems
from tv_snr.snr import Scale_SNR_SDE


class TimeSampler:
    """
    Wrapper class for time sampling functions for training.
    """

    def __init__(self):
        """
        Args:
            sampler: the sampling function.
        """
        self.sampler = torch.rand

    def __call__(self, size: Tuple) -> torch.Tensor:
        """
        Args:
            size: the size of the sample.
        """
        return self.sampler(size)


class UniformContinuous(TimeSampler):
    """
    Continous Uniform sampling from U(a,b).
    """

    def __init__(self, a: float, b: float):
        """
        Args:
            a: lower bound.
            b: upper bound.
        """
        self.a = a
        self.b = b
        self.sampler = lambda size: (b - a) * torch.rand(size) + a


class BatchSubtractCenterOfMass(trn.Transform):
    """
    subsctract center of mass from input systems batchwise.
    """

    is_preprocessor: bool = False
    is_postprocessor: bool = True
    force_apply: bool = True

    def __init__(
        self,
        name: str = "eps_pred",
        dim: int = 3,
    ):
        """
        Args:
            name: name of the property to be centered.
            dim: number of dimensions of the property to be centered.
        """
        super().__init__()
        self.name = name
        self.dim = dim

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        forward pass of the transform.

        Args:
            inputs: dictionary of input tensors.
        """
        # check shapes
        if inputs[self.name].shape[1] < self.dim:
            raise ValueError(
                f"Property {self.name} has less than {self.dim} dimensions. "
                f"Cannot subtract center of mass."
            )

        # center batchwise
        if inputs[self.name].shape[-1] == self.dim:
            inputs[self.name] = batch_center_systems(
                inputs[self.name], inputs[properties.idx_m], inputs[properties.n_atoms]
            )
        # use the first dimensions if the property has more than 'dim' dimensions.
        else:
            x = inputs[self.name][:, : self.dim]
            h = inputs[self.name][:, self.dim :]
            x_cent = batch_center_systems(
                x, inputs[properties.idx_m], inputs[properties.n_atoms]
            )
            inputs[self.name] = torch.cat((x_cent, h), dim=-1).to(
                device=inputs[self.name].device
            )

        return inputs


class Diffuse(trn.Transform):
    """
    Wrapper class for diffusion process of molecular properties.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(
        self,
        diffuse_property: str,
        diffusion_process: ForwardDiffusion,
        time_sampler: TimeSampler = TimeSampler(),
        output_key: Optional[str] = None,
        std_key: str = "sigma",
        snr_key: str = "gamma",
        var_scaler: float = 1.0,
        atomwise: bool = True,
    ):
        """
        Args:
            diffuse_property: property to diffuse.
            diffusion_process: diffusion process to use for diffusion.
            time_sampler: time sampler to use for sampling the diffusion time step.
                            Default is uniform sampling in [0,1].
            output_key: key to store the diffused property.
                        if None, the diffuse_property key is used.
            std_key: key to save the standard deviation of the diffusion kernel.
            atomwise: if True, the same diffusion time step is repeated for each atom.
        """
        super().__init__()
        self.diffuse_property = diffuse_property
        self.diffusion_process = diffusion_process
        self.time_sampler = time_sampler
        self.output_key = output_key
        self.std_key = std_key
        self.snr_key = snr_key
        self.var_scaler = var_scaler

        # Sanity check
        if (
            not self.diffusion_process.invariant
            and self.diffuse_property == properties.R
        ):
            logging.error(
                "Diffusing atom positions R without invariant constraint"
                "(invariant=False) might lead to unexpected results."
            )

        # broadcast values to all atom.
        self.atomwise = atomwise

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Define the forward diffusion process.

        Args:
            inputs: dictionary of input tensors.
        """
        x_0 = inputs[self.diffuse_property]
        device = x_0.device

        outputs = {
            f"original_{self.diffuse_property}": x_0,
        }

        x_0 = x_0 / self.var_scaler**0.5

        # get the training time step.
        t = self.time_sampler((1,))
        t = t.to(device=device)  # type: ignore

        if t.dtype not in [torch.float32, torch.float64]:
            raise ValueError(
                f"Time step must be of type float32 or float64, but is {t.dtype}."
                f"When using noise schedule with integer time steps, "
                f"normalize with t/(T-1)"
            )

        # diffuse the property.
        tmp = self.diffusion_process.diffuse(
            x_0,
            idx_m=None,
            t=t,
            return_dict=True,
            output_key=self.output_key or self.diffuse_property,
            std_key=self.std_key,
        )

        outputs.update(tmp)
        var = outputs[self.std_key].flatten().unique()

        if len(var) > 1:
            raise ValueError(
                "Different variance for each atom is not supported in this transform!"
            )

        # broadcast values to all atom.
        if self.atomwise:
            var = var.repeat(inputs[properties.n_atoms])

        if isinstance(self.diffusion_process, Scale_SNR_SDE):
            gamma = self.diffusion_process.gamma(t)
            outputs[self.snr_key] = torch.ones_like(var) * gamma

        outputs[self.std_key] = var

        inputs.update(outputs)

        return inputs


class AllToAllNeighborList(NeighborListTransform):
    """
    Calculate a full neighbor list for all atoms in the system.
    Faster than other methods and useful for small systems.
    """

    def __init__(self):
        # pass dummy large cutoff as all neighbors are connceted
        super().__init__(cutoff=1e8)

    def _build_neighbor_list(self, Z, positions, cell, pbc, cutoff):
        n_atoms = Z.shape[0]
        idx_i = torch.arange(n_atoms).repeat_interleave(n_atoms)
        idx_j = torch.arange(n_atoms).repeat(n_atoms)

        mask = idx_i != idx_j
        idx_i = idx_i[mask]
        idx_j = idx_j[mask]

        offset = torch.zeros(n_atoms * (n_atoms - 1), 3, dtype=positions.dtype)
        return idx_i, idx_j, offset
