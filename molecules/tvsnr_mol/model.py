from abc import abstractmethod
from typing import Callable, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack as spk
import schnetpack.properties as properties


class VarScaling(nn.Module):
    """
    Abstract class for scaling the variance to a normalized interval.
    """

    def __init__(
        self,
        input_key: str = "sigma",
        output_key: str = "t",
        drop_input_key: bool = False,
        **kwargs,
    ):
        """
        Args:
            key: key of the input to scale.
        """
        super().__init__(**kwargs)
        self.input_key = input_key
        self.output_key = output_key
        self.drop_input_key = drop_input_key

    @abstractmethod
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def inverse(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SNRVarScaling(VarScaling):
    def __init__(
        self,
        input_key: str = "gamma",
        scale: float = -0.125,
        shift: float = 0.35,
        **kwargs,
    ):
        super().__init__(input_key=input_key, **kwargs)
        self.scale = scale
        self.shift = shift

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        inputs[self.output_key] = (
            self.scale * torch.log(inputs[self.input_key]) + self.shift
        )

        if self.drop_input_key:
            inputs.pop(self.input_key)

        return inputs

    def inverse(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp((t - self.shift) / self.scale)


class TimeAwareEquivariant(nn.Module):
    """
    Time-aware Equivariant head for diffusion noise.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        include_time: bool = False,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        output_key: str = "eps_pred",
        time_key: Optional[str] = None,
    ):
        """
        Args:
            n_in: input dimension without time.
            n_out: output dimension for the target property.
            include_time: whether to append time as input feature.
            n_hidden: size of hidden layers.
                    If an integer, same number of node is used for all hidden
                        layers resulting in a rectangular network.
                    If None, the number of neurons is divided
                        by two after each layer starting
                        n_in resulting in a pyramidal network.
            n_layers: number of hidden layers.
            activation: activation function.
            output_key: the key under which the result will be stored.
            time_key: time key for input to the equivariant module.
        """

        super().__init__()

        self.outnet = spk.nn.build_gated_equivariant_mlp(
            n_in=n_in,
            n_out=n_out,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
            sactivation=activation,
        )

        self.output_key = output_key
        self.model_outputs = [output_key]

        self.include_time = include_time
        # add time as input scalar feature
        if self.include_time:
            self.outnet[0] = spk.nn.GatedEquivariantBlock(
                n_sin=self.outnet[0].n_sin + 1,
                n_vin=self.outnet[0].n_vin,
                n_sout=self.outnet[0].n_sout,
                n_vout=self.outnet[0].n_vout,
                n_hidden=self.outnet[0].n_hidden,
                activation=activation,
                sactivation=activation,
            )

        self.time_key = time_key
        # set default time key
        if self.include_time and self.time_key is None:
            raise ValueError(
                "Argument 'time_key' must be set when 'include_time' is True."
            )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        l0 = inputs["scalar_representation"]
        l1 = inputs["vector_representation"]

        # append time to representation
        if self.include_time:
            t = inputs[self.time_key]  # type: ignore

            # broadcast molecule-level time to all atoms
            if len(t) != len(inputs[properties.idx_m]):
                t = t[inputs[properties.idx_m]]
            t = t.unsqueeze(-1)

            # append time to scalar representation features
            l0 = torch.cat((l0, t), dim=-1)

        # predict equivariant output
        _, out = self.outnet((l0, l1))
        out = torch.squeeze(out, -1)

        inputs[self.output_key] = out

        return inputs
