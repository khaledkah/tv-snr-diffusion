import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from ase import Atoms
from torch import nn
from tqdm import tqdm

from schnetpack import properties
from schnetpack import transform as trn
from tv_snr.base import ReverseDiffusion
from tv_snr.functional import scatter_mean
from tvsnr_mol.utils import compute_neighbors, create_inputs

logger = logging.getLogger(__name__)


class Sampler:
    """
    Base class for for sampling or denoising from diffusion models.
    """

    def __init__(
        self,
        reverse_process: ReverseDiffusion,
        denoiser: Union[nn.Module, str],
        noise_pred_key: str = "eps_pred",
        std_key: str = "sigma",
        conditional: bool = False,
        guidance_strength: float = 0.0,
        cond_key: Optional[str] = None,
        cutoff: float = 5.0,
        recompute_neighbors: bool = False,
        additional_keys: List[str] = [],
        save_progress: bool = False,
        progress_stride: int = 1,
        results_on_cpu: bool = True,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        """
        Args:
            reverse_process: The reverse diffusion process to sample from.
            denoiser: The denoiser to use for the reverse process.
            noise_pred_key: The key for the noise prediction in the model output.
            std_key: The key for the standard deviation in the model input.
            cutoff: The cutoff for the neighbor list.
            recompute_neighbors: Whether to recompute the neighbors at each step.
            save_progress: Whether to save the progress of the reverse process.
            progress_stride: The stride to save the progress.
            results_on_cpu: Whether to save the results on the CPU.
            device: The device to use for the sampling.
        """
        self.reverse_process = reverse_process
        self.denoiser = denoiser
        self.noise_pred_key = noise_pred_key
        self.std_key = std_key
        self.conditional = conditional
        self.guidance_strength = guidance_strength
        self.cond_key = cond_key
        self.cutoff = cutoff
        self.save_progress = save_progress
        self.progress_stride = progress_stride
        self.recompute_neighbors = recompute_neighbors
        self.results_on_cpu = results_on_cpu
        self.additional_keys = additional_keys

        if self.conditional and self.cond_key is None:
            raise ValueError(
                "The conditional key must be provided for conditional models."
            )

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if isinstance(self.denoiser, str):
            self.denoiser = torch.load(
                self.denoiser, map_location=self.device, weights_only=False
            ).eval()
        elif self.denoiser is not None:
            self.denoiser = self.denoiser.to(self.device).eval()

        if self.get_T() is None:
            raise ValueError("The schedule must be descritised during sampling.")

        # To save the progress of the reverse process (i.e. the reverse trajectory)
        self._trajs = {}

    def create_inputs(
        self,
        inputs: List[Union[torch.Tensor, Dict[str, torch.Tensor], Atoms]],
        additional_inputs: Optional[List[Dict[str, torch.Tensor]]] = None,
        transforms: Optional[List[trn.Transform]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepares and converts the inputs for the sampler.

        Args:
            inputs: the inputs to be converted to the sampler.
            additional_inputs: Optional additional inputs to append to each molecule.
        """
        return create_inputs(
            inputs,
            additional_inputs=additional_inputs,
            transforms=transforms,
            device=self.device,
        )

    def update_model(self, model: nn.Module):
        """
        Updates the denoiser model.

        Args:
            model: the new denoiser model.
        """
        self.denoiser = model

    def sample_prior(
        self,
        inputs: Dict[str, torch.Tensor],
        t: Optional[Union[int, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Samples the prior p(x_t) for the reverse diffusion process.
        It uses the forward diffusion process to diffuse the input data if t not None,
        otherwise it samples from the tractable prior p(x_T).

        Args:
            inputs: input data with x_0 for each target property.
            t: the start time step of the reverse process,
                starting at 0 for diffusion step 1 until T-1.
            **kwargs: additional arguments to pass to the reverse process.
        """
        # get x_0 from inputs
        try:
            x_0 = (
                inputs[f"original_{properties.R}"]
                if f"original_{properties.R}" in inputs
                else inputs[properties.R]
            )
        except KeyError:
            raise KeyError(
                f"Input data must contain the true x_0 property under '{properties.R}' "
                f"or 'original_{properties.R}' to be diffused if sampling from t < T, "
                f"or dummy input to infer shape if sampling from t = T."
            )

        # prior for t < T: diffuse using p(x_t | x_0)
        if t is not None:
            if isinstance(t, int):
                t = torch.tensor(t, device=self.device)
            elif not isinstance(t, torch.Tensor):
                raise ValueError("t must be a torch.Tensor or int when not None.")

            t = t.to(self.device)

            # get prior using the forward diffusion process
            x_t = self.reverse_process.forward_process.diffuse(
                x_0,
                inputs[properties.idx_m],
                t,
                return_dict=True,
                output_key="x_t",
                **kwargs,
            )["x_t"]

        # prior for t = T: sample from tractable p(x_T)
        else:
            x_t = self.reverse_process.sample_prior(
                x_0, inputs[properties.idx_m], **kwargs
            )

        outputs = {properties.R: x_t.to(device=self.device)}

        return outputs

    def _sampling_sanity_checks(self, batch: Dict[str, torch.Tensor]):
        """
        Perform sanity checks of the input data before starting sampling.

        Args:
            batch: the batch of data.
        """
        # if the starting positions are not centered for invariance
        if (
            scatter_mean(
                batch[properties.R], batch[properties.idx_m], batch[properties.n_atoms]
            ).mean()
            > 1e-5
        ):
            logger.warning(
                "The starting positions of the atoms are not centered."
                "This violates the invariance of the probability distribution."
            )

    def _prepare_inputs(
        self, inputs: Dict[str, torch.Tensor], start: Optional[int] = None
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """Prepare input data for processing."""
        # Default is t=T
        if start is None:
            start = self.get_T()

        if not (isinstance(start, int) and 1 <= start <= self.get_T()):
            raise ValueError(
                "t must be one int between 1 and T that indicates the starting step."
                "Sampling using different starting steps is not supported yet for DDPM."
            )

        # copy inputs to avoid inplace operations
        batch = {prop: val.clone().to(self.device) for prop, val in inputs.items()}

        # check if center of geometry is close to zero
        self._sampling_sanity_checks(batch)

        # set all atoms as neighbors and compute neighbors only once before starting.
        if not self.recompute_neighbors:
            batch = compute_neighbors(
                batch,
                fully_connected=True,
                device=self.device,
                additional_keys=self.additional_keys,
            )

        self._trajs = {}

        return batch, start

    def _save_trajectory(self, batch: Dict[str, torch.Tensor], i: int):
        """Save the reverse trajectory progress if required."""
        if self.save_progress and (i % self.progress_stride == 0):
            if not self._trajs:
                self._trajs[properties.R] = [batch[properties.R].cpu().float().clone()]
            else:
                self._trajs[properties.R].append(
                    batch[properties.R].cpu().float().clone()
                )

    def _prepare_outputs(
        self, batch: Dict[str, torch.Tensor], start: int
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """Prepare final output after denoising."""
        x_0 = {
            properties.R: (
                batch[properties.R].cpu()
                if self.results_on_cpu
                else batch[properties.R]
            )
        }

        num_steps = torch.full_like(
            batch[properties.n_atoms], start, dtype=torch.long, device="cpu"
        )

        trajs = {
            k: torch.cat([elem.unsqueeze(-1) for elem in elems], dim=-1)
            for k, elems in self._trajs.items()
        }

        return x_0, num_steps, trajs

    @abstractmethod
    def get_T(self) -> int:
        """
        Returns the number of steps of the descritised reverse process.
        """
        raise NotImplementedError

    @abstractmethod
    def get_sigma(
        self, inputs: Dict[str, torch.Tensor], curr_steps: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns the standard deviation for the given reverse/time step.
        """
        raise NotImplementedError

    @abstractmethod
    def get_sigmas(self) -> torch.Tensor:
        """
        Returns the standard deviations for all reverse/time steps.
        """
        raise NotImplementedError

    @torch.no_grad()
    def apply_guidance(
        self,
        inputs: Dict[str, torch.Tensor],
        cond_noise: torch.Tensor,
    ):
        if (
            self.conditional
            and self.guidance_strength > 0.0
            and inputs[self.cond_key].sum() > 0.0  # type: ignore
        ):
            # set unconditional state
            inputs[self.cond_key] = inputs[self.cond_key] * 0  # type: ignore

            # cast input to float for the denoiser
            inputs = {
                key: val.float() if val.dtype == torch.float64 else val
                for key, val in inputs.items()
            }

            uncond_noise = self.denoiser(inputs)[self.noise_pred_key]  # type: ignore
            noise_pred = (
                1 + self.guidance_strength
            ) * cond_noise - self.guidance_strength * uncond_noise
        else:
            noise_pred = cond_noise

        return noise_pred

    def model_inputs(self, inputs: Dict[str, torch.Tensor], curr_steps: torch.Tensor):
        """
        Update the model inputs before inference.
        """
        # cast input to float for the denoiser
        inputs = {
            key: val.float() if val.dtype == torch.float64 else val
            for key, val in inputs.items()
        }

        # get the std of the current marginal p_t and and add it to the model input
        inputs[self.std_key] = self.get_sigma(inputs, curr_steps).float()

        return inputs

    @torch.no_grad()
    def inference_step(
        self, inputs: Dict[str, torch.Tensor], i: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One inference step for the model to get the score prediction.

        Args:
            inputs: input data for noise prediction.
            curr_steps: the current iteration of the reverse process.
        """
        # broadcast the current step to the batch
        curr_steps = torch.full_like(
            inputs[properties.idx_m],
            fill_value=i,
            dtype=torch.long,
            device=self.device,
        )

        # prepare the model inputs
        mod_inputs = self.model_inputs(inputs, curr_steps)

        # forward pass through the denoiser
        model_out = self.denoiser(mod_inputs)  # type: ignore

        # fetch the noise prediction
        noise_pred = model_out[self.noise_pred_key]

        # guidance if required
        noise_pred = self.apply_guidance(mod_inputs, noise_pred)

        return noise_pred, curr_steps

    @abstractmethod
    def iter(
        self, batch: Dict[str, torch.Tensor], i: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one step of the reverse process.

        Args:
            batch: the input data for the reverse process.
        """
        raise NotImplementedError

    def denoise(
        self,
        inputs: Dict[str, torch.Tensor],
        start: Optional[int] = None,
        progress_bar: bool = True,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Denoise the input data using the reverse process.

        Args:
            inputs: input data for denoising.
            start: The time step to start denoising from. Default is the last step.
        """
        batch, start = self._prepare_inputs(inputs, start)

        for i in tqdm(reversed(range(start)), disable=not progress_bar):
            # update the neighbors list if required
            if self.recompute_neighbors:
                batch = compute_neighbors(
                    batch,
                    cutoff=self.cutoff,
                    device=self.device,
                    additional_keys=self.additional_keys,
                )

            # perform one reverse step
            x_t_next, _ = self.iter(batch, i)

            # save history if required. Before updating the batch with the new state.
            self._save_trajectory(batch, start - (i + 1))

            batch[properties.R] = x_t_next

        return self._prepare_outputs(batch, start)

    def __call__(
        self,
        *args,
        **kwargs,
    ) -> Any:
        """
        Defines the default call method.
        Currently equivalent to calling ``self.denoise``.
        """
        return self.denoise(*args, **kwargs)
