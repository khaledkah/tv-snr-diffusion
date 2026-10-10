import inspect
import logging
from typing import Dict, List, Optional

import torch
from torch import nn

from schnetpack.model.base import AtomisticModel
from schnetpack.task import AtomisticTask, ModelOutput, UnsupervisedModelOutput

log = logging.getLogger(__name__)


class DiffModelOutput(ModelOutput):
    """
    define diffusion output head.
    """

    def calculate_loss(
        self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        calculate the loss.

        Args:
            pred: outputs.
            target: target values.
        """
        if self.loss_weight == 0 or self.loss_fn is None:
            return torch.tensor(0.0)

        # extract the extra arguments of the loss function if needed
        args_ = inspect.getfullargspec(self.loss_fn).args[2:]
        kwargs = {k: pred[k] for k in args_ if k in pred}

        # calculate the loss using the extra arguments if needed
        if kwargs:
            loss = self.loss_weight * self.loss_fn(
                pred[self.name], target[self.target_property], **kwargs
            )
        else:
            loss = self.loss_weight * self.loss_fn(
                pred[self.name], target[self.target_property]
            )

        return loss


class DiffusionTask(AtomisticTask):
    """
    Defines the diffusion task for pytorch lightning.
    """

    def __init__(
        self,
        model: AtomisticModel,
        outputs: List[ModelOutput],
        diffuse_property: str,
        skip_exploding_batches: bool = True,
        time_key: str = "diff_step",
        noise_key: str = "eps",
        noise_pred_key: str = "eps_pred",
        **kwargs,
    ):
        """
        Args:
            diffuse_property: property to diffuse.
            skip_exploding_batches: ignore exploding batches during training.
            time_key: key of the true diffusion time step in the input dictionary.
            noise_key: key of the true noise in the input dictionary.
            noise_pred_key: key of the predicted noise in the output dictionary.
        """
        super().__init__(model=model, outputs=outputs, **kwargs)

        self.diffuse_property = diffuse_property
        self.skip_exploding_batches = skip_exploding_batches
        self.time_key = time_key
        self.noise_key = noise_key
        self.noise_pred_key = noise_pred_key

    def setup(self, stage=None):
        """
        overwrite the pytorch lightning task setup function.
        """
        # call the parent atomistic task setup
        AtomisticTask.setup(self, stage=stage)  # type: ignore

        # force some post-processing transforms during training
        forced_postprocessors = []
        for pp in self.model.postprocessors:
            if hasattr(pp, "force_apply"):
                if pp.force_apply:
                    forced_postprocessors.append(pp)
        self.model.forced_postprocessors = nn.ModuleList(forced_postprocessors)

    def predict_without_postprocessing(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        predict without post-processing transforms.
        Note: forced post-processing transforms will still be applied.

        Args:
            batch: input batch.
        """
        tmp_postprocessors = self.model.postprocessors
        self.model.postprocessors = self.model.forced_postprocessors
        pred = self(batch)
        self.model.postprocessors = tmp_postprocessors

        return pred

    def _step(self, batch: Dict[str, torch.Tensor], subset: str) -> torch.FloatTensor:
        """
        perform one forward pass and calculate the loss and log metrics.

        Args:
            batch: input batch.
            subset: the dataset split used.
        """
        # predict output quantity
        pred = self.predict_without_postprocessing(batch)

        # extract the target values from the batch
        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }
        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except Exception:
            pass

        # apply constraints
        pred, targets = self.apply_constraints(pred, targets)

        # calculate the loss
        loss = self.loss_fn(pred, targets)

        # log loss and metrics
        self.log(
            f"{subset}_loss",
            loss,
            on_step=(subset == "train"),
            on_epoch=(subset != "train"),
            prog_bar=(subset != "train"),
        )
        self.log_metrics(pred, targets, subset)

        return loss  # type: ignore

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Optional[torch.FloatTensor]:
        """
        define the training step for pytorch lightning.

        Args:
            batch: input batch.
            batch_idx: batch index.
        """
        # perform forward pass
        loss = self._step(batch, "train")

        # skip exploding batches in backward pass
        if self.skip_exploding_batches and (
            torch.isnan(loss) or torch.isinf(loss) or loss > 1e10
        ):
            log.warning(
                f"Loss is {loss} for train batch_idx {batch_idx} and training step "
                f"{self.global_step}, training step will be skipped!"
            )
            return None

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.FloatTensor]:
        """
        define the validation step for pytorch lightning.

        Args:
            batch: input batch.
            batch_idx: batch index.
        """
        # enable non-training gradients with respect to specific quanitites if needed.
        torch.set_grad_enabled(self.grad_enabled)

        # forward pass
        loss = self._step(batch, "val")

        return {"val_loss": loss}

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.FloatTensor]:
        """
        define the test step for pytorch lightning.

        Args:
            batch: input batch.
            batch_idx: batch index.
        """
        # enable non-training gradients with respect to specific quanitites if needed.
        torch.set_grad_enabled(self.grad_enabled)

        # forward pass
        loss = self._step(batch, "test")

        return {"test_loss": loss}
