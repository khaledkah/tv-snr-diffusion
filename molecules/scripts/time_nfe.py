"""Wall-clock cost of one function evaluation (NFE) of the QM9 denoiser.

All schedules in the paper share one trained network and differ only
in the time grid and the O(n) update of the solver, so the sampling time is
(cost of one NFE) x NFE plus a small constant. This script measures both
sides of that statement on real test-split batches:

  per_nfe   one call of ``sampler.inference_step`` (model forward pass incl.
            input casting), repeated ``timing.n_repeats`` times per batch
  full      one complete ``sampler.denoise`` call per batch, i.e. T steps of
            the solver, so that full / T can be compared with per_nfe

It takes exactly the same Hydra overrides as scripts/sampling.py (sampler,
schedule, T, stochastic, data.batch_size, ...), so one schedule from
schedules.txt is timed with the same code path that generated the samples. Extra options (note the leading '+'):

  +timing.n_batches=3   test batches to time
  +timing.n_repeats=50  timed single-NFE calls per batch
  +timing.warmup=5      untimed calls first (CUDA context, cuDNN autotune)
  +timing.out=<json>    output file, named <schedule>_<mode>_<NFE>.json for
                        scripts/summarize_timing.py
  +timing.skip_full=true  only time the single forward pass (no sampling run)

Output: one JSON record with the GPU name, batch composition and the timings
in milliseconds. scripts/summarize_timing.py turns the records into the numbers
for the paper.
"""

import json
import logging
import os
import statistics
import time

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import tvsnr_mol  # noqa: F401
from schnetpack import properties

logger = logging.getLogger(__name__)
# sampling.yaml's run.dir uses this resolver
if not OmegaConf.has_resolver("uuid"):
    OmegaConf.register_new_resolver("uuid", lambda x: "timing")


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    """Run fn once and return (result, elapsed milliseconds)."""
    sync()
    t0 = time.perf_counter()
    out = fn()
    sync()
    return out, (time.perf_counter() - t0) * 1e3


@hydra.main(config_path="configs", config_name="sampling", version_base="1.2")
def main(cfg: DictConfig):
    n_batches = OmegaConf.select(cfg, "timing.n_batches", default=3)
    n_repeats = OmegaConf.select(cfg, "timing.n_repeats", default=50)
    warmup = OmegaConf.select(cfg, "timing.warmup", default=5)
    out_path = OmegaConf.select(cfg, "timing.out", default=f"timing_T{cfg.T}.json")
    skip_full = OmegaConf.select(cfg, "timing.skip_full", default=False)

    seed = cfg.seed if cfg.seed is not None else 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = instantiate(cfg.data)
    dataset.prepare_data()
    dataset.setup()
    loader = dataset.test_dataloader()

    sampler = instantiate(cfg.sampler)
    device = sampler.device
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    logger.info(f"device: {gpu}")

    records = []
    for k, batch in enumerate(loader):
        if k >= n_batches:
            break
        torch.manual_seed(seed * 100000 + k)  # same prior as scripts/sampling.py
        batch.update(sampler.sample_prior(batch, t=cfg.diff_t))
        batch = {key: val.to("cpu") for key, val in batch.items()}

        # --- single NFE: forward pass at a step in the middle of the grid ---
        prepared, start = sampler._prepare_inputs(batch)
        i_mid = start // 2
        for _ in range(warmup):
            sampler.inference_step(prepared, i_mid)
        per_nfe = [
            timed(lambda: sampler.inference_step(prepared, i_mid))[1]
            for _ in range(n_repeats)
        ]
        del prepared

        # --- full sampling run of T steps (warm-up run on the first batch) ---
        full_ms = None
        if not skip_full:
            kwargs = dict(
                start=cfg.start,
                max_steps=cfg.max_steps,
                progress_bar=False,
                order=cfg.order,
                atol=cfg.atol,
                rtol=cfg.rtol,
            )
            if k == 0:
                sampler.denoise(batch, **kwargs)
            _, full_ms = timed(lambda: sampler.denoise(batch, **kwargs))

        n_mols = int(len(batch[properties.n_atoms]))
        n_atoms = int(batch[properties.n_atoms].sum())
        rec = dict(
            batch=k,
            n_mols=n_mols,
            n_atoms=n_atoms,
            per_nfe_ms_median=statistics.median(per_nfe),
            per_nfe_ms_q25=float(np.percentile(per_nfe, 25)),
            per_nfe_ms_q75=float(np.percentile(per_nfe, 75)),
            full_ms=full_ms,
            full_ms_per_step=None if full_ms is None else full_ms / cfg.T,
        )
        logger.info(json.dumps(rec))
        records.append(rec)

    result = dict(
        gpu=gpu,
        torch=torch.__version__,
        T=int(cfg.T),
        stochastic=bool(cfg.stochastic),
        sampler=OmegaConf.select(cfg, "sampler._target_"),
        file_base_name=OmegaConf.select(cfg, "paths.file_base_name", default=None),
        batch_size=int(cfg.data.batch_size),
        n_repeats=n_repeats,
        batches=records,
    )
    out_path = hydra.utils.to_absolute_path(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    logger.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()
