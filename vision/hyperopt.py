"""Bayesian optimization of the VP-ISSNR parameters (slope = 2*eta, shift = 2*kappa)
with W&B sweeps, minimizing the FID of 50k images generated with seeds 100000-149999.

    python hyperopt.py --dataset cifar --steps 4 --solver heun --count 32
"""

import os
import random
import shutil
import string
from functools import partial
from pathlib import Path

import click
import numpy as np
import torch
import wandb

import dnnlib
from fid import calculate_inception_stats, calculate_fid_from_inception_stats
from generate_tv_snr import _main
from torch_utils import distributed as dist

MODELS = {
    "cifar": "edm-cifar10-32x32-uncond-vp.pkl",
    "ffhq": "edm-ffhq-64x64-uncond-vp.pkl",
    "afhq": "edm-afhqv2-64x64-uncond-vp.pkl",
    "imagenet": "edm-imagenet-64x64-cond-adm.pkl",
}

FID_REFS = {
    "cifar": "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz",
    "ffhq": "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/ffhq-64x64.npz",
    "afhq": "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/afhqv2-64x64.npz",
    "imagenet": "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz",
}


def calc_fid(image_path, ref_path, num_expected, seed, batch):
    """Calculate FID for a given set of images."""
    dist.print0(f'Loading dataset reference statistics from "{ref_path}"...')
    ref = None
    if dist.get_rank() == 0:
        with dnnlib.util.open_url(ref_path) as f:
            ref = dict(np.load(f))

    mu, sigma = calculate_inception_stats(
        image_path=image_path,
        num_expected=num_expected,
        seed=seed,
        max_batch_size=batch,
    )
    dist.print0("Calculating FID...")
    if dist.get_rank() == 0:
        fid = calculate_fid_from_inception_stats(mu, sigma, ref["mu"], ref["sigma"])
        print(f"{fid:g}")
    torch.distributed.barrier()
    return fid


def sig_hyperopt(shift, slope, steps, dataset, solver, models_dir, n_images, tmp_dir):
    fname = "".join(random.choices(string.ascii_letters + string.digits, k=15))
    outdir = Path(tmp_dir) / fname
    outdir.mkdir(parents=True, exist_ok=True)

    network = MODELS[dataset]
    if models_dir is not None:
        network = os.path.join(models_dir, network)
    else:
        network = f"https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/{network}"

    kwargs = {
        "outdir": str(outdir),
        "network_pkl": network,
        "solver": solver,
        "disc_type": "forward",
        "snr_schedule": "issnr",
        "scale_schedule": "constant",
        "num_steps": steps,
        "subdirs": True,
        # seeds 100000-149999 are reserved for tuning
        "seeds": list(range(100_000, 100_000 + n_images)),
        "class_idx": None,
        "max_batch_size": 64,
        "rho": 7.0,
        "grid": False,
        "shift": shift,
        "slope": slope,
    }

    nfe = _main(**kwargs)

    fid_score = calc_fid(str(outdir), FID_REFS[dataset], n_images, seed=0, batch=64)
    shutil.rmtree(outdir)

    return fid_score, nfe


def sweep_f(first_run, **kwargs):
    if first_run[0]:
        # the first trial is pinned to VP-OTFM (eta=1, kappa=0)
        run = wandb.init(allow_val_change=True)
        wandb.config.__dict__["_locked"] = {}
        wandb.config.update({"shift": 0, "slope": 2}, allow_val_change=True)
        first_run[0] = False
    else:
        run = wandb.init()

    c = wandb.config
    fid, nfe = sig_hyperopt(
        c.shift, c.slope, int(c.steps), c.dataset, c.solver, **kwargs
    )
    wandb.log({"fid": fid, "nfe": nfe})
    run.finish()


@click.command()
@click.option("--dataset", type=click.Choice(list(MODELS)), required=True)
@click.option(
    "--steps",
    type=int,
    required=True,
    help="Number of sampling steps (Heun: NFE = 2*steps-1)",
)
@click.option(
    "--solver", type=click.Choice(["euler", "heun"]), default="heun", show_default=True
)
@click.option(
    "--count", type=int, default=32, show_default=True, help="Number of trials"
)
@click.option(
    "--sweep_id",
    type=str,
    default=None,
    help="Join an existing sweep instead of creating one",
)
@click.option("--project", type=str, default="tv-snr", show_default=True)
@click.option("--entity", type=str, default=None)
@click.option(
    "--models_dir",
    type=str,
    default=None,
    help="Local directory with the EDM pickles [default: download]",
)
@click.option("--n_images", type=int, default=50000, show_default=True)
@click.option("--tmp_dir", type=str, default="/tmp", show_default=True)
def hyperopt_main(
    dataset,
    steps,
    solver,
    count,
    sweep_id,
    project,
    entity,
    models_dir,
    n_images,
    tmp_dir,
):
    first_run = [sweep_id is None]
    if sweep_id is None:
        nfe = 2 * steps - 1 if solver == "heun" else steps
        sweep_configuration = {
            "name": f"{dataset}_hyperopt NFE={nfe} {solver}",
            "method": "bayes",
            "metric": {"goal": "minimize", "name": "fid"},
            "parameters": {
                # eta in [0.5, 4], kappa in [-3, 3]
                "shift": {"min": -6.0, "max": 6.0, "distribution": "uniform"},
                "slope": {"min": 1.0, "max": 8.0, "distribution": "uniform"},
                "dataset": {"value": dataset},
                "steps": {"value": steps},
                "solver": {"value": solver},
            },
        }
        sweep_id = wandb.sweep(
            sweep=sweep_configuration, project=project, entity=entity
        )

    print("Running with sweep id:", sweep_id)
    dist.init()

    swp = partial(
        sweep_f, first_run, models_dir=models_dir, n_images=n_images, tmp_dir=tmp_dir
    )
    wandb.agent(sweep_id, function=swp, count=count, project=project, entity=entity)


if __name__ == "__main__":
    hyperopt_main()
