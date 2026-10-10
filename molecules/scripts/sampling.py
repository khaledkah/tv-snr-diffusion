import logging
import os
import uuid

import ase.db
import hydra
import numpy as np
import torch
from ase import Atoms
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from tvsnr_mol.utils import check_validity, generate_bonds_data
from schnetpack import properties

_id = uuid.uuid1()
OmegaConf.register_new_resolver("uuid", lambda x: str(_id))

logger = logging.getLogger()
logFormatter = logging.Formatter(
    "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
if len(logger.handlers) > 0:
    logger.removeHandler(logger.handlers[0])
logger.addHandler(consoleHandler)


bonds_data = generate_bonds_data()

if torch.cuda.is_available():
    use_cpu = False
    device = torch.device("cuda")
else:
    use_cpu = True
    device = torch.device("cpu")


num_mols = 0.0
num_stable_mols = 0.0
num_stable_mols_wo_h = 0.0
num_connected_mols = 0.0
num_connected_mols_wo_h = 0.0


def compute_metrics(batch):
    global num_mols, num_stable_mols, num_connected_mols, num_stable_mols_wo_h, num_connected_mols_wo_h

    validity_res = check_validity(batch, *bonds_data.values())

    # infer metrics from validity results
    num_mols += len(validity_res["stable_molecules"])
    num_stable_mols += np.sum(validity_res["stable_molecules"])
    num_connected_mols += np.sum(validity_res["connected"])
    num_stable_mols_wo_h += np.sum(validity_res["stable_molecules_wo_h"])
    num_connected_mols_wo_h += np.sum(validity_res["connected_wo_h"])

    metrics = {
        "connectivity": np.array(validity_res["connected"], dtype=int),
        "stable_atoms": np.concatenate(validity_res["stable_atoms"]).astype(int),
        "stable_molecules": np.array(validity_res["stable_molecules"], dtype=int),
        "stable_atoms_wo_h": np.concatenate(validity_res["stable_atoms_wo_h"]).astype(
            int
        ),
        "stable_molecules_wo_h": np.array(
            validity_res["stable_molecules_wo_h"], dtype=int
        ),
        "connectivity_wo_h": np.array(validity_res["connected_wo_h"], dtype=int),
    }

    logger.info("Current metrics:")
    logger.info(f"frac stable molecules: {num_stable_mols / num_mols}")
    logger.info(f"frac connected molecules: {num_connected_mols / num_mols}")
    logger.info(f"frac stable molecules without H: {num_stable_mols_wo_h / num_mols}")
    logger.info(
        f"frac connected molecules without H: {num_connected_mols_wo_h / num_mols}"
    )
    logger.info("\n")

    return metrics


def save_to_db(path, samples, trajs, num_steps, metrics=None):
    target_db = ase.db.connect(path)

    for i in samples[properties.idx_m].unique(sorted=True):
        mask = samples[properties.idx_m] == i

        mol = Atoms(
            numbers=samples[properties.Z][mask].cpu(),
            positions=samples[properties.R][mask].cpu(),
        )

        data = {
            "trajs": (
                trajs[properties.R][mask].numpy() if properties.R in trajs else None
            ),
            "steps": trajs["step"][i].numpy() if "step" in trajs else None,
            "num_steps": (np.array([num_steps[i].item()])),
        }

        if metrics:
            for k, v in metrics.items():
                if len(v) == len(samples[properties.n_atoms]):
                    data[k] = v[[i]]
                else:
                    data[k] = v[mask.numpy()]

        target_db.write(
            mol,
            data=data,
            num_atoms=len(mol.numbers),
            orig_id=samples[properties.idx][i].item() + 1,
            num_steps=data["num_steps"][0],
            stable=(
                int(data["stable_molecules"][0]) if "stable_molecules" in data else -1
            ),
        )


def set_up_dirs(cfg):
    os.makedirs(cfg.paths.save_dir, exist_ok=True)

    file_base_path = os.path.join(cfg.paths.save_dir, cfg.paths.file_base_name)

    # Check for existing targets
    if not cfg.save_as_tensors:
        file_base_path += ".db"

    if os.path.exists(file_base_path):
        raise ValueError(f"Target {file_base_path} already exists.")

    return file_base_path


@hydra.main(config_path="configs", config_name="sampling", version_base="1.2")
def main(cfg: DictConfig):
    if cfg.seed is not None:
        logger.info(f"Setting seed: {cfg.seed}")
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    # paths
    config_dir_path = os.path.join(cfg.run.dir)
    run_id = cfg.run.id
    config_file_path = os.path.join(config_dir_path, "config.yaml")

    # save resolved config file
    with open(config_file_path, "w") as f:
        OmegaConf.save(cfg, f, resolve=True)

    logger.info(f"Run ID: {run_id}")
    logger.info(f"Config file: {config_file_path}")

    # set up directories and get target file path
    file_base_path = set_up_dirs(cfg)

    # Instantiate the dataset from the config
    dataset = instantiate(cfg.data)
    dataset.prepare_data()
    dataset.setup()

    if cfg.split == "test":
        dataloader = dataset.test_dataloader()
    elif cfg.split == "val":
        dataloader = dataset.val_dataloader()
    elif cfg.split == "train":
        dataloader = dataset.train_dataloader()
    else:
        raise ValueError(f"Split {cfg.split} not recognized.")

    # Instantiate the sampler from the config
    sampler = instantiate(cfg.sampler)

    if file_base_path.endswith(".db"):
        target = ase.db.connect(file_base_path)
        target.metadata = {  # type: ignore
            "run_id": run_id,
            "config_path": config_file_path,
        }
    else:
        os.makedirs(file_base_path, exist_ok=False)

    # number of samples
    n = 0

    for k, batch in tqdm(enumerate(dataloader)):
        # Re-seed per batch so that the standard normal drawn for the prior is
        # identical for a given (seed, batch index) across schedules, NFEs and
        # solvers. Without this, stochastic samplers consume the global RNG
        # inside the reverse loop and the priors of subsequent batches diverge.
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed * 100000 + k)

        priors = sampler.sample_prior(batch, t=cfg.diff_t)
        batch.update(priors)

        # Save GPU mem: batch will be cloned and moved to gpu inside the sampler
        batch = {k: v.to("cpu") for k, v in batch.items()}

        samples, num_steps, trajs = sampler.denoise(
            batch,
            start=cfg.start,
            max_steps=cfg.max_steps,
            progress_bar=True,
            order=cfg.order,
            atol=cfg.atol,
            rtol=cfg.rtol,
        )

        # Save results
        samples.update(
            {
                prop: val.cpu()
                for prop, val in batch.items()
                if prop not in samples
                and prop
                in [
                    properties.R,
                    properties.Z,
                    properties.idx_m,
                    properties.idx,
                    properties.n_atoms,
                ]
            }
        )

        metrics = compute_metrics(samples) if cfg.compute_metrics else None

        if cfg.save_as_tensors:
            # save as torch tensors
            results = {"samples": samples, "trajs": trajs, "num_steps": num_steps}

            if metrics:
                results["metrics"] = {
                    k: torch.from_numpy(v).cpu() for k, v in metrics.items()
                }

            torch.save(results, os.path.join(file_base_path, f"{k}.pt"))
        else:
            # save in ASE Database
            save_to_db(file_base_path, samples, trajs, num_steps, metrics)

        n += len(samples[properties.n_atoms])

        if n >= cfg.num_samples:
            break


if __name__ == "__main__":
    main()
