"""Endpoint (gamma_min / gamma_max) and train/test composition statistics.

Reports gamma_min and gamma_max of every schedule, the maximum pairwise distance
on the training split, and the train/test split at the molecule and at the
composition (molecular formula) level.

Note on conventions: internally the code stores the SQUARED SNR (the variable
called ``gamma`` in ``tv_snr``), whereas the paper's gamma is the
SNR itself. This script reports the paper convention, gamma = sqrt(code gamma).

Usage:
    python3 scripts/schedule_endpoints.py \
        --qm9-db data/qm9.db --split-file data/split.npz
"""

import argparse
import collections
import json

import numpy as np


def schedule_endpoints():
    """gamma_min / gamma_max of every schedule used in the paper."""
    import torch
    from tv_snr.noise_schedules import CosineSchedule, LinearSchedule
    from tv_snr.snr_schedules import (
        InverseSigmoid,
        KveToSNRSchedule,
        NoiseToSNRSchedule,
        VeToSNRSchedule,
    )

    def cos(v):
        return NoiseToSNRSchedule(
            noise_schedule=CosineSchedule(s=0.008, v=v, discretize=False, T=1000),
            t_min=(0.0001 if v == 1.0 else 0.01),
            t_max=0.999,
        )

    schedules = {
        "SMLD / VP-SMLD": VeToSNRSchedule(
            sigma_min=0.002, sigma_max=30.0, t_min=0.0, t_max=1.0
        ),
        "EDM-UT / VP-EDM-UT": KveToSNRSchedule(
            sigma_min=0.002, sigma_max=30.0, rho=7.0, t_min=0.0, t_max=1.0
        ),
        "OTFM / VP-OTFM": InverseSigmoid(
            slope=2.0, shift=0.0, t_min=0.0001, t_max=0.995
        ),
        "DDPM-cos (nu=1.0)": cos(1.0),
        "DDPM-cos (nu=2.5)": cos(2.5),
        "DDPM-linear": NoiseToSNRSchedule(
            noise_schedule=LinearSchedule(
                beta_start=0.1, beta_end=20.0, discretize=False, T=1000
            ),
            t_min=1e-05,
            t_max=1.0,
        ),
        "VP-ISSNR (ours)": InverseSigmoid(slope=2.0, shift=4.0, t_min=0.01, t_max=0.99),
    }

    rows = []
    for name, sch in schedules.items():
        # forward() maps t in [0,1] onto [t_min, t_max] internally
        g0 = float(sch(torch.tensor([0.0], dtype=torch.float64))[0]) ** 0.5
        g1 = float(sch(torch.tensor([1.0], dtype=torch.float64))[0]) ** 0.5
        rows.append(
            {
                "schedule": name,
                "t_min": sch.t_min,
                "t_max": sch.t_max,
                "gamma_max": g0,
                "gamma_min": g1,
                "sigma_max_equiv": 1.0 / g1,
            }
        )
    return rows


def composition_stats(qm9_db, split_file, data_var):
    """Pairwise-distance and molecular-formula statistics of the split."""
    import ase.db

    split = np.load(split_file)
    idx = {k: split[k] for k in ("train_idx", "val_idx", "test_idx") if k in split}

    formulas = {k: collections.Counter() for k in idx}
    max_pdist = 0.0
    per_mol_max = []

    with ase.db.connect(qm9_db) as conn:
        for name, ids in idx.items():
            for i in ids:
                atoms = conn.get(int(i) + 1).toatoms()  # ASE ids are 1-based
                formulas[name][atoms.get_chemical_formula()] += 1
                if name == "train_idx":
                    p = atoms.positions
                    d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
                    m = float(d.max())
                    per_mol_max.append(m)
                    max_pdist = max(max_pdist, m)

    per_mol_max = np.asarray(per_mol_max)
    train_f, test_f = set(formulas["train_idx"]), set(formulas["test_idx"])
    n_test_mols = int(sum(formulas["test_idx"].values()))
    n_test_in_train = sum(c for f, c in formulas["test_idx"].items() if f in train_f)

    return {
        "n_train": int(len(idx["train_idx"])),
        "n_val": int(len(idx.get("val_idx", []))),
        "n_test": n_test_mols,
        "max_pairwise_distance_train_A": max_pdist,
        "mean_per_molecule_max_distance_train_A": float(per_mol_max.mean()),
        "std_per_molecule_max_distance_train_A": float(per_mol_max.std()),
        "n_unique_formulas_train": len(train_f),
        "n_unique_formulas_test": len(test_f),
        "n_test_formulas_seen_in_train": len(test_f & train_f),
        "frac_test_formulas_seen_in_train": len(test_f & train_f) / max(1, len(test_f)),
        "frac_test_molecules_with_formula_seen_in_train": n_test_in_train
        / max(1, n_test_mols),
        # the model is conditioned on standardized coordinates
        "data_var": data_var,
        "max_pairwise_distance_standardized": max_pdist / np.sqrt(data_var),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qm9-db", default="data/qm9.db")
    ap.add_argument("--split-file", default="data/split.npz")
    ap.add_argument("--data-var", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--skip-data", action="store_true", help="only print the schedule endpoints"
    )
    args = ap.parse_args()

    result = {"schedules": schedule_endpoints()}
    print(
        f"{'schedule':22s} {'t_min':>8s} {'t_max':>7s} {'gamma_max':>12s} {'gamma_min':>11s} {'1/gamma_min':>12s}"
    )
    print("-" * 78)
    for r in result["schedules"]:
        print(
            f"{r['schedule']:22s} {r['t_min']:8.5g} {r['t_max']:7.5g} "
            f"{r['gamma_max']:12.4g} {r['gamma_min']:11.4g} {r['sigma_max_equiv']:12.4g}"
        )

    if not args.skip_data:
        print(
            "\ncomputing split statistics (this reads the whole QM9 database) ...",
            flush=True,
        )
        result["split"] = composition_stats(args.qm9_db, args.split_file, args.data_var)
        result["split"]["split_file"] = args.split_file
        print()
        for k, v in result["split"].items():
            print(f"  {k}: {v}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
