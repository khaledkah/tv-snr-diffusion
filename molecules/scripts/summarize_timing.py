"""Summarize the JSON records of scripts/time_nfe.py into the numbers for the paper.

    python3 scripts/summarize_timing.py <timing_dir>/*.json

Prints, per (schedule, mode):
  - the cost of one NFE (median over batches and repeats), per batch and per molecule
  - a least-squares fit  full sampling time = a + b * NFE  over the timed NFEs,
    whose slope b should agree with the single-NFE cost if the solver overhead
    is negligible
and then one pooled per-NFE figure.
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np


def main(paths):
    recs = []
    for p in paths:
        r = json.load(open(p))
        # files are named <schedule>_<mode>_<NFE>.json
        r["label"] = os.path.basename(p)[: -len(".json")].rsplit("_", 2)[0]
        recs.append(r)
    if not recs:
        sys.exit("no timing files given")
    gpus = sorted({r["gpu"] for r in recs})
    if len(gpus) > 1:
        print(f"WARNING: records from several devices {gpus}; report them separately")

    by = defaultdict(list)
    for r in recs:
        by[(r["label"], "SDE" if r["stochastic"] else "ODE")].append(r)

    per_nfe_all, per_mol_all = [], []
    print(
        f"device: {', '.join(gpus)}   batch size: {sorted({r['batch_size'] for r in recs})}\n"
    )
    for (name, mode), rs in sorted(by.items()):
        rs = sorted(rs, key=lambda r: r["T"])
        per_nfe = [b["per_nfe_ms_median"] for r in rs for b in r["batches"]]
        per_mol = [
            b["per_nfe_ms_median"] / b["n_mols"] for r in rs for b in r["batches"]
        ]
        per_nfe_all += per_nfe
        per_mol_all += per_mol
        T = np.array([r["T"] for r in rs], float)
        full = np.array([np.mean([b["full_ms"] for b in r["batches"]]) for r in rs])
        print(f"{name} [{mode}]")
        print(
            f"  one NFE: {np.median(per_nfe):.1f} ms per batch "
            f"(range over batches/NFEs {min(per_nfe):.1f}-{max(per_nfe):.1f}), "
            f"{np.median(per_mol):.3f} ms per molecule"
        )
        if len(T) >= 2:
            b, a = np.polyfit(T, full, 1)
            print(
                f"  full run fit: {a:.0f} ms + {b:.1f} ms x NFE   "
                f"(slope / single-NFE cost = {b / np.median(per_nfe):.2f})"
            )
        for t, f in zip(T, full):
            print(f"    NFE {int(t):>4}: {f / 1e3:8.2f} s per batch")
        print()

    n_mols = int(np.median([b["n_mols"] for r in recs for b in r["batches"]]))
    ms = np.median(per_nfe_all)
    print("pooled over all schedules:")
    print(
        f"  one NFE = {ms:.1f} ms per batch of ~{n_mols} molecules = "
        f"{np.median(per_mol_all):.3f} ms per molecule"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
