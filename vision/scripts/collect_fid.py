"""Aggregate the FID runs into a table.

Reads every <EXPDIR>/<DATASET>_<LABEL>_<SOLVER>_steps<STEPS>_seed<S>/{fid,nfe}.txt
written by scripts/run_fid.sh and reports mean and standard deviation over seeds.

    python scripts/collect_fid.py [exp] [--csv fid.csv]
"""

import argparse
import os
import re
import sys
from collections import defaultdict

import numpy as np

PATTERN = re.compile(
    r"^(?P<dataset>[^_]+)_(?P<label>.+)_(?P<solver>[^_]+)_steps(?P<steps>\d+)_seed(?P<seed>\d+)$"
)
ORDER = [
    "EDM",
    "EDM-UT",
    "VP-EDM-UT",
    "SMLD",
    "VP-SMLD",
    "OTFM",
    "VP-OTFM",
    "ISSNR-fixed",
    "ISSNR-eta",
    "ISSNR-BO",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("expdir", nargs="?", default="exp")
    ap.add_argument("--csv", default=None, help="also write the table as CSV")
    args = ap.parse_args()

    if not os.path.isdir(args.expdir):
        sys.exit(f"No such directory: {args.expdir}")

    runs = defaultdict(dict)  # (dataset, label, solver, nfe) -> {seed: fid}
    missing = []
    for name in sorted(os.listdir(args.expdir)):
        m = PATTERN.match(name)
        if not m:
            continue
        path = os.path.join(args.expdir, name, "fid.txt")
        if not os.path.exists(path):
            missing.append(name)
            continue
        nfe = round(float(open(os.path.join(args.expdir, name, "nfe.txt")).read()))
        key = (m["dataset"], m["label"], m["solver"], nfe)
        text = open(path).read().strip()
        try:
            runs[key][int(m["seed"])] = float(text)
        except ValueError:
            missing.append(f"{name} (unparseable: {text!r})")

    if not runs:
        sys.exit(f"No finished runs found under {args.expdir}.")

    rows = []
    print(
        f"{'dataset':<9}{'schedule':<12}{'solver':<7}{'NFE':>5}{'seeds':>7}{'FID mean':>11}{'std':>8}{'min':>8}{'max':>8}"
    )
    for dataset, label, solver, nfe in sorted(
        runs,
        key=lambda k: (k[0], k[2], ORDER.index(k[1]) if k[1] in ORDER else 99, k[3]),
    ):
        per_seed = runs[(dataset, label, solver, nfe)]
        v = np.array([per_seed[s] for s in sorted(per_seed)])
        std = v.std(ddof=1) if len(v) > 1 else 0.0
        print(
            f"{dataset:<9}{label:<12}{solver:<7}{nfe:>5}{len(v):>7}{v.mean():>11.3f}{std:>8.3f}{v.min():>8.3f}{v.max():>8.3f}"
        )
        rows.append(
            dict(
                dataset=dataset,
                schedule=label,
                solver=solver,
                nfe=nfe,
                n_seeds=len(v),
                fid_mean=v.mean(),
                fid_std=std,
                fid_min=v.min(),
                fid_max=v.max(),
                per_seed=";".join(f"{s}:{per_seed[s]:.4f}" for s in sorted(per_seed)),
            )
        )

    if missing:
        print(f"\nincomplete runs ({len(missing)}):")
        for m in missing[:20]:
            print("  " + m)

    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwritten to {args.csv}")


if __name__ == "__main__":
    main()
