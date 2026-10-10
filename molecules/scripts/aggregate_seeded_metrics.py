"""Aggregate the five-seed molecular metrics into one tidy JSON file.

Maps every <NAME>_<MODE>_<SOLVER>_nfe<NFE>_seed<SEED>.vun.json written by
scripts/run_seeded.sh and scripts/compute_metrics_dir.sh back to its
(schedule, mode, NFE, seed), with the names used in the paper's figures and tables.

    python3 scripts/aggregate_seeded_metrics.py <sample_dir> [out.json] [--solver euler]

Feeds scripts/plot_seeded_stability.py and scripts/make_metrics_table.py.
"""

import argparse, json, os, re, glob, sys
from collections import defaultdict

import numpy as np

PATTERN = re.compile(
    r"^(?P<name>.+)_(?P<mode>ODE|SDE)_(?P<solver>[a-z]+)_nfe(?P<nfe>\d+)_seed(?P<seed>\d+)$"
)
LABEL = {"DDPM-cos-v1.0": "DDPM-cos $\\nu$=1.0", "DDPM-cos-v2.5": "DDPM-cos $\\nu$=2.5"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sample_dir")
    ap.add_argument("out", nargs="?", default="agg.json")
    ap.add_argument("--solver", default="euler")
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(f"{args.sample_dir}/*.vun.json")):
        stem = os.path.basename(f)[: -len(".vun.json")]
        m = PATTERN.match(stem)
        if m is None or m["solver"] != args.solver:
            continue
        lab, mode, nfe, seed = (
            LABEL.get(m["name"], m["name"]),
            m["mode"],
            int(m["nfe"]),
            int(m["seed"]),
        )
        d = json.load(open(f))
        rows.append(
            dict(
                schedule=lab,
                mode=mode,
                nfe=nfe,
                seed=seed,
                base=stem,
                stability=d["stability"]["rate"],
                validity=d["validity"]["rate"],
                uniqueness=d["uniqueness"]["rate"],
                novelty=d["novelty"]["rate"],
                n_generated=d["n_generated"],
                stab_count=d["stability"]["count"],
                stab_total=d["stability"]["total"],
                val_count=d["validity"]["count"],
                val_total=d["validity"]["total"],
                uni_count=d["uniqueness"]["count"],
                uni_total=d["uniqueness"]["total"],
                nov_count=d["novelty"]["count"],
                nov_total=d["novelty"]["total"],
            )
        )

    groups = defaultdict(list)
    for r in rows:
        groups[(r["schedule"], r["mode"], r["nfe"])].append(r)

    print(
        f"{len(rows)} result files -> {len(groups)} (schedule, mode, NFE) groups",
        file=sys.stderr,
    )

    out = []
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2])):
        g = groups[key]

        def ms(field):
            v = np.array([x[field] for x in g])
            return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0

        st, sts = ms("stability")
        va, vas = ms("validity")
        un, uns = ms("uniqueness")
        no, nos = ms("novelty")
        out.append(
            dict(
                schedule=key[0],
                mode=key[1],
                nfe=key[2],
                n_seeds=len(g),
                stability_mean=st,
                stability_std=sts,
                validity_mean=va,
                validity_std=vas,
                uniqueness_mean=un,
                uniqueness_std=uns,
                novelty_mean=no,
                novelty_std=nos,
                stab_count_sum=sum(x["stab_count"] for x in g),
                stab_total_sum=sum(x["stab_total"] for x in g),
                val_count_sum=sum(x["val_count"] for x in g),
                val_total_sum=sum(x["val_total"] for x in g),
                uni_count_sum=sum(x["uni_count"] for x in g),
                uni_total_sum=sum(x["uni_total"] for x in g),
                nov_count_sum=sum(x["nov_count"] for x in g),
                nov_total_sum=sum(x["nov_total"] for x in g),
                seeds=sorted(x["seed"] for x in g),
            )
        )

    out_path = args.out
    with open(out_path, "w") as f:
        json.dump({"groups": out, "rows": rows}, f, indent=1)
    print(f"wrote {out_path}", file=sys.stderr)

    # coverage report
    labs = sorted({r["schedule"] for r in rows})
    nfes = sorted({r["nfe"] for r in rows})
    print(f"\nNFEs present: {nfes}", file=sys.stderr)
    print("\ncoverage (n seeds per NFE), '.' = missing:", file=sys.stderr)
    print(
        f"{'schedule':<20}{'mode':<5}" + "".join(f"{n:>6}" for n in nfes),
        file=sys.stderr,
    )
    for lab in labs:
        for mode in ("ODE", "SDE"):
            cells, any_cell = [], False
            for n in nfes:
                g = groups.get((lab, mode, n))
                cells.append(f"{len(g):>6}" if g else "     .")
                any_cell = any_cell or bool(g)
            if any_cell:
                print(f"{lab:<20}{mode:<5}" + "".join(cells), file=sys.stderr)


if __name__ == "__main__":
    main()
