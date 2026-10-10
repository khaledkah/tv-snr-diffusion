"""LaTeX tables for the five-seed molecular metrics.

One table per metric: schedules as rows, grouped into a reverse-ODE and a
reverse-SDE block, NFE as columns. Within each column, every value whose
1-std interval overlaps the best value's 1-std interval is set in bold, i.e.
every value that is not resolvably worse than the best one at five seeds.
This is deliberately not "only the nominal maximum": at high NFE the
schedules are expected to converge (the TV only affects sample quality
through discretization, section~sec:LTE), so making that indistinguishability
visible in bold is itself part of what the table should show, rather than an
artifact of hiding it behind a single bold cell.

    python3 scripts/make_metrics_table.py --agg agg.json --metric stability
    python3 scripts/make_metrics_table.py --agg agg.json --all
    python3 scripts/make_metrics_table.py --agg agg.json --binom

--binom writes the table of 95% Wilson binomial confidence intervals for the
stability rate, from the stable/total counts pooled over all seeds of a cell.
"""

import argparse, json, math, sys
from collections import defaultdict

# display order; non-VP schedules first, then their VP counterparts, ours last
ODE_ORDER = [
    "EDM",
    "EDM-UT",
    "SMLD",
    "OTFM",
    "VP-EDM-UT",
    "VP-SMLD",
    "VP-OTFM",
    "DDPM-linear",
    "DDPM-cos $\\nu$=2.5",
    "DDPM-cos $\\nu$=1.0",
    "VP-ISSNR",
]
SDE_ORDER = [
    "VP-EDM-UT",
    "VP-SMLD",
    "VP-OTFM",
    "DDPM-linear",
    "DDPM-cos $\\nu$=2.5",
    "DDPM-cos $\\nu$=1.0",
    "VP-ISSNR",
]
PRETTY = {
    "DDPM-cos $\\nu$=1.0": "DDPM-cos [$\\nu=1.0$]",
    "DDPM-cos $\\nu$=2.5": "DDPM-cos [$\\nu=2.5$]",
    "VP-ISSNR": "\\textbf{VP-ISSNR (ours)}",
}

METRICS = {
    "stability": ("stability_mean", "stability_std", "Stability rate"),
    "validity": ("validity_mean", "validity_std", "Validity"),
    "uniqueness": ("uniqueness_mean", "uniqueness_std", "Uniqueness"),
    "novelty": ("novelty_mean", "novelty_std", "Novelty"),
}
# Below this many stable structures per run the ratio metrics are computed from
# a handful of molecules and carry no information.
MIN_STABLE = 50
RATIO_METRICS = {"uniqueness", "novelty"}


def usable(g, metric):
    if metric not in RATIO_METRICS:
        return True
    return g["stab_count_sum"] / max(1, g["n_seeds"]) >= MIN_STABLE


def block(by, order, mode, metric, nfes, mean_k, std_k):
    rows, present = [], [s for s in order if (s, mode) in by]
    # best (mean, std) per column, over the rows that have a usable number
    best = {}
    for n in nfes:
        vals = [
            (by[(s, mode)][n][mean_k], by[(s, mode)][n][std_k])
            for s in present
            if n in by[(s, mode)] and usable(by[(s, mode)][n], metric)
        ]
        best[n] = max(vals, key=lambda t: t[0]) if vals else None
    for s in present:
        cells = []
        for n in nfes:
            g = by[(s, mode)].get(n)
            if g is None:
                cells.append("--")
            elif not usable(g, metric):
                cells.append("---")
            else:
                mean, std = g[mean_k], g[std_k]
                txt = f"{mean*100:.1f}$_{{\\pm{std*100:.1f}}}$"
                if best[n] is not None:
                    best_mean, best_std = best[n]
                    # bold every value statistically indistinguishable from the
                    # best one, i.e. whose 1-std interval overlaps the best
                    # value's 1-std interval (best_mean - mean <= std + best_std,
                    # since mean <= best_mean by construction). This is the
                    # point of the five-seed reruns: it shows where a schedule
                    # is not resolvably worse, rather than only where it is
                    # nominally largest.
                    if (best_mean - mean) <= (std + best_std) + 1e-12:
                        txt = f"\\textbf{{{mean*100:.1f}}}$_{{\\pm{std*100:.1f}}}$"
                cells.append(txt)
        rows.append(f"{PRETTY.get(s, s)} & " + " & ".join(cells) + " \\\\")
    return rows, [s for s in order if (s, mode) not in by]


def table(groups, metric):
    mean_k, std_k, title = METRICS[metric]
    by = defaultdict(dict)
    for g in groups:
        by[(g["schedule"], g["mode"])][g["nfe"]] = g
    nfes = sorted({n for s in by.values() for n in s})

    out = [
        f"% ---- {title} ----",
        "\\begin{table}",
        "\\centering",
        f"\\caption{{\\rebuttal{{{title} (\\%) on QM9 as a function of the number of function "
        f"evaluations, for the reverse ODE and the reverse SDE with the Euler solver. Each entry is the mean "
        f"over five independent sampling runs with different random seeds, with $\\pm$ one standard deviation "
        f"over those seeds as a subscript; at a given seed all schedules use the same initial noise samples and "
        f"the same molecular compositions. Within each column, every value whose $\\pm 1$ standard deviation "
        f"interval overlaps that of the column's largest value -- i.e.\\ every value not resolvably worse than "
        f"the best one at this sample size -- is set in bold.}}}}",
        f"\\label{{tab:Metrics{metric.capitalize()}}}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}l" + "c" * len(nfes) + "@{}}",
        "\\br",
        "Schedule & " + " & ".join(f"\\textbf{{{n}}}" for n in nfes) + " \\\\",
        "\\mr",
        f"\\multicolumn{{{len(nfes)+1}}}{{l}}{{\\emph{{Reverse ODE}}}} \\\\",
    ]
    ode_rows, ode_missing = block(by, ODE_ORDER, "ODE", metric, nfes, mean_k, std_k)
    out += ode_rows
    out += [
        "\\mr",
        f"\\multicolumn{{{len(nfes)+1}}}{{l}}{{\\emph{{Reverse SDE}}}} \\\\",
    ]
    sde_rows, sde_missing = block(by, SDE_ORDER, "SDE", metric, nfes, mean_k, std_k)
    out += sde_rows
    out += ["\\br", "\\end{tabular}", "\\end{table}", ""]
    missing = sorted(set(ode_missing) | set(sde_missing))
    return "\n".join(out), missing


def wilson(k, n, z=1.959963984540054):
    """95% Wilson score interval for k successes out of n trials."""
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def binom_table(groups):
    by = defaultdict(dict)
    for g in groups:
        by[(g["schedule"], g["mode"])][g["nfe"]] = g
    nfes = sorted({n for s in by.values() for n in s})
    totals = sorted({g["stab_total_sum"] for g in groups})
    per_run = sorted({g["stab_total_sum"] // g["n_seeds"] for g in groups})
    halves = []

    def rows(order, mode):
        out = []
        for s in [s for s in order if (s, mode) in by]:
            cells = []
            for n in nfes:
                g = by[(s, mode)].get(n)
                if g is None:
                    cells.append("--")
                    continue
                lo, hi = wilson(g["stab_count_sum"], g["stab_total_sum"])
                halves.append((hi - lo) / 2)
                cells.append(f"[{lo*100:.1f}, {hi*100:.1f}]")
            out.append(f"{PRETTY.get(s, s)} & " + " & ".join(cells) + " \\\\")
        return out

    body = rows(ODE_ORDER, "ODE")
    sde = rows(SDE_ORDER, "SDE")

    def num(x):
        return f"${x:,}$".replace(",", "\\,")

    if len(totals) == 1:
        n_txt = f"all {num(totals[0])} generated structures of the five runs ({num(per_run[0])} per run)"
    else:
        # some runs differ in size; say so rather than quoting one number
        n_txt = (
            f"the generated structures of the five runs, between {num(totals[0])} "
            f"and {num(totals[-1])} per entry"
        )
    out = [
        "% ---- Stability rate, binomial confidence intervals ----",
        "\\begin{table}",
        "\\centering",
        f"\\caption{{\\rebuttal{{Binomial $95\\,\\%$ confidence intervals (Wilson score) of the "
        f"stability rate (\\%) in table~\\ref{{tab:MetricsStability}}. Each interval is computed from "
        f"{n_txt}, i.e.\\ it quantifies the sampling uncertainty of the rate for the fixed set of test "
        f"compositions. The half-width is at most ${max(halves)*100:.1f}$ percentage points.}}}}",
        "\\label{tab:StabilityBinomialCI}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{@{}l" + "c" * len(nfes) + "@{}}",
        "\\br",
        "Schedule & " + " & ".join(f"\\textbf{{{n}}}" for n in nfes) + " \\\\",
        "\\mr",
        f"\\multicolumn{{{len(nfes)+1}}}{{l}}{{\\emph{{Reverse ODE}}}} \\\\",
    ]
    out += body
    out += [
        "\\mr",
        f"\\multicolumn{{{len(nfes)+1}}}{{l}}{{\\emph{{Reverse SDE}}}} \\\\",
    ]
    out += sde
    out += ["\\br", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", required=True)
    ap.add_argument("--metric", choices=list(METRICS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument(
        "--binom",
        action="store_true",
        help="Wilson confidence intervals of the stability rate",
    )
    args = ap.parse_args()
    groups = json.load(open(args.agg))["groups"]
    if args.binom:
        print(binom_table(groups))
        return
    metrics = list(METRICS) if args.all else [args.metric]
    if not metrics or metrics == [None]:
        sys.exit("pass --metric or --all")
    missing = set()
    for m in metrics:
        txt, miss = table(groups, m)
        print(txt)
        missing |= set(miss)
    if missing:
        print(f"% NOT YET AVAILABLE: {', '.join(sorted(missing))}", file=sys.stderr)


if __name__ == "__main__":
    main()
