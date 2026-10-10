"""FID vs. NFE panels and the FID table (tab:ImageFID) from the csv written by
scripts/collect_fid.py, so the figure and the table always come from the same numbers.

    # figure 4 (bottom) and table 2
    python scripts/plot_fid.py fid.csv --outdir figures > table.tex
    # figure 4 (top-right): exploding/modulated vs. constant TV
    python scripts/plot_fid.py fid.csv --outdir figures --group vevp --datasets CIFAR FFHQ
    # appendix: other solvers and datasets
    python scripts/plot_fid.py fid.csv --outdir figures --group all --solver euler
    python scripts/plot_fid.py fid.csv --outdir figures --group all --datasets AFHQ imagenet
"""

import argparse
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NFES = [3, 7, 15, 31, 63]
TITLES = {"CIFAR": "CIFAR-10", "FFHQ": "FFHQ", "AFHQ": "AFHQ", "imagenet": "ImageNet"}
SOLVERS = {"heun": "Heun", "euler": "Euler", "rk45": "RK45", "dpm": "DPM Solver"}
# (key in the csv, legend label, colour, linestyle, bold), in the style of the
# published panels
STYLES = {
    "EDM": ("EDM", "darkorange", ":", False),
    "EDM-UT": ("EDM-UT", "darkorange", ":", False),
    "VP-EDM-UT": ("VP-EDM-UT (ours)", "darkorange", "-", False),
    "SMLD": ("SMLD", "green", ":", False),
    "VP-SMLD": ("VP-SMLD (ours)", "green", "-", False),
    "OTFM": ("OTFM", "brown", ":", False),
    "VP-OTFM": ("VP-OTFM (ours)", "brown", "-", False),
    "ISSNR-fixed": (
        "VP-ISSNR (Ours)\n[$\\eta = 1.5$, $\\kappa = 1.0$]",
        "red",
        "-",
        True,
    ),
    "ISSNR-eta": (
        "VP-ISSNR (Ours)\n[$\\eta$ scaled, $\\kappa = 0$]",
        "olive",
        "-",
        True,
    ),
    "ISSNR-BO": ("VP-ISSNR (Ours)\n[$\\eta$, $\\kappa$ tuned]", "blue", "-", True),
}
GROUPS = {
    "main": ["EDM", "OTFM", "ISSNR-eta", "ISSNR-BO"],
    "vevp": ["EDM-UT", "VP-EDM-UT", "SMLD", "VP-SMLD", "OTFM", "VP-OTFM"],
    "all": ["EDM", "SMLD", "OTFM", "ISSNR-fixed", "ISSNR-eta", "ISSNR-BO"],
}
# Table rows, in order. ISSNR-fixed is in the table only, not in the figure.
TABLE_ROWS = ["EDM", "OTFM", "ISSNR-fixed", "ISSNR-eta", "ISSNR-BO"]
TABLE_LABEL = {
    "EDM": "EDM",
    "OTFM": "OTFM",
    "ISSNR-fixed": "VP-ISSNR$[\\eta,\\kappa$ fixed$]$ (ours)",
    "ISSNR-eta": "VP-ISSNR$[\\eta$ scaled$]$ (ours)",
    "ISSNR-BO": "VP-ISSNR$[\\eta,\\kappa$ tuned$]$ (ours)",
}


def load(path, solver):
    by = {}
    for r in csv.DictReader(open(path)):
        if r["solver"] == solver:
            by[(r["dataset"], r["schedule"], int(r["nfe"]))] = (
                float(r["fid_mean"]),
                float(r["fid_std"]),
            )
    return by


def panel(by, ds, solver, schedules, group, outdir):
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    # Only the mean over seeds is drawn: the seed spread is well under one FID
    # unit, i.e. smaller than the markers on the logarithmic axis. The standard
    # deviations are listed in the accompanying table (tab:ImageFID).
    nfes_all = sorted({n for (d, _, n) in by if d == ds})
    for key in schedules:
        label, color, ls, _ = STYLES[key]
        nfes = [n for n in nfes_all if (ds, key, n) in by]
        if not nfes:
            continue
        mean = np.array([by[(ds, key, n)][0] for n in nfes])
        ax.plot(
            nfes,
            mean,
            marker="o",
            markersize=4.5,
            linestyle=ls,
            linewidth=2.2,
            color=color,
            label=label,
        )

    ax.set_title(f"{SOLVERS[solver]} - ODE ({TITLES.get(ds, ds)})", fontsize=21)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(nfes_all)
    ax.set_xticklabels(nfes_all, fontsize=16, weight="bold")
    yt = [2**i for i in range(1, 10)]
    ax.set_yticks(yt)
    ax.set_yticklabels(yt, fontsize=16, weight="bold")
    ax.set_ylim(1.6, 700)
    ax.grid(True, which="major", linestyle="--", linewidth=0.7)
    ax.set_ylabel("FID", fontsize=19, weight="bold")
    ax.set_xlabel("NFE", fontsize=19, weight="bold")

    leg = ax.legend(fontsize=14, loc="upper right", framealpha=0.9)
    bold = {STYLES[k][0] for k in schedules if STYLES[k][3]}
    for txt in leg.get_texts():
        if txt.get_text() in bold:
            txt.set_fontweight("bold")
    out = os.path.join(outdir, f"{ds}_{solver}_fid_vs_nfe_{group}.pdf")
    fig.savefig(out, format="pdf", bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def cell(mean, std, bold):
    # two decimals below 100, so that a spread of a few hundredths does not
    # round to a misleading "0.0"
    if mean >= 100:
        m, s = f"{mean:.1f}", f"{std:.1f}"
    else:
        m, s = f"{mean:.2f}", f"{std:.2f}"
    m = f"\\textbf{{{m}}}" if bold else m
    return f"{m}$_{{\\pm{s}}}$"


def table(by, datasets):
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{\\rebuttal{FID (\\textbf{lower is better}) on CIFAR-10 and FFHQ with Heun's method as a function of the "
        "number of function evaluations (NFE), for the schedules of "
        "the two lower panels of figure~\\ref{fig:ComparisonImage}. Each entry is the mean over independent sampling runs "
        "with different random seeds, each computed from $50\\,000$ generated images, with $\\pm 1$ standard deviation "
        "over the seeds as a subscript. At a given seed all schedules start from the same initial noise, and the tuned "
        "variant is evaluated on seeds disjoint from those used in its Bayesian optimization. VP-ISSNR$[\\eta,\\kappa$ fixed$]$ "
        "uses $\\eta=1.5$ and $\\kappa=1.0$ at every NFE and for both datasets; VP-ISSNR$[\\eta$ scaled$]$ "
        "applies one rule for $\\eta$ to both datasets; neither involves tuning. VP-ISSNR$[\\eta,\\kappa$ tuned$]$ uses the "
        "values of table~\\ref{tbl:hyperopt_values}. Within each column, every value whose $\\pm 1$ standard deviation "
        "interval overlaps that of the lowest FID is set in bold.}}",
        "\\label{tab:ImageFID}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}l" + "c" * len(NFES) + "@{}}",
        "\\br",
        "Schedule~/~NFE & " + " & ".join(f"\\textbf{{{n}}}" for n in NFES) + " \\\\",
    ]
    for ds in datasets:
        title = TITLES.get(ds, ds)
        lines += [
            "\\mr",
            f"\\multicolumn{{{len(NFES)+1}}}{{l}}{{\\emph{{{title}}}}} \\\\",
        ]
        rows = [k for k in TABLE_ROWS if all((ds, k, n) in by for n in NFES)]
        if not rows:
            continue
        best = {
            n: min((by[(ds, k, n)] for k in rows), key=lambda t: t[0]) for n in NFES
        }
        for key in rows:
            cells = []
            for n in NFES:
                m, s = by[(ds, key, n)]
                bm, bs = best[n]
                cells.append(cell(m, s, (m - bm) <= (s + bs) + 1e-12))
            lines.append(f"{TABLE_LABEL[key]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\br", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--solver", default="heun", choices=list(SOLVERS))
    ap.add_argument("--group", default="main", choices=list(GROUPS))
    ap.add_argument("--datasets", nargs="+", default=["CIFAR", "FFHQ"])
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    by = load(args.csv, args.solver)
    for ds in args.datasets:
        print(
            f"% wrote {panel(by, ds, args.solver, GROUPS[args.group], args.group, args.outdir)}"
        )
    if args.group == "main":
        print(table(by, args.datasets))


if __name__ == "__main__":
    main()
