"""Figure 3 of the manuscript, redrawn from the five-seed sampling runs.

Reproduces the three stability-vs-NFE panels, with the mean over seeds as the
line. The seed-to-seed standard deviation (median 0.6 percentage points) is not
drawn, since it is smaller than the markers; it is listed in table D1 instead.

    python3 scripts/plot_seeded_stability.py --agg agg.json --outdir DIR [--solver euler]

With --agg_edm (runs of the denoiser trained on the EDM schedule), the panel
comparing the two denoisers is drawn as well.

Reads the aggregated metrics written by the companion aggregation step; any
schedule that has no runs yet is skipped with a warning, so the script can be
re-run as further results arrive.
"""

import argparse, json, os, sys
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# tab10 indices as used by the original analysis notebook, so that the colours
# of the redrawn panels match the published ones.
TAB10 = plt.get_cmap("tab10").colors
COLOR = {
    "VP-ISSNR": TAB10[3],
    "EDM-UT": TAB10[1],
    "VP-EDM-UT": TAB10[1],
    "EDM": TAB10[1],
    "SMLD": TAB10[2],
    "VP-SMLD": TAB10[2],
    "OTFM": TAB10[5],
    "VP-OTFM": TAB10[5],
    "DDPM-cos $\\nu$=1.0": TAB10[6],
    "DDPM-cos $\\nu$=2.5": TAB10[4],
    "DDPM-linear": TAB10[7],
}

PANELS = {
    # name: (mode, [(schedule, legend label, linestyle, bold), ...])
    "stab_ve_vp_Euler_ODE_forward_model_poly_seeded": (
        "ODE",
        [
            ("EDM-UT", "EDM-UT", "--", False),
            ("VP-EDM-UT", "VP-EDM-UT (ours)", "-", False),
            ("SMLD", "SMLD", "--", False),
            ("VP-SMLD", "VP-SMLD (ours)", "-", False),
            ("OTFM", "OTFM", "--", False),
            ("VP-OTFM", "VP-OTFM (ours)", "-", False),
        ],
    ),
    "stab_Euler_ODE_forward_model_poly_seeded": (
        "ODE",
        [
            ("EDM", "EDM", "-.", False),
            ("VP-EDM-UT", "VP-EDM-UT (ours)", "-", False),
            ("VP-SMLD", "VP-SMLD (ours)", "-", False),
            ("VP-OTFM", "VP-OTFM (ours)", "-", False),
            ("DDPM-linear", "DDPM-linear", "-", False),
            ("DDPM-cos $\\nu$=2.5", "DDPM-cos [$\\nu$ = 2.5]", "-", False),
            ("DDPM-cos $\\nu$=1.0", "DDPM-cos [$\\nu$ = 1.0]", "-", False),
            (
                "VP-ISSNR",
                "VP-ISSNR (Ours)\n[$\\eta$ = 1.0, $\\kappa$ = 2.0]",
                "-",
                True,
            ),
        ],
    ),
    "stab_Euler_SDE_forward_model_poly_seeded": (
        "SDE",
        [
            ("VP-EDM-UT", "VP-EDM-UT (ours)", "-", False),
            ("VP-SMLD", "VP-SMLD (ours)", "-", False),
            ("VP-OTFM", "VP-OTFM (ours)", "-", False),
            ("DDPM-linear", "DDPM-linear", "-", False),
            ("DDPM-cos $\\nu$=2.5", "DDPM-cos [$\\nu$ = 2.5]", "-", False),
            ("DDPM-cos $\\nu$=1.0", "DDPM-cos [$\\nu$ = 1.0]", "-", False),
            (
                "VP-ISSNR",
                "VP-ISSNR (Ours)\n[$\\eta$ = 1.0, $\\kappa$ = 2.0]",
                "-",
                True,
            ),
        ],
    ),
}


def load(agg_path):
    data = json.load(open(agg_path))["groups"]
    by = defaultdict(dict)
    for g in data:
        by[(g["schedule"], g["mode"])][g["nfe"]] = g
    return by


def draw(panel_name, mode, entries, by, outdir, reference=None, solver="Euler"):
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    missing = []
    for sched, label, ls, bold in entries:
        series = by.get((sched, mode))
        if not series:
            missing.append(sched)
            continue
        nfes = sorted(series)
        mean = np.array([series[n]["stability_mean"] for n in nfes])
        ax.plot(
            nfes,
            mean,
            marker="o",
            markersize=4.5,
            linestyle=ls,
            linewidth=2.2,
            color=COLOR.get(sched, "black"),
            label=label,
            zorder=3 if bold else 2,
        )

    # the SDE panel repeats the best ODE schedule as a black reference curve
    if reference is not None:
        series = by.get(reference)
        if series:
            nfes = sorted(series)
            mean = np.array([series[n]["stability_mean"] for n in nfes])
            ax.plot(
                nfes,
                mean,
                marker="o",
                markersize=4,
                linestyle=":",
                linewidth=2.0,
                color="black",
                label="ODE [VP-ISSNR]",
                zorder=2,
            )

    ax.set_title(f"{solver} - {mode} (QM9)", fontsize=21)
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", linestyle="--", linewidth=0.7)
    ticks = [2**i for i in range(2, 10 if solver == "Heun" else 9)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, fontsize=16, weight="bold")
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticklabels(
        [f"{v:.1f}" for v in np.arange(0, 1.1, 0.1)], fontsize=16, weight="bold"
    )
    ax.set_ylim(-0.02, 1.0)
    ax.set_ylabel("Stability", fontsize=19, weight="bold")
    ax.set_xlabel("NFE", fontsize=19, weight="bold")

    # single-column legend in the lower right, as in the original submission;
    # the SDE panel's ODE reference curve is listed first, as it was there
    handles, labels = ax.get_legend_handles_labels()
    if reference is not None:
        handles, labels = handles[-1:] + handles[:-1], labels[-1:] + labels[:-1]
    leg = ax.legend(
        handles,
        labels,
        fontsize=15,
        ncol=1,
        loc="lower right",
        framealpha=0.9,
        borderpad=0.4,
        labelspacing=0.25,
        handlelength=2.0,
    )
    # bold the proposed schedule; matched by label text, since the reference
    # curve adds a legend entry that is not in `entries`
    bold_labels = {lab for _, lab, _, bold in entries if bold}
    for txt in leg.get_texts():
        if txt.get_text() in bold_labels:
            txt.set_fontweight("bold")
    fig.tight_layout()

    out = os.path.join(outdir, panel_name + ".pdf")
    fig.savefig(out, format="pdf")
    fig.savefig(out.replace(".pdf", ".png"), format="png", dpi=150)
    plt.close(fig)
    return out, missing


MODEL_EFFECT = [
    ("EDM-UT", "EDM-UT"),
    ("SMLD", "SMLD"),
    ("DDPM-cos $\\nu$=1.0", "DDPM-cos [$\\nu$ = 1.0]"),
    ("VP-ISSNR", "VP-ISSNR (Ours)\n[$\\eta$ = 1.0, $\\kappa$ = 2.0]"),
]


def draw_model_effect(by, by_edm, outdir):
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for data, ls in [(by, "-"), (by_edm, "--")]:
        for sched, _ in MODEL_EFFECT:
            series = data.get((sched, "ODE"))
            if series:
                nfes = sorted(series)
                ax.plot(
                    nfes,
                    [series[n]["stability_mean"] for n in nfes],
                    marker="o",
                    markersize=4.5,
                    linestyle=ls,
                    linewidth=2.2,
                    color=COLOR[sched],
                )
    handles = [
        plt.Line2D([], [], color="black", linestyle="-", label="Polynomial [train]"),
        plt.Line2D([], [], color="black", linestyle="--", label="EDM [train]"),
    ]
    handles += [
        plt.Line2D([], [], color=COLOR[s], marker="o", label=lab)
        for s, lab in MODEL_EFFECT
    ]
    ax.legend(handles=handles, fontsize=13, loc="lower right", framealpha=0.9)
    ax.set_title("Euler - ODE", fontsize=21)
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", linestyle="--", linewidth=0.7)
    ticks = [2**i for i in range(2, 9)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, fontsize=16, weight="bold")
    ax.set_ylim(-0.02, 1.0)
    ax.set_ylabel("Stability", fontsize=19, weight="bold")
    ax.set_xlabel("NFE", fontsize=19, weight="bold")
    fig.tight_layout()
    out = os.path.join(outdir, "model_effect_stab_Euler_ODE_seeded.pdf")
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--solver", default="euler", choices=["euler", "heun", "dpm"])
    ap.add_argument("--agg_edm", default=None)
    args = ap.parse_args()
    solver = {"euler": "Euler", "heun": "Heun", "dpm": "DPMSolver"}[args.solver]
    os.makedirs(args.outdir, exist_ok=True)
    by = load(args.agg)

    all_missing = set()
    for name, (mode, entries) in PANELS.items():
        if not any((sched, mode) in by for sched, *_ in entries):
            continue
        ref = ("VP-ISSNR", "ODE") if mode == "SDE" else None
        name = name.replace("Euler", solver)
        out, missing = draw(
            name, mode, entries, by, args.outdir, reference=ref, solver=solver
        )
        all_missing |= set(missing)
        print(
            f"wrote {out}" + (f"   (missing: {', '.join(missing)})" if missing else "")
        )
    if args.agg_edm:
        print(f"wrote {draw_model_effect(by, load(args.agg_edm), args.outdir)}")
    if all_missing:
        print(
            f"\nWARNING: no runs for {sorted(all_missing)}; those curves are absent.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
