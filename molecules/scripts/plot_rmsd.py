"""Median RMSD between generated structures and their DFT-relaxed counterparts
as a function of the NFE (appendix figure rmsd_vs_nfe).

The relaxations (ASE BFGS with PySCF, B3LYP/STO-3G pre-optimization followed by
B3LYP/6-31G(2df,p)) were run externally; data/rmsd_results.npz holds the results.
Each key is "<schedule>/<nfe>" and holds the RMSD of 600 structures, where NaN
means that the optimization did not converge or failed.

    python scripts/plot_rmsd.py [--results data/rmsd_results.npz] [--out rmsd_vs_nfe.pdf]
"""

import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAB10 = plt.get_cmap("tab10").colors
# (schedule key in the results file, legend label, colour)
SCHEDULES = [
    (
        "inversesigmoid_t_min_0.05_t_max_0.978_slope_3.0_shift_4.0_1.0",
        "VP-ISSNR (Ours)\n[$\\eta=1.5$ $\\kappa=2.0$]",
        TAB10[3],
    ),
    (
        "kve_t_min_0.0_t_max_1.0_rho_7.0_sigma_min_0.002_sigma_max_30.0_1.0",
        "VP-EDM-UT",
        TAB10[1],
    ),
    (
        "inversesigmoid_t_min_0.001_t_max_0.99_slope_2.0_shift_0.0_1.0",
        "VP-OTFM",
        TAB10[5],
    ),
    ("ve_t_min_0.0_t_max_1.0_sigma_min_0.002_sigma_max_30.0_1.0", "VP-SMLD", TAB10[2]),
    ("cosine_s_0.008_v_2.5_1.0", "DDPM-cos [$\\nu=2.5$]", TAB10[4]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="data/rmsd_results.npz")
    ap.add_argument("--out", default="rmsd_vs_nfe.pdf")
    args = ap.parse_args()
    results = np.load(args.results)

    plt.figure(figsize=(6.5, 5))
    for key, label, color in SCHEDULES:
        nfes = sorted(
            int(k.split("/")[1]) for k in results.files if k.split("/")[0] == key
        )
        medians = []
        for nfe in nfes:
            rmsd = results[f"{key}/{nfe}"]
            medians.append(np.median(rmsd[np.isfinite(rmsd)]))
        plt.plot(nfes, medians, label=label, marker="o", color=color)

    handles, labels = plt.gca().get_legend_handles_labels()
    # the proposed schedule last and in bold
    handles, labels = handles[1:] + handles[:1], labels[1:] + labels[:1]
    plt.legend(handles, labels, fontsize=14)
    plt.gca().get_legend().get_texts()[-1].set_fontweight("bold")
    plt.title("Euler - SDE", fontsize=18)
    plt.xscale("log", base=2)
    plt.grid(True, which="both", linestyle="--", linewidth=0.7)
    plt.xticks([2**i for i in range(3, 10)], [2**i for i in range(3, 10)], fontsize=14)
    plt.ylabel("RMSD [Ang]", fontsize=16)
    plt.xlabel("NFE", fontsize=16)
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
