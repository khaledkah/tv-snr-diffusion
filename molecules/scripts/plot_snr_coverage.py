"""Density of log-SNR values visited during training, per training schedule.

Training draws t ~ U[0,1], so the density of visited log-SNR is the push-forward
of the uniform density through log gamma(t):

    p(lambda) = |d log gamma / dt|^{-1}   evaluated at t = (log gamma)^{-1}(lambda)

which is computed here by numerical differentiation of log gamma(t) on a dense
grid. Note the code stores the SQUARED SNR, so the paper's log-SNR is half the
code's log gamma.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tv_snr.noise_schedules import PolynomialSchedule
from tv_snr.snr_schedules import KveToSNRSchedule, NoiseToSNRSchedule


def log_snr(sch, t):
    """Paper-convention log-SNR, log gamma = 0.5 * log(code gamma)."""
    return 0.5 * torch.log(sch(torch.as_tensor(t, dtype=torch.float64)))


def density(sch, n=20001):
    t = np.linspace(0.0, 1.0, n)
    lam = log_snr(sch, t).numpy()
    # p(lambda) d lambda = dt  =>  p = |d lambda / dt|^{-1}
    dl_dt = np.gradient(lam, t)
    return lam, 1.0 / np.abs(dl_dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="snr_coverage.pdf")
    args = ap.parse_args()

    schedules = {
        "Polynomial (training)": NoiseToSNRSchedule(
            noise_schedule=PolynomialSchedule(s=1e-5, discretize=False, T=None),
            t_min=1e-5,
            t_max=1.0,
        ),
        "EDM (training)": KveToSNRSchedule(
            sigma_min=0.002, sigma_max=30.0, rho=7.0, t_min=0.0, t_max=1.0
        ),
    }
    colors = {"Polynomial": "tab:green", "EDM": "tab:orange"}

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for label, sch in schedules.items():
        lam, p = density(sch)
        c = colors[label.split()[0]]
        ax.plot(lam, p / abs(np.trapezoid(p, lam)), lw=2.2, label=label, color=c)

    ax.set_xlabel(r"$\log \gamma$  (log-SNR)", fontsize=12)
    ax.set_ylabel("density of training samples", fontsize=12)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-0.01, 0.42)  # headroom so the legend does not cover the peak
    ax.grid(True, ls="--", lw=0.6, alpha=0.7)
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out, format="pdf")
    print(f"written {args.out}")

    # a compact quantitative summary for the text
    for label, sch in schedules.items():
        lam, p = density(sch)
        p = p / abs(np.trapezoid(p, lam))
        band = (lam > -2) & (lam < 2)
        frac = abs(np.trapezoid(p[band], lam[band]))
        print(
            f"{label:34s} fraction of training samples with |log gamma| < 2: {frac:.3f}"
        )


if __name__ == "__main__":
    main()
