Public source code for our paper: [TV/SNR - Disentangling Total-Variance and Signal-to-Noise-Ratio Improves Diffusion Models](https://arxiv.org/abs/2502.08598)

The repository contains

| Folder | Content |
|---|---|
| `tv_snr/` | The TV/SNR framework, independent of the data modality: `Scale_SNR_SDE`, the SNR schedules (ISSNR and all schedules of table 1), the TV (scale) schedules and the samplers (Euler(-Maruyama), Heun, RK45, DPM-Solver). |
| `vision/` | Image experiments with the pretrained networks of [EDM](https://github.com/NVlabs/edm) (based on the EDM code). |
| `molecules/` | Molecular structure generation on QM9 (training, sampling, metrics, tables and figures), based on [SchNetPack](https://github.com/atomistic-machine-learning/schnetpack). |
| `notebooks/` | Toy examples with an analytic score (figure 1 and the toy figures of the appendix). |

**Conventions.** The code works with squared quantities: `snr_sch(t)` returns the squared SNR $\gamma^2(t)$ and `scale_sch(t)` the squared total variance $\tau^2(t)$ of the paper. The inverse sigmoid SNR schedule (`InverseSigmoid`) is parametrized by `slope` $= 2\eta$ and `shift` $= 2\kappa$, i.e. VP-ISSNR with $\eta=1$, $\kappa=2$ is `slope=2, shift=4`. A constant TV is `ConstScale`, the exploding TV of VE/EDM is `VeScale`, and the TV of OTFM is `FMScale`.

## Installation

```
pip install -e ".[vision]"      # image experiments
pip install -e ".[molecules]"   # molecular experiments
```

## Images (`vision/`)

All commands are run from `vision/`. Generate images, e.g. with VP-ISSNR on CIFAR-10 (default: $\eta=1.5$, $\kappa=1$, Heun):

```
python generate_tv_snr.py --outdir=out --seeds=0-63 --steps=8 \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl
```

Schedules of the paper (see `settings.txt` for all of them):

| Schedule | Flags |
|---|---|
| EDM | `--snr_schedule linear --scale_schedule ve` (Karras time grid) |
| EDM-UT / VP-EDM-UT | `--snr_schedule kve --scale_schedule ve` / `constant` |
| SMLD / VP-SMLD | `--snr_schedule ve --scale_schedule ve` / `constant` |
| OTFM / VP-OTFM | `--snr_schedule issnr --slope 2 --shift 0 --scale_schedule fm` / `constant` |
| VP-ISSNR | `--snr_schedule issnr --slope <2 eta> --shift <2 kappa>` |
| VP-ISSNR [$\eta$ scaled] | `--snr_schedule issnr --eta_scaling` |

The solver is chosen with `--solver euler|heun|rk45|dpm`.

**Reproducing the FID results.** `settings.txt` lists every image experiment of the paper (figure 4, table 2 and the appendix). `scripts/run_fid.sh` generates 50k images and computes the FID for every row and seed (seed index `i` uses the images `50000*i, ..., 50000*i+49999`; table 2 uses the seed indices 0, 1, 3, 4, 5, since index 2 is used by the Bayesian optimization):

```
FILTER="^CIFAR|.*|heun" bash scripts/run_fid.sh     # FILTER selects rows of settings.txt
python scripts/collect_fid.py exp --csv fid.csv
python scripts/plot_fid.py fid.csv --outdir figures > table.tex                              # figure 4 (bottom), table 2
python scripts/plot_fid.py fid.csv --outdir figures --group vevp                            # figure 4 (top-right)
python scripts/plot_fid.py fid.csv --outdir figures --group all --solver euler              # appendix, solvers
python scripts/plot_fid.py fid.csv --outdir figures --group all --datasets AFHQ imagenet    # appendix, datasets
```

**Bayesian optimization** of $\eta$ and $\kappa$ (32 trials with W&B, FID of 50k images with seeds 100000-149999):

```
python hyperopt.py --dataset cifar --steps 4 --solver heun --count 32 --entity <wandb entity>
```

## Molecules (`molecules/`)

All commands are run from `molecules/`. `data/split.npz` is the QM9 split used in the paper (55k/10k/10k); QM9 is downloaded to `data/qm9.db` on first use.

**Training.** The denoiser used for all results (polynomial noise schedule, $\tau=1$), and the denoiser trained on the EDM schedule (appendix):

```
python scripts/train.py experiment=qm9_poly
python scripts/train.py experiment=qm9_edm
```

The model is written to `runs/<id>/best_model`.

**Sampling.** `schedules.txt` lists the 11 schedules of figure 3. `scripts/run_seeded.sh` samples 2,560 molecules for every schedule, ODE/SDE, NFE (4-256) and seed (0-4):

```
bash scripts/run_seeded.sh runs/<id>/best_model samples/euler
SOLVER=heun bash scripts/run_seeded.sh runs/<id>/best_model samples/heun           # appendix
SOLVER=dpm bash scripts/run_seeded.sh runs/<id>/best_model samples/dpm             # appendix
bash scripts/run_seeded.sh runs/<edm-id>/best_model samples/edm_model              # appendix, EDM-trained model
```

A single run: `python scripts/sampling.py T=8 seed=0 paths.denoiser_path=<model> paths.save_dir=<dir> paths.file_base_name=<name>` (default: VP-ISSNR, Euler, ODE).

**Metrics.** Stability, uniqueness and novelty are computed with [cG-SchNet](https://github.com/atomistic-machine-learning/cG-SchNet) (Open Babel), validity with RDKit:

```
git clone https://github.com/atomistic-machine-learning/cG-SchNet.git
bash scripts/compute_metrics_dir.sh samples/euler
```

**Tables and figures.**

```
python scripts/aggregate_seeded_metrics.py samples/euler agg.json
python scripts/make_metrics_table.py --agg agg.json --all        # tables of stability, validity, uniqueness, novelty
python scripts/make_metrics_table.py --agg agg.json --binom      # binomial confidence intervals
python scripts/plot_seeded_stability.py --agg agg.json --outdir figures                  # figure 3
python scripts/aggregate_seeded_metrics.py samples/heun agg_heun.json --solver heun
python scripts/plot_seeded_stability.py --agg agg_heun.json --outdir figures --solver heun
python scripts/aggregate_seeded_metrics.py samples/edm_model agg_edm.json
python scripts/plot_seeded_stability.py --agg agg.json --agg_edm agg_edm.json --outdir figures
python scripts/plot_snr_coverage.py --out figures/snr_coverage.pdf
python scripts/schedule_endpoints.py
python scripts/plot_rmsd.py --out figures/rmsd_vs_nfe.pdf
```

**Timing** of one NFE (`scripts/summarize_timing.py` summarizes the records):

```
python scripts/time_nfe.py T=8 data.batch_size=258 paths.denoiser_path=<model> paths.save_dir=timing \
    paths.file_base_name=VP-ISSNR +timing.out=timing/VP-ISSNR_ODE_8.json
python scripts/summarize_timing.py timing/*.json
```

## Toy examples (`notebooks/`)

`toy_1d.ipynb` produces the 1D panels of figure 1, the toy figures with all schedules and the EDM time grid of the appendix; `toy_gmm.ipynb` the 2D Gaussian mixture of the appendix.

## How to cite
if you use TV/SNR in your research, please cite the corresponding publication:

Kahouli, K., Ripken, W., Gugler, S., Unke, O. T., Müller, K. R., & Nakajima, S. (2025). Enhancing Diffusion Models Efficiency by Disentangling Total-Variance and Signal-to-Noise Ratio. arXiv preprint arXiv:2502.08598.

    @article{kahouli2025disentangling,
      title={Disentangling Total-Variance and Signal-to-Noise-Ratio Improves Diffusion Models},
      author={Kahouli, Khaled and Ripken, Winfried and Gugler, Stefan and Unke, Oliver T and M{\"u}ller, Klaus-Robert and Nakajima, Shinichi},
      journal={arXiv preprint arXiv:2502.08598},
      year={2025}
    }
