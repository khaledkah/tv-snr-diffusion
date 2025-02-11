Public source code for our paper: Enhancing Diffusion Models Efficiency by Disentangling Total-Variance and Signal-to-Noise Ratio

This code is based on the original [EDM repo](https://github.com/NVlabs/edm) and can be used in the same way using generate_tv_snr.py to generate samples.

We provide the notebook to generate our toy examples under analytic_score_toy_example.ipynb.
We add the code for our method in the tv_snr folder.

Example command to generate images using our best method:
```
python generate_tv_snr.py --outdir=out --snr_schedule sig3 --linear_time --num_steps 64 --tau 1.0 \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/baseline/baseline-cifar10-32x32-uncond-vp.pkl --grid

```