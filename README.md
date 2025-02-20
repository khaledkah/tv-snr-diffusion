Public source code for our paper: [TV/SNR - Enhancing Diffusion Models Efficiency by Disentangling Total-Variance and Signal-to-Noise Ratio](https://arxiv.org/abs/2502.08598) 

The repository is currently under refactoring! 

The computer vision code is based on the original [EDM repo](https://github.com/NVlabs/edm) and can be used in the same way using generate_tv_snr.py to generate samples.

We provide the notebook to generate our toy examples under analytic_score_toy_example.ipynb.
We add the code for our method in the tv_snr folder.

Example command to generate images using our best method:
```
python generate_tv_snr.py --outdir=out --snr_schedule sig3 --linear_time --num_steps 64 --tau 1.0 \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/baseline/baseline-cifar10-32x32-uncond-vp.pkl --grid

```

## How to cite
if you use MoreRed in your research, please cite the corresponding publication:

Kahouli, K., Ripken, W., Gugler, S., Unke, O. T., Müller, K. R., & Nakajima, S. (2025). Enhancing Diffusion Models Efficiency by Disentangling Total-Variance and Signal-to-Noise Ratio. arXiv preprint arXiv:2502.08598.

    @article{kahouli2025enhancing,
      title={Enhancing Diffusion Models Efficiency by Disentangling Total-Variance and Signal-to-Noise Ratio},
      author={Kahouli, Khaled and Ripken, Winfried and Gugler, Stefan and Unke, Oliver T and M{\~A}{\v{z}}ller, Klaus-Robert and Nakajima, Shinichi},
      journal={arXiv preprint arXiv:2502.08598},
      year={2025}
    }
