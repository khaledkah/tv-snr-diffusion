Public source code for our paper: [TV/SNR - Enhancing Diffusion Models Efficiency by Disentangling Total-Variance and Signal-to-Noise Ratio](https://arxiv.org/abs/2502.08598) 

While the repository is currently being refactored, we provide the notebook to generate our toy examples under `analytic_score_toy_example.ipynb`. The folder `tv_snr` contains the source code for the TV/SNR framework, which is independent of the data modality. The refactored version, which includes the code for molecular structure generation, will be made public soon.

The computer vision code is based on the original [EDM repo](https://github.com/NVlabs/edm) and can be used in the same way by using generate_tv_snr.py to generate samples and reproduce the results reported in the paper. Example command to generate images using our best method:
```
python generate_tv_snr.py --outdir=out --snr_schedule sig3 --linear_time --num_steps 64 --tau 1.0 \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/baseline/baseline-cifar10-32x32-uncond-vp.pkl --grid

```

## How to cite
if you use TV/SNR in your research, please cite the corresponding publication:

Kahouli, K., Ripken, W., Gugler, S., Unke, O. T., Müller, K. R., & Nakajima, S. (2025). Enhancing Diffusion Models Efficiency by Disentangling Total-Variance and Signal-to-Noise Ratio. arXiv preprint arXiv:2502.08598.

    @article{kahouli2025disentangling,
      title={Disentangling Total-Variance and Signal-to-Noise-Ratio Improves Diffusion Models},
      author={Kahouli, Khaled and Ripken, Winfried and Gugler, Stefan and Unke, Oliver T and M{\"u}ller, Klaus-Robert and Nakajima, Shinichi},
      journal={arXiv preprint arXiv:2502.08598},
      year={2025}
    }
