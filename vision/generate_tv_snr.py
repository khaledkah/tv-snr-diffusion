# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Our adaptation of the EDM generate script with our options"."""

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
from tv_snr_sampler import tv_snr_sampler

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=0), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: 0.002]', metavar='FLOAT',            type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: 80]', metavar='FLOAT',              type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)

@click.option('--solver',                  help='ODE solver', metavar='euler|heun|rk45|dpm',                        type=click.Choice(['euler', 'heun', 'rk45', 'dpm']), default='heun', show_default=True)
@click.option('--grid',                    help='Save images in a single grid per batch', is_flag=True)

@click.option('--disc_type',               help='Gamma discretization', metavar='forward|midpoint|avg|difference',  type=click.Choice(['forward', 'midpoint', 'avg', 'difference']), default='forward', show_default=True)
@click.option('--snr_schedule',            help='SNR schedule (linear: EDM on the Karras time grid)', metavar='linear|kve|ve|issnr', type=click.Choice(['linear', 'kve', 've', 'issnr']), default='issnr', show_default=True)
@click.option('--scale_schedule',          help='TV schedule', metavar='ve|constant|fm',                            type=click.Choice(['ve', 'constant', 'fm']), default='constant', show_default=True)
@click.option('--slope',                   help='ISSNR slope (2*eta)', metavar='FLOAT',                             type=click.FloatRange(min=0, max=10), default=3, show_default=True)
@click.option('--shift',                   help='ISSNR shift (2*kappa)', metavar='FLOAT',                           type=click.FloatRange(min=-10, max=10), default=2, show_default=True)
@click.option('--eta_scaling',             help='Scale eta with the NFE and set kappa=0',                           is_flag=True)

def main(**kwargs):
    dist.init()
    _main(**kwargs)


def _main(network_pkl, outdir, subdirs, seeds, class_idx, max_batch_size, grid, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), **sampler_kwargs):
    """Generate random images with the TV/SNR sampler and the pretrained EDM networks.

    Examples:

    \b
    # Generate 64 images with VP-ISSNR and save them as out/*.png
    python generate_tv_snr.py --outdir=out --seeds=0-63 --batch=64 --steps=8 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl
    """
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    nfes = []

    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        # Generate images.
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        images, nfe = tv_snr_sampler(net, latents, class_labels, randn_like=rnd.randn_like, **sampler_kwargs)
        nfes.append(nfe)

        # Save images.
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        
        if grid:
            grid_w = int(np.ceil(np.sqrt(batch_size)))
            grid_h = int(np.ceil(batch_size / grid_w))
            assert grid_h * grid_w == batch_size

            images_np = images_np.reshape(grid_h, grid_w, *images_np.shape[1:]).transpose(0, 2, 1, 3, 4)
            images_np = images_np.reshape(grid_h * images_np.shape[1], grid_w * images_np.shape[3], images_np.shape[4])
            images_np = np.expand_dims(images_np, axis=0)

        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{seed:06d}.png')
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
            else:
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)

    # save nfe
    avg_nfe = np.array(nfes).mean()
    print(f"Used {avg_nfe} NFEs on average. Saving to file..")
    with open(os.path.join(outdir, 'nfe.txt'), 'w') as f:
        f.write(f"{avg_nfe:.2f}")

    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')
    return avg_nfe

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
