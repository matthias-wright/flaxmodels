import jax
import jax.numpy as jnp
import flax
from flax.core import frozen_dict
import numpy as np
import dill as pickle
import flaxmodels as fm
import data_pipeline
import checkpoint
from PIL import Image
from tqdm import tqdm
import argparse
import functools
import os


def style_mixing(args):
    num_devices = jax.device_count()
    ckpt = checkpoint.load_checkpoint(args.ckpt_path)
    config = ckpt['config']

    dtype = jnp.float32

    mapping_net = fm.stylegan2.MappingNetwork(z_dim=config.z_dim,
                                              c_dim=config.c_dim,
                                              w_dim=config.w_dim,
                                              num_ws=int(np.log2(config.resolution)) * 2 - 3,
                                              num_layers=8,
                                              dtype=dtype)

    synthesis_net = fm.stylegan2.SynthesisNetwork(resolution=config.resolution,
                                                  num_channels=config.img_channels,
                                                  w_dim=config.w_dim,
                                                  fmap_base=config.fmap_base,
                                                  dtype=dtype)

    params_ema_G = ckpt['params_ema_G']
    params_ema_G = params_ema_G.unfreeze()
    synthesis_params = {'params': params_ema_G['params']['synthesis_network'],
                        'noise_consts': params_ema_G['noise_consts']['synthesis_network']}
    synthesis_params = frozen_dict.freeze(synthesis_params)
    
    mapping_params = {'params': params_ema_G['params']['mapping_network'],
                      'moving_stats': params_ema_G['moving_stats']['mapping_network']}
    mapping_params = frozen_dict.freeze(mapping_params)

    synthesis_apply = jax.jit(functools.partial(synthesis_net.apply, noise_mode='const'))
    mapping_apply = jax.jit(functools.partial(mapping_net.apply, truncation_psi=args.truncation_psi, train=False))

    all_seeds = args.row_seeds + args.col_seeds
    # Generate noise inputs, [minibatch, component]
    all_z = jnp.concatenate([jax.random.normal(jax.random.PRNGKey(seed), shape=(1, 512)) for seed in all_seeds])
    # Generate latent vectors, [minibatch, num_ws, component]
    all_w = mapping_apply(mapping_params, all_z)
    # Generate images, [minibatch, H, W, 3]
    all_images = synthesis_apply(synthesis_params, all_w)
    # Normalize image to be in range [0, 1]
    all_images = (all_images - jnp.min(all_images)) / (jnp.max(all_images) - jnp.min(all_images))
    col_images = jnp.concatenate([all_images[i] for i in range(len(args.row_seeds))], axis=0)
    row_images = jnp.concatenate([all_images[len(args.row_seeds) + i] for i in range(len(args.col_seeds))], axis=1)

    images_grid = []
    
    cutoff = mapping_net.num_ws // 2

    # Generate style mixing images
    for row in range(len(args.row_seeds)):
        image_row = []
        for col in range(len(args.col_seeds)):
            # Combine first 9 dimensions from row seed latent w with last 9 dimensions from col seed latent w
            w = jnp.concatenate([all_w[row, :cutoff], all_w[len(args.row_seeds) + col, cutoff:]], axis=0)
            # Add batch dimension
            w = jnp.expand_dims(w, axis=0)
            image = synthesis_apply(synthesis_params, w)
            # Remove batch dimension
            image = jnp.squeeze(image, axis=0)

            # Normalize image to be in range [0, 1]
            image = (image - jnp.min(image)) / (jnp.max(image) - jnp.min(image))
            image_row.append(image)
        image_row = jnp.concatenate(image_row, axis=1)
        images_grid.append(image_row)

    images_grid = jnp.concatenate(images_grid, axis=0)

    # Add row and column images to the grid
    border = 20
    grid = np.ones((row_images.shape[0] + images_grid.shape[0] + border, 
                    col_images.shape[1] + images_grid.shape[1] + border,
                    3))
    grid[grid.shape[0] - images_grid.shape[0]:, grid.shape[1] - images_grid.shape[1]:] = images_grid
    grid[:row_images.shape[0], grid.shape[1] - row_images.shape[1]:] = row_images
    grid[grid.shape[0] - col_images.shape[0]:, :col_images.shape[1]] = col_images
    Image.fromarray(np.uint8(np.clip(grid * 255, 0, 255))).save(os.path.join(args.out_path, 'style_mixing.png'))
    print('Style mixing grid saved at:', args.out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, help='Path to the checkpoint.')
    parser.add_argument('--out_path', type=str, default='generated_images', help='Path where the generated images are stored.')
    parser.add_argument('--num_images', type=int, default=100, help='Number of images to generate.')
    parser.add_argument('--truncation_psi', type=float, default=0.5, help='Controls truncation (trading off variation for quality). If 1, truncation is disabled.')
    parser.add_argument('--row_seeds', type=int, nargs='*', help='List of random seeds for row images.')
    parser.add_argument('--col_seeds', type=int, nargs='*', help='List of random seeds for column images.')
    args = parser.parse_args()
    assert len(args.row_seeds) == len(args.col_seeds), 'row_seeds and col_seeds must have the same length.'
    os.makedirs(args.out_path, exist_ok=True)

    style_mixing(args)


