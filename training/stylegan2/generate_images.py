import jax
import jax.numpy as jnp
import flax
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


def generate_images(args):
    num_devices = jax.device_count()
    ckpt = checkpoint.load_checkpoint(args.ckpt_path)
    config = ckpt['config']
    
    dtype = jnp.float32

    generator_ema = fm.stylegan2.Generator(resolution=config.resolution,
                                           num_channels=config.img_channels,
                                           z_dim=config.z_dim,
                                           c_dim=config.c_dim,
                                           w_dim=config.w_dim,
                                           num_ws=int(np.log2(config.resolution)) * 2 - 3,
                                           num_mapping_layers=8,
                                           fmap_base=config.fmap_base,
                                           dtype=dtype)

    generator_apply = jax.jit(functools.partial(generator_ema.apply, truncation_psi=args.truncation_psi, train=False, noise_mode='const'))
    params_ema_G = ckpt['params_ema_G']

    for seed in tqdm(args.seeds):
        rng = jax.random.PRNGKey(seed)
        z_latent = jax.random.normal(rng, shape=(1, config.z_dim))
        labels = None
        if config.c_dim > 0:
            labels = jax.random.randint(rng, shape=(1,), minval=0, maxval=config.c_dim)
            labels = jax.nn.one_hot(labels, num_classes=config.c_dim)
            labels = jnp.reshape(labels, (1, config.c_dim))

        image = generator_apply(params_ema_G, jax.lax.stop_gradient(z_latent), labels)
        image = (image - jnp.min(image)) / (jnp.max(image) - jnp.min(image))

        Image.fromarray(np.uint8(np.clip(image[0] * 255, 0, 255))).save(os.path.join(args.out_path, f'{seed}.png'))
    print('Images saved at:', args.out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, help='Path to the checkpoint.')
    parser.add_argument('--out_path', type=str, default='generated_images', help='Path where the generated images are stored.')
    parser.add_argument('--truncation_psi', type=float, default=0.5, help='Controls truncation (trading off variation for quality). If 1, truncation is disabled.')
    parser.add_argument('--seeds', type=int, nargs='*', help='List of random seeds.')
    args = parser.parse_args()
    os.makedirs(args.out_path, exist_ok=True)

    generate_images(args)


