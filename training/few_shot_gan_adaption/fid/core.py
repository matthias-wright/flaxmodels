import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import numpy as np
import os
import functools
import argparse
import scipy
from tqdm import tqdm

from . import inception
from . import utils


class FID:

    def __init__(self, generator, dataset, config, use_cache=True, truncation_psi=1.0):
        """
        Evaluates the FID score for a given generator and a given dataset.
        Implementation mostly taken from https://github.com/matthias-wright/jax-fid

        Reference: https://arxiv.org/abs/1706.08500

        Args:
            generator (nn.Module): Generator network.
            dataset (tf.data.Dataset): Dataset containing the real images.
            config (argparse.Namespace): Configuration.
            use_cache (bool): If True, only compute the activation stats once for the real images and store them.
            truncation_psi (float): Controls truncation (trading off variation for quality). If 1, truncation is disabled.
        """
        self.num_images = config.num_fid_images
        self.batch_size = config.batch_size
        self.c_dim = config.c_dim
        self.z_dim = config.z_dim
        self.dataset = dataset
        self.num_devices = jax.device_count()
        self.use_cache = use_cache

        if self.use_cache:
            self.cache = {}

        rng = jax.random.PRNGKey(0)
        inception_net = inception.InceptionV3(pretrained=True)
        self.inception_params = inception_net.init(rng, jnp.ones((1, config.resolution, config.resolution, 3)))
        self.inception_params = flax.jax_utils.replicate(self.inception_params)
        #self.inception = jax.jit(functools.partial(model.apply, train=False))
        self.inception_apply = jax.pmap(functools.partial(inception_net.apply, train=False), axis_name='batch')
        
        self.generator_apply = jax.pmap(functools.partial(generator.apply, truncation_psi=truncation_psi, train=False, noise_mode='const'), axis_name='batch')

    def compute_fid(self, generator_params, seed_offset=0):
        generator_params = flax.jax_utils.replicate(generator_params)
        mu_real, sigma_real = self.compute_stats_for_dataset()
        mu_fake, sigma_fake = self.compute_stats_for_generator(generator_params, seed_offset)
        fid_score = self.compute_frechet_distance(mu_real, mu_fake, sigma_real, sigma_fake, eps=1e-6)
        return fid_score

    def compute_frechet_distance(self, mu1, mu2, sigma1, sigma2, eps=1e-6):
        # Taken from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_1d(sigma1)
        sigma2 = np.atleast_1d(sigma2)

        assert mu1.shape == mu2.shape
        assert sigma1.shape == sigma2.shape

        diff = mu1 - mu2

        covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

    def compute_stats_for_dataset(self):
        if self.use_cache and 'mu' in self.cache and 'sigma' in self.cache:
            print('Use cached statistics for dataset...')
            return self.cache['mu'], self.cache['sigma']
        
        print()
        print('Compute statistics for dataset...')
        pbar = tqdm(total=self.num_images)
        image_count = 0

        activations = []
        for batch in utils.prefetch(self.dataset, n_prefetch=2):
            act = self.inception_apply(self.inception_params, jax.lax.stop_gradient(batch['image']))
            act = jnp.reshape(act, (self.num_devices * self.batch_size, -1))
            activations.append(act)

            pbar.update(self.num_devices * self.batch_size)
            image_count += self.num_devices * self.batch_size
            if image_count >= self.num_images:
                break
        pbar.close()

        activations = jnp.concatenate(activations, axis=0)
        activations = activations[:self.num_images]
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        self.cache['mu'] = mu
        self.cache['sigma'] = sigma
        return mu, sigma

    def compute_stats_for_generator(self, generator_params, seed_offset):
        print()
        print('Compute statistics for generator...')
        num_batches = int(np.ceil(self.num_images / (self.batch_size * self.num_devices))) 

        pbar = tqdm(total=self.num_images)
        activations = []

        for i in range(num_batches):
            rng = jax.random.PRNGKey(seed_offset + i)
            z_latent = jax.random.normal(rng, shape=(self.num_devices, self.batch_size, self.z_dim))

            labels = None
            if self.c_dim > 0:
                labels = jax.random.randint(rng, shape=(self.num_devices * self.batch_size,), minval=0, maxval=self.c_dim)
                labels = jax.nn.one_hot(labels, num_classes=self.c_dim)
                labels = jnp.reshape(labels, (self.num_devices, self.batch_size, self.c_dim))
            
            image, _ = self.generator_apply(generator_params, jax.lax.stop_gradient(z_latent), labels)
            image = (image - jnp.min(image)) / (jnp.max(image) - jnp.min(image))

            image = 2 * image - 1
            act = self.inception_apply(self.inception_params, jax.lax.stop_gradient(image))
            act = jnp.reshape(act, (self.num_devices * self.batch_size, -1))
            activations.append(act)
            pbar.update(self.num_devices * self.batch_size)
        pbar.close()

        activations = jnp.concatenate(activations, axis=0)
        activations = activations[:self.num_images]
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma


