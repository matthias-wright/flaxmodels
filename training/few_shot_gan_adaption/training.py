import tensorflow as tf
import tensorflow_datasets as tfds
import jax
import jax.numpy as jnp
import flax
from flax.optim import dynamic_scale as dynamic_scale_lib
from flax.core import frozen_dict
import optax
import numpy as np
import functools
from tqdm import tqdm
import argparse
import wandb
import os
from PIL import Image
import copy

# TODO
#import flaxmodels as fm
import model

import data_pipeline
import checkpoint
import training_utils
import training_steps
from fid import FID


def train_and_evaluate(config):
    num_devices = jax.device_count()

    #--------------------------------------
    # Data
    #--------------------------------------
    ds_train, dataset_info = data_pipeline.get_data(data_dir=config.data_dir,
                                                    img_size=config.resolution,
                                                    img_channels=config.img_channels,
                                                    num_classes=config.c_dim,
                                                    num_devices=num_devices,
                                                    batch_size=config.batch_size)


    #--------------------------------------
    # Seeding and Precision
    #--------------------------------------
    rng = jax.random.PRNGKey(config.random_seed)

    if config.mixed_precision:
        dtype = jnp.float16
    else:
        dtype = jnp.float32

    platform = jax.local_devices()[0].platform
    if config.mixed_precision and platform == 'gpu':
        dynamic_scale_G_main = dynamic_scale_lib.DynamicScale()
        dynamic_scale_D_main = dynamic_scale_lib.DynamicScale()
        dynamic_scale_G_reg = dynamic_scale_lib.DynamicScale()
        dynamic_scale_D_reg = dynamic_scale_lib.DynamicScale()
        clip_conv = 256
        num_fp16_res = 4
    else:
        dynamic_scale_G_main = None
        dynamic_scale_D_main = None
        dynamic_scale_G_reg = None
        dynamic_scale_D_reg = None
        clip_conv = None
        num_fp16_res = 0

    #--------------------------------------
    # Initialize Models
    #--------------------------------------
    print('Initialize models...')

    # Load checkpoint for source model
    ckpt_source = checkpoint.load_checkpoint(config.source_ckpt_path, replicate=False)
    config_source = ckpt_source['config']
    
    # Generator initialization for training
    mapping_net = model.MappingNetwork(z_dim=config_source.z_dim,
                                       c_dim=config_source.c_dim,
                                       w_dim=config_source.w_dim,
                                       num_ws=int(np.log2(config_source.resolution)) * 2 - 3,
                                       num_layers=8,
                                       dtype=dtype)

    synthesis_net = model.SynthesisNetwork(resolution=config_source.resolution,
                                           num_channels=config_source.img_channels,
                                           w_dim=config_source.w_dim,
                                           fmap_base=config_source.fmap_base,
                                           num_fp16_res=num_fp16_res,
                                           clip_conv=clip_conv,
                                           dtype=dtype)

    rng, init_rng = jax.random.split(rng)
    mapping_net_vars = mapping_net.init(init_rng,
                                        jnp.ones((1, config_source.z_dim)),
                                        jnp.ones((1, config_source.c_dim)))

    mapping_net_params, moving_stats = mapping_net_vars['params'], mapping_net_vars['moving_stats']

    synthesis_net_vars = synthesis_net.init(init_rng,
                                            jnp.ones((1, mapping_net.num_ws, config_source.w_dim)))
    synthesis_net_params, noise_consts = synthesis_net_vars['params'], synthesis_net_vars['noise_consts']

    params_G = frozen_dict.FrozenDict({'mapping': mapping_net_params,
                                       'synthesis': synthesis_net_params})

    # Discriminator initialization for training
    discriminator = model.Discriminator(resolution=config_source.resolution,
                                        num_channels=config_source.img_channels,
                                        c_dim=config_source.c_dim,
                                        mbstd_group_size=config_source.mbstd_group_size,
                                        num_fp16_res=num_fp16_res,
                                        clip_conv=clip_conv,
                                        dtype=dtype)

    rng, init_rng = jax.random.split(rng)
    params_D = discriminator.init(init_rng,
                                  jnp.ones((1, config_source.resolution, config_source.resolution, config_source.img_channels)),
                                  jnp.ones((1, config_source.c_dim)))
    
    # Exponential average Generator initialization
    generator_ema = model.Generator(resolution=config_source.resolution,
                                    num_channels=config_source.img_channels,
                                    z_dim=config_source.z_dim,
                                    c_dim=config_source.c_dim,
                                    w_dim=config_source.w_dim,
                                    num_ws=int(np.log2(config_source.resolution)) * 2 - 3,
                                    num_mapping_layers=8,
                                    fmap_base=config_source.fmap_base,
                                    num_fp16_res=num_fp16_res,
                                    clip_conv=clip_conv,
                                    dtype=dtype)

    params_ema_G = generator_ema.init(init_rng,
                                      jnp.ones((1, config_source.z_dim)),
                                      jnp.ones((1, config_source.c_dim)))

    # Source Generator initialization
    generator_source = model.Generator(resolution=config_source.resolution,
                                       num_channels=config_source.img_channels,
                                       z_dim=config_source.z_dim,
                                       c_dim=config_source.c_dim,
                                       w_dim=config_source.w_dim,
                                       num_ws=int(np.log2(config_source.resolution)) * 2 - 3,
                                       num_mapping_layers=8,
                                       fmap_base=config_source.fmap_base,
                                       num_fp16_res=num_fp16_res,
                                       clip_conv=clip_conv,
                                       dtype=dtype)

    params_source_G = generator_source.init(init_rng,
                                            jnp.ones((1, config_source.z_dim)),
                                            jnp.ones((1, config_source.c_dim)))


    #--------------------------------------
    # Initialize States and Optimizers
    #--------------------------------------
    print('Initialize states...')
    tx_G = optax.adam(learning_rate=config.learning_rate, b1=0.0, b2=0.99)
    tx_D = optax.adam(learning_rate=config.learning_rate, b1=0.0, b2=0.99)
    
    state_G = ckpt_source['state_G']
    params_ema_G = ckpt_source['params_ema_G']
    state_D = ckpt_source['state_D']
    params_D = training_utils.merge_params(params_D, state_D.params)

    # Copy over the parameters from the training generator to the source generator
    params_source_G = training_utils.update_generator_ema(state_G, params_source_G, config_source, ema_beta=0)
    
    state_G = training_utils.TrainStateG.create(apply_fn=generator_source.apply,
                                                apply_mapping=mapping_net.apply,
                                                apply_synthesis=synthesis_net.apply,
                                                params=state_G.params,
                                                source_params=params_source_G,
                                                moving_stats=state_G.moving_stats,
                                                noise_consts=state_G.noise_consts,
                                                tx=tx_G,
                                                dynamic_scale_main=dynamic_scale_G_main,
                                                dynamic_scale_reg=dynamic_scale_G_reg,
                                                epoch=0)

    state_D = training_utils.TrainStateD.create(apply_fn=discriminator.apply,
                                                params=params_D,
                                                tx=tx_D,
                                                dynamic_scale_main=dynamic_scale_D_main,
                                                dynamic_scale_reg=dynamic_scale_D_reg,
                                                epoch=0)

    # Running mean of path length for path length regularization
    pl_mean = jnp.zeros((), dtype=dtype)

    # Sample the anchor points. When sampling noise close to the anchor points, use the image-level adversarial loss (Eq. 4 in the paper)
    rng, key = jax.random.split(rng)
    z_latent_anchors = jax.random.normal(rng, (dataset_info['num_examples'], config_source.z_dim), dtype)

    # TODO
    import dill as pickle
    _ckpt = pickle.load(open('logging/default/sketches_noise/checkpoints/ckpt_8000.pickle', 'rb'))
    z_latent_anchors = _ckpt['z_latent_anchors']


    step = 0
    epoch_offset = 0
    best_fid_score = np.inf

    # Replicate states across devices 
    pl_mean = flax.jax_utils.replicate(pl_mean)
    state_G = flax.jax_utils.replicate(state_G)
    state_D = flax.jax_utils.replicate(state_D)

    #--------------------------------------
    # Precompile train and eval steps
    #--------------------------------------
    print('Precompile training steps...')
    p_main_step_G = jax.pmap(training_steps.main_step_G, axis_name='batch')
    p_regul_step_G = jax.pmap(functools.partial(training_steps.regul_step_G, config=config_source), axis_name='batch')
    p_dist_step_G = jax.pmap(functools.partial(training_steps.dist_step_G, config=config), axis_name='batch')

    p_main_step_D = jax.pmap(training_steps.main_step_D, static_broadcasted_argnums=7, axis_name='batch')
    p_regul_step_D = jax.pmap(functools.partial(training_steps.regul_step_D, config=config_source), static_broadcasted_argnums=3, axis_name='batch')
    
    #--------------------------------------
    # Training 
    #--------------------------------------
    print('Start training...')
    fid_metric = FID(generator_ema, ds_train, config_source)
    
    # Dict to collect training statistics / losses
    metrics = {}

    for epoch in range(epoch_offset, config.num_epochs):
        pbar = tqdm(total=dataset_info['num_examples'])

        for batch in data_pipeline.prefetch(ds_train, config.num_prefetch):
            pbar.update(num_devices * config.batch_size)

            if config_source.c_dim == 0:
                # No labels in the dataset
                batch['label'] = None

            # If True, sample z_latent around anchor regions
            sub_space = step % config.subspace_freq == 0
            
            # Create two latent noise vectors and combine them for the style mixing regularization
            rng, key = jax.random.split(rng)
            #z_latent1 = jax.random.normal(key, (num_devices, config.batch_size, config_source.z_dim), dtype)
            z_latent1 = training_utils.sample_z(sub_space, z_latent_anchors, config.subspace_std, num_devices, config.batch_size, config_source.z_dim, key, dtype)

            rng, key = jax.random.split(rng)
            z_latent2 = training_utils.sample_z(sub_space, z_latent_anchors, config.subspace_std, num_devices, config.batch_size, config_source.z_dim, key, dtype)

            # Split PRNGs across devices
            rkey = jax.random.split(key, num=num_devices)
            mixing_prob = flax.jax_utils.replicate(config_source.mixing_prob)

            #--------------------------------------
            # Update Discriminator
            #--------------------------------------
            state_D, metrics = p_main_step_D(state_G, state_D, batch, z_latent1, z_latent2, metrics, mixing_prob, sub_space, rkey)
            if step % config_source.D_reg_interval == 0:
                state_D, metrics = p_regul_step_D(state_D, batch, metrics, sub_space)

            #--------------------------------------
            # Update Generator
            #--------------------------------------
            state_G, metrics = p_main_step_G(state_G, state_D, batch, z_latent1, z_latent2, metrics, mixing_prob, rkey)

            state_G, metrics = p_dist_step_G(state_G, batch, z_latent1, metrics, rng=rkey)
            if step % config_source.G_reg_interval == 0:
                H, W = batch['image'].shape[1], batch['image'].shape[2]
                rng, key = jax.random.split(rng)
                pl_noise = jax.random.normal(key, batch['image'].shape, dtype=dtype) / np.sqrt(H * W)
                state_G, metrics, pl_mean = p_regul_step_G(state_G, batch, z_latent1, pl_noise, pl_mean, metrics, rng=rkey)

            params_ema_G = training_utils.update_generator_ema(flax.jax_utils.unreplicate(state_G),
                                                               params_ema_G,
                                                               config_source)
            
            #--------------------------------------
            # Logging and Checkpointing
            #--------------------------------------
            if step % config.save_every == 0 and config.disable_fid:
                # If FID evaluation is disabled, a checkpoint will be saved every 'save_every' steps.
                if jax.process_index() == 0:
                    checkpoint.save_checkpoint(config.ckpt_dir, state_G, state_D, params_ema_G, config, step, z_latent_anchors)

            if step % config.eval_fid_every == 0 and not config.disable_fid:
                # If FID evaluation is enabled, only save a checkpoint if FID score is better.
                if jax.process_index() == 0:
                    print()
                    print('Compute FID...')
                    fid_score = fid_metric.compute_fid(params_ema_G).item()
                    print('FID:', fid_score)
                    print()
                    if fid_score < best_fid_score:
                        best_fid_score = fid_score
                        checkpoint.save_checkpoint(config.ckpt_dir, state_G, state_D, params_ema_G, config, step, z_latent_anchors)

            if step % config.log_every == 0:
                if config.wandb:
                    if not config.disable_fid: wandb.log({'training/gen/fid': fid_score}, commit=False)
                    wandb.log({'training/gen/loss': jnp.mean(metrics['G_loss']).item()}, step=step, commit=False)
                    wandb.log({'training/dis/loss': jnp.mean(metrics['D_loss']).item()}, step=step, commit=False)
                    wandb.log({'training/dis/fake_logits': jnp.mean(metrics['fake_logits']).item()}, step=step, commit=False)
                    wandb.log({'training/dis/real_logits': jnp.mean(metrics['real_logits']).item()}, step=step, commit=False)
                    train_snapshot = training_utils.get_training_snapshot(image_real=flax.jax_utils.unreplicate(batch['image']),
                                                                          image_gen=flax.jax_utils.unreplicate(metrics['image_gen']), max_num=10)
                    wandb.log({'training/snapshot': wandb.Image(train_snapshot)})

            step += 1
        
        # Sync moving stats across devices
        state_G = training_utils.sync_moving_stats(state_G)
        
        # Sync moving average of path length mean (Generator regularization)
        pl_mean = jax.pmap(lambda x: jax.lax.pmean(x, axis_name='batch'), axis_name='batch')(pl_mean)
        
        # Generate evaluation images after epoch
        if config.wandb:
            labels = None if config_source.c_dim == 0 else batch['label'][0]
            image_gen_eval = training_steps.eval_step_G(generator_ema, params=params_ema_G, z_latent=z_latent1[0], labels=labels, truncation=1)
            image_gen_eval_trunc = training_steps.eval_step_G(generator_ema, params=params_ema_G, z_latent=z_latent1[0], labels=labels, truncation=0.5)
            eval_snapshot = training_utils.get_eval_snapshot(image=image_gen_eval, max_num=10)
            eval_snapshot_trunc = training_utils.get_eval_snapshot(image=image_gen_eval_trunc, max_num=10)
            wandb.log({'eval/snapshot': wandb.Image(eval_snapshot)}, step=step, commit=False)
            wandb.log({'eval/snapshot_trunc': wandb.Image(eval_snapshot_trunc)}, step=step)

        pbar.close()

