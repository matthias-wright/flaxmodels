import tensorflow as tf
import tensorflow_datasets as tfds
import jax
import jax.numpy as jnp
import flax
from flax.core import frozen_dict
import optax
import numpy as np
import functools
from tqdm import tqdm
import argparse
import wandb
import os
from PIL import Image

import flaxmodels as fm
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
        dynamic_scale_G_main = flax.optim.DynamicScale()
        dynamic_scale_D_main = flax.optim.DynamicScale()
        dynamic_scale_G_reg = flax.optim.DynamicScale()
        dynamic_scale_D_reg = flax.optim.DynamicScale()
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
    
    # Generator initialization for training
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
                                                  num_fp16_res=num_fp16_res,
                                                  clip_conv=clip_conv,
                                                  dtype=dtype)

    rng, init_rng = jax.random.split(rng)
    mapping_net_vars = mapping_net.init(init_rng,
                                        jnp.ones((1, config.z_dim)),
                                        jnp.ones((1, config.c_dim)))

    mapping_net_params, moving_stats = mapping_net_vars['params'], mapping_net_vars['moving_stats']

    synthesis_net_vars = synthesis_net.init(init_rng,
                                            jnp.ones((1, mapping_net.num_ws, config.w_dim)))
    synthesis_net_params, noise_consts = synthesis_net_vars['params'], synthesis_net_vars['noise_consts']

    params_G = frozen_dict.FrozenDict({'mapping': mapping_net_params,
                                       'synthesis': synthesis_net_params})

    # Discriminator initialization for training
    discriminator = fm.stylegan2.Discriminator(resolution=config.resolution,
                                               num_channels=config.img_channels,
                                               c_dim=config.c_dim,
                                               mbstd_group_size=config.mbstd_group_size,
                                               num_fp16_res=num_fp16_res,
                                               clip_conv=clip_conv,
                                               dtype=dtype)
    rng, init_rng = jax.random.split(rng)
    params_D = discriminator.init(init_rng,
                                  jnp.ones((1, config.resolution, config.resolution, config.img_channels)),
                                  jnp.ones((1, config.c_dim)))
    
    # Exponential average Generator initialization
    generator_ema = fm.stylegan2.Generator(resolution=config.resolution,
                                           num_channels=config.img_channels,
                                           z_dim=config.z_dim,
                                           c_dim=config.c_dim,
                                           w_dim=config.w_dim,
                                           num_ws=int(np.log2(config.resolution)) * 2 - 3,
                                           num_mapping_layers=8,
                                           fmap_base=config.fmap_base,
                                           num_fp16_res=num_fp16_res,
                                           clip_conv=clip_conv,
                                           dtype=dtype)

    params_ema_G = generator_ema.init(init_rng,
                                      jnp.ones((1, config.z_dim)),
                                      jnp.ones((1, config.c_dim)))


    #--------------------------------------
    # Initialize States and Optimizers
    #--------------------------------------
    print('Initialize states...')
    tx_G = optax.adam(learning_rate=config.learning_rate, b1=0.0, b2=0.99)
    tx_D = optax.adam(learning_rate=config.learning_rate, b1=0.0, b2=0.99)
    
    state_G = training_utils.TrainStateG.create(apply_fn=None,
                                                apply_mapping=mapping_net.apply,
                                                apply_synthesis=synthesis_net.apply,
                                                params=params_G,
                                                moving_stats=moving_stats,
                                                noise_consts=noise_consts,
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
    
    # Copy over the parameters from the training generator to the ema generator
    params_ema_G = training_utils.update_generator_ema(state_G, params_ema_G, config, ema_beta=0)

    # Running mean of path length for path length regularization
    pl_mean = jnp.zeros((), dtype=dtype)
    pl_mean = flax.jax_utils.replicate(pl_mean)

    # Replicate states across devices 
    state_G = flax.jax_utils.replicate(state_G)
    state_D = flax.jax_utils.replicate(state_D)
    
    step = 0
    epoch_offset = 0
    best_fid_score = np.inf
    if config.resume:
        ckpt_path = checkpoint.get_latest_checkpoint(config.ckpt_dir)
        if ckpt_path is None:
            print('Could not find checkpoint, start training from scratch.')
        else:
            print('Resume training from checkpoint:', ckpt_path)
            ckpt = checkpoint.load_checkpoint(ckpt_path)
            step = ckpt['step']
            epoch_offset = ckpt['epoch']
            best_fid_score = ckpt['fid_score']
            pl_mean = ckpt['pl_mean']
            state_G = ckpt['state_G']
            state_D = ckpt['state_D']
            params_ema_G = ckpt['params_ema_G']
            config = ckpt['config']

    
    #--------------------------------------
    # Precompile train and eval steps
    #--------------------------------------
    print('Precompile training steps...')
    p_main_step_G = jax.pmap(training_steps.main_step_G, axis_name='batch')
    p_regul_step_G = jax.pmap(functools.partial(training_steps.regul_step_G, config=config), axis_name='batch')

    p_main_step_D = jax.pmap(training_steps.main_step_D, axis_name='batch')
    p_regul_step_D = jax.pmap(functools.partial(training_steps.regul_step_D, config=config), axis_name='batch')
    
    #--------------------------------------
    # Training 
    #--------------------------------------
    print('Start training...')
    fid_metric = FID(generator_ema, ds_train, config)
    
    # Dict to collect training statistics / losses
    metrics = {}

    for epoch in range(epoch_offset, config.num_epochs):
        pbar = tqdm(total=dataset_info['num_examples'])

        for batch in data_pipeline.prefetch(ds_train, config.num_prefetch):
            pbar.update(num_devices * config.batch_size)

            if config.c_dim == 0:
                # No labels in the dataset
                batch['label'] = None
            
            # Create two latent noise vectors and combine them for the style mixing regularization
            rng, key = jax.random.split(rng)
            z_latent1 = jax.random.normal(key, (num_devices, config.batch_size, config.z_dim), dtype)

            rng, key = jax.random.split(rng)
            z_latent2 = jax.random.normal(key, (num_devices, config.batch_size, config.z_dim), dtype)

            # Split PRNGs across devices
            rkey = jax.random.split(key, num=num_devices)
            mixing_prob = flax.jax_utils.replicate(config.mixing_prob)

            #--------------------------------------
            # Update Discriminator
            #--------------------------------------
            state_D, metrics = p_main_step_D(state_G, state_D, batch, z_latent1, z_latent2, metrics, mixing_prob, rkey)
            if step % config.D_reg_interval == 0:
                state_D, metrics = p_regul_step_D(state_D, batch, metrics)

            #--------------------------------------
            # Update Generator
            #--------------------------------------
            state_G, metrics = p_main_step_G(state_G, state_D, batch, z_latent1, z_latent2, metrics, mixing_prob, rkey)
            if step % config.G_reg_interval == 0:
                H, W = batch['image'].shape[1], batch['image'].shape[2]
                rng, key = jax.random.split(rng)
                pl_noise = jax.random.normal(key, batch['image'].shape, dtype=dtype) / np.sqrt(H * W)
                state_G, metrics, pl_mean = p_regul_step_G(state_G, batch, z_latent1, pl_noise, pl_mean, metrics, rng=rkey)

            params_ema_G = training_utils.update_generator_ema(flax.jax_utils.unreplicate(state_G),
                                                               params_ema_G,
                                                               config)
            
            #--------------------------------------
            # Logging and Checkpointing
            #--------------------------------------
            if step % config.save_every == 0 and config.disable_fid:
                # If FID evaluation is disabled, a checkpoint will be saved every 'save_every' steps.
                if jax.process_index() == 0:
                    checkpoint.save_checkpoint(config.ckpt_dir, state_G, state_D, params_ema_G, pl_mean, config, step, epoch)

            if step % config.eval_fid_every == 0 and not config.disable_fid:
                # If FID evaluation is enabled, only save a checkpoint if FID score is better.
                if jax.process_index() == 0:
                    print()
                    print('Compute FID...')
                    fid_score = fid_metric.compute_fid(params_ema_G).item()
                    if config.wandb:
                        wandb.log({'training/gen/fid': fid_score}, step=step)
                    print('FID:', fid_score)
                    print()
                    if fid_score < best_fid_score:
                        best_fid_score = fid_score
                        checkpoint.save_checkpoint(config.ckpt_dir, state_G, state_D, params_ema_G, pl_mean, config, step, epoch, fid_score=fid_score)


            if step % config.log_every == 0:
                if config.wandb:
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
            labels = None if config.c_dim == 0 else batch['label'][0]
            image_gen_eval = training_steps.eval_step_G(generator_ema, params=params_ema_G, z_latent=z_latent1[0], labels=labels, truncation=1)
            image_gen_eval_trunc = training_steps.eval_step_G(generator_ema, params=params_ema_G, z_latent=z_latent1[0], labels=labels, truncation=0.5)
            eval_snapshot = training_utils.get_eval_snapshot(image=image_gen_eval, max_num=10)
            eval_snapshot_trunc = training_utils.get_eval_snapshot(image=image_gen_eval_trunc, max_num=10)
            wandb.log({'eval/snapshot': wandb.Image(eval_snapshot)}, commit=False)
            wandb.log({'eval/snapshot_trunc': wandb.Image(eval_snapshot_trunc)})

        pbar.close()

