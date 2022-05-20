import tensorflow as tf
import tensorflow_datasets as tfds
import jax
import jax.numpy as jnp
from jax.lib import xla_bridge
import flax
from flax.optim import dynamic_scale as dynamic_scale_lib
import flax.linen as nn
from flax.training import train_state
from flax.training import common_utils
from flax.training import checkpoints
from flax.training import lr_schedule
import optax
import numpy as np
import dataclasses
import functools
from tqdm import tqdm
from typing import Any
import argparse
import wandb
import os

import flaxmodels as fm


def cross_entropy_loss(logits, labels):
    """
    Computes the cross entropy loss.

    Args:
        logits (tensor): Logits, shape [B, num_classes].
        labels (tensor): Labels, shape [B,].

    Returns:
        (tensor): Cross entropy loss, shape [].
    """
    return -jnp.sum(common_utils.onehot(labels, num_classes=logits.shape[1]) * logits) / labels.shape[0]


def compute_metrics(logits, labels):
    """
    Computes the cross entropy loss and accuracy.

    Args:
        logits (tensor): Logits, shape [B, num_classes].
        labels (tensor): Labels, shape [B,].

    Returns:
        (dict): Dictionary containing the cross entropy loss and accuracy.
    """
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    metrics = {'loss': loss, 'accuracy': accuracy}
    return metrics


class TrainState(train_state.TrainState):
    """
    Simple train state for the common case with a single Optax optimizer.

    Attributes:
        batch_stats (Any): Collection used to store an exponential moving
                           average of the batch statistics.
        dynamic_scale (dynamic_scale_lib.DynamicScale): Dynamic loss scaling for mixed precision gradients.
        epoch (int): Current epoch.
    """
    batch_stats: Any
    dynamic_scale: dynamic_scale_lib.DynamicScale
    epoch: int


def restore_checkpoint(state, path):
    """
    Restores checkpoint with best validation score.

    Args:
        state (train_state.TrainState): Training state.
        path (str): Path to checkpoint.

    Returns:
        (train_state.TrainState): Training state from checkpoint.
    """
    return checkpoints.restore_checkpoint(path, state)


def save_checkpoint(state, step_or_metric, path):
    """
    Saves a checkpoint from the given state.

    Args:
        state (train_state.TrainState): Training state.
        step_or_metric (int of float): Current training step or metric to identify the checkpoint.
        path (str): Path to the checkpoint directory.

    """
    if jax.process_index() == 0:
        state = jax.device_get(jax.tree_map(lambda x: x[0], state))
        checkpoints.save_checkpoint(path, state, step_or_metric, keep=3)


def sync_batch_stats(state):
    """
    Sync the batch statistics across devices.

    Args:
        state (train_state.TrainState): Training state.
    
    Returns:
        (train_state.TrainState): Updated training state.
    """
    cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')
    return state.replace(batch_stats=cross_replica_mean(state.batch_stats))


def configure_dataloader(ds, prerocess, num_devices, batch_size):
    # https://www.tensorflow.org/tutorials/load_data/images
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.map(lambda x, y: (prerocess(x), y), tf.data.AUTOTUNE)
    ds = ds.batch(batch_size=num_devices * batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def train_step(state, batch):

    def loss_fn(params):
        logits, new_model_state = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
                                                 batch['image'],
                                                 mutable=['batch_stats'])
        loss = cross_entropy_loss(logits, batch['label'])
        return loss, (new_model_state, logits)

    dynamic_scale = state.dynamic_scale

    if dynamic_scale:
        grad_fn = dynamic_scale.value_and_grad(loss_fn, has_aux=True, axis_name='batch')
        dynamic_scale, is_fin, aux, grads = grad_fn(state.params)
        # dynamic loss takes care of averaging gradients across replicas
    else:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        aux, grads = grad_fn(state.params)
        # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
        grads = jax.lax.pmean(grads, axis_name='batch')
    new_model_state, logits = aux[1]
    metrics = compute_metrics(logits, batch['label'])

    new_state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    
    if dynamic_scale:
        # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
        # params should be restored (= skip this step).
        new_state = new_state.replace(opt_state=jax.tree_multimap(functools.partial(jnp.where, is_fin),
                                                                  new_state.opt_state,
                                                                  state.opt_state),
                                      params=jax.tree_multimap(functools.partial(jnp.where, is_fin),
                                                               new_state.params,
                                                               state.params))
        metrics['scale'] = dynamic_scale.scale

    return new_state, metrics


def eval_step(state, batch):
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    logits = state.apply_fn(variables, batch['image'], train=False, mutable=False)
    return compute_metrics(logits, batch['label'])


def train_and_evaluate(config):
    num_devices = jax.device_count()

    #--------------------------------------
    # Data
    #--------------------------------------
    def train_prerocess(x):
        x = tf.image.random_crop(x, size=(config.img_size, config.img_size, config.img_channels))
        x = tf.image.random_flip_up_down(x)
        x = tf.image.random_flip_left_right(x)
        # Cast to float because if the image has data type int, the following augmentations will convert it
        # to float then apply the transformations and convert it back to int.
        x = tf.cast(x, dtype='float32')
        x = tf.image.random_brightness(x, max_delta=0.5)
        x = tf.image.random_contrast(x, lower=0.1, upper=1.0)
        x = tf.image.random_hue(x, max_delta=0.5)
        x = (x - 127.5) / 127.5
        return x

    def val_prerocess(x):
        x = tf.expand_dims(x, axis=0)
        #x = tf.keras.layers.experimental.preprocessing.CenterCrop(height=config.img_size, width=config.img_size)(x)
        x = tf.image.random_crop(x, size=(x.shape[0], config.img_size, config.img_size, config.img_channels))
        x = tf.squeeze(x, axis=0)
        x = tf.cast(x, dtype='float32')
        x = (x - 127.5) / 127.5
        return x

    ds_train = tfds.load('imagenette/320px-v2',
                         split='train',
                         as_supervised=True,
                         shuffle_files=True,
                         data_dir=config.data_dir)
    ds_val = tfds.load('imagenette/320px-v2',
                       split='validation',
                       as_supervised=True,
                       shuffle_files=True,
                       data_dir=config.data_dir)

    dataset_size = ds_train.__len__().numpy()

    ds_train = configure_dataloader(ds_train, train_prerocess, num_devices, config.batch_size)
    ds_val = configure_dataloader(ds_val, val_prerocess, num_devices, config.batch_size)


    #--------------------------------------
    # Seeding, Devices, and Precision
    #--------------------------------------
    rng = jax.random.PRNGKey(config.random_seed)

    if config.mixed_precision:
        dtype = jnp.float16
    else:
        dtype = jnp.float32

    platform = jax.local_devices()[0].platform
    if config.mixed_precision and platform == 'gpu':
        dynamic_scale = dynamic_scale_lib.DynamicScale()
    else:
        dynamic_scale = None


    #--------------------------------------
    # Initialize Models
    #--------------------------------------
    rng, init_rng = jax.random.split(rng)
    
    if config.arch == 'resnet18':
        model = fm.ResNet18(output='log_softmax', pretrained=None, num_classes=config.num_classes, dtype=dtype)
    elif config.arch == 'resnet34':
        model = fm.ResNet34(output='log_softmax', pretrained=None, num_classes=config.num_classes, dtype=dtype)
    elif config.arch == 'resnet50':
        model = fm.ResNet50(output='log_softmax', pretrained=None, num_classes=config.num_classes, dtype=dtype)
    elif config.arch == 'resnet101':
        model = fm.ResNet101(output='log_softmax', pretrained=None, num_classes=config.num_classes, dtype=dtype)
    elif config.arch == 'resnet152':
        model = fm.ResNet152(output='log_softmax', pretrained=None, num_classes=config.num_classes, dtype=dtype)

    variables = model.init(init_rng, jnp.ones((1, config.img_size, config.img_size, config.img_channels), dtype=dtype))
    params, batch_stats = variables['params'], variables['batch_stats']
    
    #--------------------------------------
    # Initialize Optimizer
    #--------------------------------------
    steps_per_epoch = dataset_size // config.batch_size

    learning_rate_fn = lr_schedule.create_cosine_learning_rate_schedule(config.learning_rate,
                                                                        steps_per_epoch,
                                                                        config.num_epochs - config.warmup_epochs,
                                                                        config.warmup_epochs)

    tx = optax.adam(learning_rate=learning_rate_fn)

    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              tx=tx,
                              batch_stats=batch_stats,
                              dynamic_scale=dynamic_scale,
                              epoch=0)
    
    step = 0
    epoch_offset = 0
    if config.resume:
        ckpt_path = checkpoints.latest_checkpoint(config.ckpt_dir)
        state = restore_checkpoint(state, ckpt_path)
        step = jax.device_get(state.step)
        epoch_offset = jax.device_get(state.epoch)
    
    state = flax.jax_utils.replicate(state)
    
    #--------------------------------------
    # Create train and eval steps
    #--------------------------------------
    p_train_step = jax.pmap(functools.partial(train_step), axis_name='batch')
    p_eval_step = jax.pmap(eval_step, axis_name='batch')


    #--------------------------------------
    # Training 
    #--------------------------------------

    best_val_acc = 0.0

    for epoch in range(epoch_offset, config.num_epochs):
        pbar = tqdm(total=dataset_size)

        accuracy = 0.0
        n = 0
        for image, label in ds_train.as_numpy_iterator():
            pbar.update(num_devices * config.batch_size)
            image = image.astype(dtype)
            label = label.astype(dtype)

            if image.shape[0] % num_devices != 0:
                # Batch size must be divisible by the number of devices
                continue

            # Reshape images from [num_devices * batch_size, height, width, img_channels]
            # to [num_devices, batch_size, height, width, img_channels].
            # The first dimension will be mapped across devices with jax.pmap.
            image = jnp.reshape(image, (num_devices, -1) + image.shape[1:])
            label = jnp.reshape(label, (num_devices, -1) + label.shape[1:])

            state, metrics = p_train_step(state, {'image': image, 'label': label})
            accuracy += metrics['accuracy']
            n += 1

            if step % config.log_every == 0:
                if config.wandb:
                    if 'scale' in metrics:
                        wandb.log({'training/scale': jnp.mean(metrics['scale']).item()}, step=step, commit=False)
                    wandb.log({'training/accuracy': jnp.mean(metrics['accuracy']).item()}, step=step)
            step += 1

        pbar.close()
        accuracy /= n

        print(f'Epoch: {epoch}')
        print('Training accuracy:', jnp.mean(accuracy))
        
        #--------------------------------------
        # Validation 
        #--------------------------------------
        # Sync batch stats
        state = sync_batch_stats(state)

        accuracy = 0.0
        n = 0
        for image, label in ds_val.as_numpy_iterator():
            image = image.astype(dtype)
            label = label.astype(dtype)
            if image.shape[0] % num_devices != 0:
                continue
            
            # Reshape images from [num_devices * batch_size, height, width, img_channels]
            # to [num_devices, batch_size, height, width, img_channels].
            # The first dimension will be mapped across devices with jax.pmap.
            image = jnp.reshape(image, (num_devices, -1) + image.shape[1:])
            label = jnp.reshape(label, (num_devices, -1) + label.shape[1:])
            metrics = p_eval_step(state, {'image': image, 'label': label})
            accuracy += metrics['accuracy']
            n += 1
        accuracy /= n
        print('Validation accuracy:', jnp.mean(accuracy))
        accuracy = jnp.mean(accuracy).item()

        if accuracy > best_val_acc:
            best_val_acc = accuracy
            state = dataclasses.replace(state, **{'step': flax.jax_utils.replicate(step), 'epoch': flax.jax_utils.replicate(epoch)})
            save_checkpoint(state, jnp.mean(accuracy).item(), config.ckpt_dir)

        if config.wandb:
            wandb.log({'validation/accuracy': jnp.mean(accuracy).item()}, step=step)

