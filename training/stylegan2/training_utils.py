import jax
import jax.numpy as jnp
from jaxlib.xla_extension import DeviceArray
import flax
from flax.optim import dynamic_scale as dynamic_scale_lib
from flax.core import frozen_dict
from flax.training import train_state
from flax import struct
import numpy as np
from PIL import Image
from typing import Any, Callable


def sync_moving_stats(state):
    """
    Sync moving statistics across devices.

    Args:
        state (train_state.TrainState): Training state.

    Returns:
        (train_state.TrainState): Updated training state.
    """
    cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')
    return state.replace(moving_stats=cross_replica_mean(state.moving_stats))


def update_generator_ema(state_G, params_ema_G, config, ema_beta=None):
    """
    Update exponentially moving average of the generator weights.
    Moving stats and noise constants will be copied over.
    
    Args:
        state_G (train_state.TrainState): Generator state.
        params_ema_G (frozen_dict.FrozenDict): Parameters of the ema generator.
        config (Any): Config object.
        ema_beta (float): Beta parameter of the ema. If None, will be computed
                          from 'ema_nimg' and 'batch_size'.

    Returns:
        (frozen_dict.FrozenDict): Updates parameters of the ema generator.
    """
    def _update_ema(src, trg, beta):
        for name, src_child in src.items():
            if isinstance(src_child, DeviceArray):
                trg[name] = src[name] + ema_beta * (trg[name] - src[name])
            else:
                _update_ema(src_child, trg[name], beta)
    
    if ema_beta is None:
        ema_nimg = config.ema_kimg * 1000
        ema_beta = 0.5 ** (config.batch_size / max(ema_nimg, 1e-8))

    params_ema_G = params_ema_G.unfreeze()

    # Copy over moving stats
    params_ema_G['moving_stats']['mapping_network'] = state_G.moving_stats
    params_ema_G['noise_consts']['synthesis_network'] = state_G.noise_consts 
    
    # Update exponentially moving average of the trainable parameters
    _update_ema(state_G.params['mapping'], params_ema_G['params']['mapping_network'], ema_beta)
    _update_ema(state_G.params['synthesis'], params_ema_G['params']['synthesis_network'], ema_beta)

    params_ema_G = frozen_dict.freeze(params_ema_G)
    return params_ema_G


class TrainStateG(train_state.TrainState):
    """
    Generator train state for a single Optax optimizer.

    Attributes:
        apply_mapping (Callable): Apply function of the Mapping Network.
        apply_synthesis (Callable): Apply function of the Synthesis Network.
        dynamic_scale (dynamic_scale_lib.DynamicScale): Dynamic loss scaling for mixed precision gradients.
        epoch (int): Current epoch.
        moving_stats (Any): Moving average of the latent W. 
        noise_consts (Any): Noise constants from synthesis layers.
    """
    apply_mapping: Callable = struct.field(pytree_node=False)
    apply_synthesis: Callable = struct.field(pytree_node=False)
    dynamic_scale_main: dynamic_scale_lib.DynamicScale
    dynamic_scale_reg: dynamic_scale_lib.DynamicScale
    epoch: int
    moving_stats: Any=None
    noise_consts: Any=None


class TrainStateD(train_state.TrainState):
    """
    Discriminator train state for a single Optax optimizer.

    Attributes:
        dynamic_scale (dynamic_scale_lib.DynamicScale): Dynamic loss scaling for mixed precision gradients.
        epoch (int): Current epoch.
    """
    dynamic_scale_main: dynamic_scale_lib.DynamicScale
    dynamic_scale_reg: dynamic_scale_lib.DynamicScale
    epoch: int


def get_training_snapshot(image_real, image_gen, max_num=10):
    """
    Creates a snapshot of generated images and real images.
    
    Args:
        images_real (DeviceArray): Batch of real images, shape [B, H, W, C].
        images_gen (DeviceArray): Batch of generated images, shape [B, H, W, C].
        max_num (int): Maximum number of images used for snapshot.

    Returns:
        (PIL.Image): Training snapshot. Top row: generated images, bottom row: real images.
    """
    if image_real.shape[0] > max_num:
        image_real = image_real[:max_num]
    if image_gen.shape[0] > max_num:
        image_gen = image_gen[:max_num]

    image_real = jnp.split(image_real, image_real.shape[0], axis=0)
    image_gen = jnp.split(image_gen, image_gen.shape[0], axis=0)

    image_real = [jnp.squeeze(x, axis=0) for x in image_real]
    image_gen = [jnp.squeeze(x, axis=0) for x in image_gen]

    image_real = jnp.concatenate(image_real, axis=1)
    image_gen = jnp.concatenate(image_gen, axis=1)

    image_gen = (image_gen - np.min(image_gen)) / (np.max(image_gen) - np.min(image_gen))
    image_real = (image_real - np.min(image_real)) / (np.max(image_real) - np.min(image_real))
    image = jnp.concatenate((image_gen, image_real), axis=0)
    
    image = np.uint8(image * 255)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return Image.fromarray(image)


def get_eval_snapshot(image, max_num=10):
    """
    Creates a snapshot of generated images.

    Args:
        image (DeviceArray): Generated images, shape [B, H, W, C].

    Returns:
        (PIL.Image): Eval snapshot.
    """
    if image.shape[0] > max_num:
        image = image[:max_num]

    image = jnp.split(image, image.shape[0], axis=0)
    image = [jnp.squeeze(x, axis=0) for x in image]
    image = jnp.concatenate(image, axis=1)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = np.uint8(image * 255)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return Image.fromarray(image)
