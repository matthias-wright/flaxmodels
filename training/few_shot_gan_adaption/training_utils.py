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


def cosine_similarity(x, y, eps=1e-08):
    """
    Computes the cosine similarity between 'x' and 'y'.

    Args:
        x (DeviceArray): Array of shape [N, D].
        y (DeviceArray): Array of shape [M, D].

    Returns:
        (DeviceArray): Cosine similarity of shape [N, M].
    """
    return jnp.matmul(x, y.transpose()) / (jnp.maximum(jnp.linalg.norm(x, axis=1) * jnp.linalg.norm(y, axis=1), eps))


def kl_div(p, q, reduction='mean'):
    """
    Computes the KL divergence between 'p' and 'q'.

    Args:
        p (DeviceArray): Array of arbitrary shape in log-probabilities.
        q (DeviceArray): Array of arbitrary shape in log-probabilities.
        reduction (str): Method of reduction. Choices: 'mean', 'sum'.

    Returns:
        (DeviceArray): KL divergence.
    """
    if reduction == 'mean':
        reduce = jnp.mean
    elif reduction == 'sum':
        reduce = jnp.sum
    else:
        ValueError(f'Invalid argument passed to \'reduction\': {reduction}.')

    return reduce(jnp.exp(p) * (p - q))


def similarity_distribution(features):
    """
    Computes the pairwise similarities of a batch in the feature space.
    See Sec. 3.1 of the paper for details.

    Args:
        features (list): List of features.
    
    Returns:
        (DeviceArray): Similarity distribution in log space.
    """
    assert len(features) > 0, "Feature list should not be empty."

    batch_size = features[0].shape[0]
    cos_similarities = jnp.zeros((batch_size, batch_size))

    for i in range(len(features)):
        feat = jnp.reshape(features[i], (batch_size, -1))
        cos_similarities += cosine_similarity(feat, feat)
    return jax.nn.log_softmax(cos_similarities / len(features), axis=1)


def sample_z(sub_region, z_latent_anchors, std, num_devices, batch_size, z_dim, key, dtype):
    """
    Samples z_latent (Z).
    
    Args:
        sub_region (bool): If True, sample z_latent around anchor regions.
        z_latent_anchors (DeviceArray): z_latent anchor points, shape [N, z_dim].
        std (float): Standard deviation for sampling around anchor points.
        num_devices (int): Number of devices.
        batch_size (int): Batch size per device.
        z_dim (int): Dimensionality of z_latent.
        key (jax.random.PRNGKey): Random PRNG.
        dtype (str): Data type.
    """
    if sub_region:
        idcs = jax.random.randint(key, minval=0, maxval=z_latent_anchors.shape[0], shape=(num_devices, batch_size))
        key, _ = jax.random.split(key)
        z_latent = z_latent_anchors[tuple(idcs)] + jax.random.normal(key, (num_devices, batch_size, z_dim), dtype) * std
    else:
        z_latent = jax.random.normal(key, (num_devices, batch_size, z_dim), dtype)
    return z_latent


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
            if isinstance(src_child, DeviceArray) or isinstance(src_child, np.ndarray):
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
        source_params (Any): Parameters of the source generator.
    """
    apply_mapping: Callable = struct.field(pytree_node=False)
    apply_synthesis: Callable = struct.field(pytree_node=False)
    dynamic_scale_main: dynamic_scale_lib.DynamicScale
    dynamic_scale_reg: dynamic_scale_lib.DynamicScale
    epoch: int
    moving_stats: Any=None
    noise_consts: Any=None
    source_params: Any=None


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


def merge_params(params_trg, params_src):
    """
    Copies the parameters from 'params_src' into 'params_trg'.

    Args:
        params_trg (frozen_dict.FrozenDict): Parameter tree that will receive parameters from 'params_src'.
        params_src (frozen_dict.FrozenDict): Parameter tree that will be copied into 'params_trg'.
    Returns:
        (frozen_dict.FrozenDict): Updated version of 'params_trg'.
    """
    def _merge_params(trg, src):
        for name, src_child in src.items():
            if isinstance(src_child, DeviceArray) or isinstance(src_child, np.ndarray):
                trg[name] = jnp.array(src[name])
            else:
                _merge_params(trg[name], src_child)
    
    params_trg = params_trg.unfreeze()

    _merge_params(params_trg, params_src)

    params_trg = frozen_dict.freeze(params_trg)
    return params_trg
