import flax
import dill as pickle
import os
import glob


def save_checkpoint(ckpt_dir, state_G, state_D, params_ema_G, config, step, z_latent_anchors, keep=2):
    """
    Saves checkpoint.

    Args:
        ckpt_dir (str): Path to the directory, where checkpoints are saved.
        state_G (train_state.TrainState): Generator state.
        state_D (train_state.TrainState): Discriminator state.
        params_ema_G (frozen_dict.FrozenDict): Parameters of the ema generator.
        config (argparse.Namespace): Configuration.
        step (int): Current step.
        z_latent_anchors (DeviceArray): Noise anchors.
        keep (int): Number of checkpoints to keep.
    """
    
    state_G = flax.jax_utils.unreplicate(state_G)
    state_D = flax.jax_utils.unreplicate(state_D)
    params_G = {'params': {'mapping_network': state_G.params.unfreeze()['mapping'], 'synthesis_network': state_G.params.unfreeze()['synthesis']},
                'moving_stats': {'mapping_network': state_G.moving_stats},
                'noise_consts': {'synthesis_network': state_G.noise_consts}}

    ckpt_dict = {'params_G': params_G,
                 'params_D': state_D.params,
                 'params_ema_G': params_ema_G,
                 'z_latent_anchors': z_latent_anchors,
                 'config': config}

    with open(os.path.join(ckpt_dir, f'ckpt_{step}.pickle'), 'wb') as handle:
        pickle.dump(ckpt_dict, handle, protocol=pickle.DEFAULT_PROTOCOL)

    ckpts = glob.glob(os.path.join(ckpt_dir, '*.pickle'))
    if len(ckpts) > keep:
        oldest_ckpt = min(ckpts, key=os.path.getctime)
        os.remove(oldest_ckpt)


def load_checkpoint(filename, replicate=True):
    """
    Loads checkpoints.

    Args:
        filename (str): Path to the checkpoint file.
        replicate (bool): If True, replicate parameters across devices.

    Returns:
        (dict): Checkpoint.
    """
    state_dict = pickle.load(open(filename, 'rb'))
    if replicate:
        state_dict['state_G'] = flax.jax_utils.replicate(state_dict['state_G'])
        state_dict['state_D'] = flax.jax_utils.replicate(state_dict['state_D'])
        state_dict['pl_mean'] = flax.jax_utils.replicate(state_dict['pl_mean'])
    return state_dict


def get_latest_checkpoint(ckpt_dir):
    """
    Returns the path of the latest checkpoint.

    Args:
        ckpt_dir (str): Path to the directory, where checkpoints are saved.

    Returns:
        (str): Path to latest checkpoint (if it exists).
    """
    ckpts = glob.glob(os.path.join(ckpt_dir, '*.pickle'))
    if len(ckpts) == 0:
        return None
    latest_ckpt = max(ckpts, key=os.path.getctime)
    return latest_ckpt

