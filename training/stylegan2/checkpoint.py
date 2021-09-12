import flax
import dill as pickle
import os
import glob


def save_checkpoint(ckpt_dir, state_G, state_D, params_ema_G, pl_mean, config, step, epoch, fid_score=None, keep=2):
    """
    Saves checkpoint.

    Args:
        ckpt_dir (str): Path to the directory, where checkpoints are saved.
        state_G (train_state.TrainState): Generator state.
        state_D (train_state.TrainState): Discriminator state.
        params_ema_G (frozen_dict.FrozenDict): Parameters of the ema generator.
        pl_mean (array): Moving average of the path length (generator regularization).
        config (argparse.Namespace): Configuration.
        step (int): Current step.
        epoch (int): Current epoch.
        fid_score (float): FID score corresponding to the checkpoint.
        keep (int): Number of checkpoints to keep.
    """
    state_dict = {'state_G': flax.jax_utils.unreplicate(state_G),
                  'state_D': flax.jax_utils.unreplicate(state_D),
                  'params_ema_G': params_ema_G,
                  'pl_mean': flax.jax_utils.unreplicate(pl_mean),
                  'config': config,
                  'fid_score': fid_score,
                  'step': step,
                  'epoch': epoch}

    with open(os.path.join(ckpt_dir, f'ckpt_{step}.pickle'), 'wb') as handle:
        pickle.dump(state_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ckpts = glob.glob(os.path.join(ckpt_dir, '*.pickle'))
    if len(ckpts) > keep:
        oldest_ckpt = min(ckpts, key=os.path.getctime)
        os.remove(oldest_ckpt)


def load_checkpoint(filename):
    """
    Loads checkpoints.

    Args:
        filename (str): Path to the checkpoint file.

    Returns:
        (dict): Checkpoint.
    """
    state_dict = pickle.load(open(filename, 'rb'))
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

