import flax.linen as nn
from flax.core import FrozenDict
import jax.numpy as jnp


#---------------------------------------------------------------#
# Normalization
#---------------------------------------------------------------#


def batch_norm(x, train, epsilon=1e-05, momentum=0.1, params=None):
    # if params is not None, train must be false
    if params is None:
        x = nn.BatchNorm(epsilon=epsilon, momentum=momentum, use_running_average=not train)(x)
    else:
        params_ = FrozenDict({'batch_stats': {'mean': jnp.array(params['mean']), 'var': jnp.array(params['var'])}, 
                              'params': {'scale': jnp.array(params['scale']), 'bias': jnp.array(params['bias'])}})
        x = nn.BatchNorm(epsilon=epsilon,
                         momentum=momentum,
                         use_running_average=True).apply(params_, x) 
    return x


