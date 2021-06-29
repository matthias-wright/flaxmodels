import flax.linen as nn
from flax.core import FrozenDict
import jax.numpy as jnp

from flax.linen.module import compact, merge_param
from typing import (Any, Callable, Optional, Tuple)
from jax.nn import initializers
from jax import lax

PRNGKey = Any
Array = Any
Shape = Tuple[int]
Dtype = Any


#---------------------------------------------------------------#
# Normalization
#---------------------------------------------------------------#
def batch_norm(x, train, epsilon=1e-05, momentum=0.1, params=None, dtype='float32'):
    if params is None:
        x = BatchNorm(epsilon=epsilon,
                      momentum=momentum,
                      use_running_average=not train,
                      dtype=dtype)(x)
    else:
        x = BatchNorm(epsilon=epsilon,
                      momentum=momentum,
                      bias_init=lambda *_ : jnp.array(params['bias']),
                      scale_init=lambda *_ : jnp.array(params['scale']),
                      mean_init=lambda *_ : jnp.array(params['mean']),
                      var_init=lambda *_ : jnp.array(params['var']),
                      use_running_average=not train,
                      dtype=dtype)(x)
    return x


def _absolute_dims(rank, dims):
    return tuple([rank + dim if dim < 0 else dim for dim in dims])


class BatchNorm(nn.Module):
    """BatchNorm Module.

    Taken from: https://github.com/google/flax/blob/master/flax/linen/normalization.py

    Attributes:
        use_running_average: if True, the statistics stored in batch_stats
                             will be used instead of computing the batch statistics on the input.
    axis: the feature or non-batch axis of the input.
    momentum: decay rate for the exponential moving average of the batch statistics.
    epsilon: a small float added to variance to avoid dividing by zero.
    dtype: the dtype of the computation (default: float32).
    use_bias:  if True, bias (beta) is added.
    use_scale: if True, multiply by scale (gamma).
               When the next layer is linear (also e.g. nn.relu), this can be disabled
               since the scaling will be done by the next layer.
    bias_init: initializer for bias, by default, zero.
    scale_init: initializer for scale, by default, one.
    axis_name: the axis name used to combine batch statistics from multiple
               devices. See `jax.pmap` for a description of axis names (default: None).
    axis_index_groups: groups of axis indices within that named axis
                       representing subsets of devices to reduce over (default: None). For
                       example, `[[0, 1], [2, 3]]` would independently batch-normalize over
                       the examples on the first two and last two devices. See `jax.lax.psum`
                       for more details.
    """
    use_running_average: Optional[bool] = None
    axis: int = -1
    momentum: float = 0.99
    epsilon: float = 1e-5
    dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones
    mean_init: Callable[[Shape], Array] = lambda s: jnp.zeros(s, jnp.float32)
    var_init: Callable[[Shape], Array] = lambda s: jnp.ones(s, jnp.float32)
    axis_name: Optional[str] = None
    axis_index_groups: Any = None

    @compact
    def __call__(self, x, use_running_average: Optional[bool] = None):
        """Normalizes the input using batch statistics.
        
        NOTE:
        During initialization (when parameters are mutable) the running average
        of the batch statistics will not be updated. Therefore, the inputs
        fed during initialization don't need to match that of the actual input
        distribution and the reduction axis (set with `axis_name`) does not have
        to exist.
        Args:
            x: the input to be normalized.
            use_running_average: if true, the statistics stored in batch_stats
                                 will be used instead of computing the batch statistics on the input.
        Returns:
            Normalized inputs (the same shape as inputs).
        """
        use_running_average = merge_param(
            'use_running_average', self.use_running_average, use_running_average)
        x = jnp.asarray(x, jnp.float32)
        axis = self.axis if isinstance(self.axis, tuple) else (self.axis,)
        axis = _absolute_dims(x.ndim, axis)
        feature_shape = tuple(d if i in axis else 1 for i, d in enumerate(x.shape))
        reduced_feature_shape = tuple(d for i, d in enumerate(x.shape) if i in axis)
        reduction_axis = tuple(i for i in range(x.ndim) if i not in axis)

        # see NOTE above on initialization behavior
        initializing = self.is_mutable_collection('params')

        ra_mean = self.variable('batch_stats', 'mean',
                                self.mean_init,
                                reduced_feature_shape)
        ra_var = self.variable('batch_stats', 'var',
                               self.var_init,
                               reduced_feature_shape)

        if use_running_average:
            mean, var = ra_mean.value, ra_var.value
        else:
            mean = jnp.mean(x, axis=reduction_axis, keepdims=False)
            mean2 = jnp.mean(lax.square(x), axis=reduction_axis, keepdims=False)
            if self.axis_name is not None and not initializing:
                concatenated_mean = jnp.concatenate([mean, mean2])
                mean, mean2 = jnp.split(
                    lax.pmean(
                        concatenated_mean,
                        axis_name=self.axis_name,
                        axis_index_groups=self.axis_index_groups), 2)
            var = mean2 - lax.square(mean)

            if not initializing:
                ra_mean.value = self.momentum * ra_mean.value + (1 - self.momentum) * mean
                ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var

        y = x - mean.reshape(feature_shape)
        mul = lax.rsqrt(var + self.epsilon)
        if self.use_scale:
            scale = self.param('scale',
                               self.scale_init,
                               reduced_feature_shape).reshape(feature_shape)
            mul = mul * scale
        y = y * mul
        if self.use_bias:
            bias = self.param('bias',
                              self.bias_init,
                              reduced_feature_shape).reshape(feature_shape)
            y = y + bias
        return jnp.asarray(y, self.dtype)


