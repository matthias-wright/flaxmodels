import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from jax import jit
import numpy as np
from functools import partial
from typing import Any
import h5py


#------------------------------------------------------
# Other 
#------------------------------------------------------
def minibatch_stddev_layer(x, group_size=None, num_new_features=1):
    if group_size is None:
        group_size = x.shape[0]
    else:
        # Minibatch must be divisible by (or smaller than) group_size.
        group_size = min(group_size, x.shape[0])

    G = group_size
    F = num_new_features
    _, H, W, C = x.shape
    c = C // F

    # [NHWC] Cast to FP32.
    y = x.astype(jnp.float32)
    # [GnHWFc] Split minibatch N into n groups of size G, and channels C into F groups of size c.
    y = jnp.reshape(y, newshape=(G, -1, H, W, F, c))
    # [GnHWFc] Subtract mean over group.
    y -= jnp.mean(y, axis=0)
    # [nHWFc] Calc variance over group.
    y = jnp.mean(jnp.square(y), axis=0)
    # [nHWFc] Calc stddev over group.
    y = jnp.sqrt(y + 1e-8)
    # [nF] Take average over channels and pixels.
    y = jnp.mean(y, axis=(1, 2, 4))
    # [nF] Cast back to original data type.
    y = y.astype(x.dtype)
    # [n11F] Add missing dimensions.
    y = jnp.reshape(y, newshape=(-1, 1, 1, F))
    # [NHWC] Replicate over group and pixels.
    y = jnp.tile(y, (G, H, W, 1))
    return jnp.concatenate((x, y), axis=3)


#------------------------------------------------------
# Activation 
#------------------------------------------------------
def apply_activation(x, activation='linear', alpha=0.2, gain=np.sqrt(2)):
    gain = jnp.array(gain, dtype=x.dtype)
    if activation == 'relu':
        return jax.nn.relu(x) * gain
    if activation == 'leaky_relu':
        return jax.nn.leaky_relu(x, negative_slope=alpha) * gain
    return x


#------------------------------------------------------
# Weights 
#------------------------------------------------------
def get_weight(shape, lr_multiplier=1, bias=True, param_dict=None, layer_name='', key=None):
    if param_dict is None:
        w = random.normal(key, shape=shape, dtype=jnp.float32) / lr_multiplier
        if bias: b = jnp.zeros(shape=(shape[-1],), dtype=jnp.float32)
    else:
        w = jnp.array(param_dict[layer_name]['weight']).astype(jnp.float32)
        if bias: b = jnp.array(param_dict[layer_name]['bias']).astype(jnp.float32)
    
    if bias: return w, b
    return w


def equalize_lr_weight(w, lr_multiplier=1):
    """
    Equalized learning rate, see: https://arxiv.org/pdf/1710.10196.pdf.

    Args:
        w (tensor): Weight parameter. Shape [kernel, kernel, fmaps_in, fmaps_out]
                    for convolutions and shape [in, out] for MLPs.
        lr_multiplier (float): Learning rate multiplier.

    Returns:
        (tensor): Scaled weight parameter.
    """
    in_features = np.prod(w.shape[:-1])
    gain = lr_multiplier / np.sqrt(in_features)
    w *= gain
    return w


def equalize_lr_bias(b, lr_multiplier=1):
    """
    Equalized learning rate, see: https://arxiv.org/pdf/1710.10196.pdf.

    Args:
        b (tensor): Bias parameter.
        lr_multiplier (float): Learning rate multiplier.

    Returns:
        (tensor): Scaled bias parameter.
    """
    gain = lr_multiplier
    b *= gain
    return b


#------------------------------------------------------
# Normalization
#------------------------------------------------------
def normalize_2nd_moment(x, eps=1e-8):
    return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=1, keepdims=True) + eps)


#------------------------------------------------------
# Upsampling
#------------------------------------------------------
def setup_filter(f, normalize=True, flip_filter=False, gain=1, separable=None):
    """
    Convenience function to setup 2D FIR filter for `upfirdn2d()`.

    Args:
        f (tensor): Tensor or python list of the shape.
        normalize (bool): Normalize the filter so that it retains the magnitude.
                          for constant input signal (DC)? (default: True).
        flip_filter (bool): Flip the filter? (default: False).
        gain (int): Overall scaling factor for signal magnitude (default: 1).
        separable: Return a separable filter? (default: select automatically).

    Returns:
        (tensor): Output filter of shape [filter_height, filter_width] or [filter_taps]
    """
    # Validate.
    if f is None:
        f = 1
    f = jnp.array(f, dtype=jnp.float32)
    assert f.ndim in [0, 1, 2]
    assert f.size > 0
    if f.ndim == 0:
        f = f[jnp.newaxis]

    # Separable?
    if separable is None:
        separable = (f.ndim == 1 and f.size >= 8)
    if f.ndim == 1 and not separable:
        f = jnp.outer(f, f)
    assert f.ndim == (1 if separable else 2)

    # Apply normalize, flip, gain, and device.
    if normalize:
        f /= jnp.sum(f)
    if flip_filter:
        for i in range(f.ndim):
            f = jnp.flip(f, axis=i)
    f = f * (gain ** (f.ndim / 2))
    return f


def upfirdn2d(x, f, padding=(2, 1, 2, 1), up=1, down=1, strides=(1, 1), flip_filter=False, gain=1):

    if f is None:
        f = jnp.ones((1, 1), dtype=jnp.float32)

    B, H, W, C = x.shape
    padx0, padx1, pady0, pady1 = padding

    # upsample by inserting zeros
    x = jnp.reshape(x, newshape=(B, H, 1, W, 1, C))
    x = jnp.pad(x, pad_width=((0, 0), (0, 0), (0, up - 1), (0, 0), (0, up - 1), (0, 0)))
    x = jnp.reshape(x, newshape=(B, H * up, W * up, C))

    # padding
    x = jnp.pad(x, pad_width=((0, 0), (max(pady0, 0), max(pady1, 0)), (max(padx0, 0), max(padx1, 0)), (0, 0)))
    x = x[:, max(-pady0, 0) : x.shape[1] - max(-pady1, 0), max(-padx0, 0) : x.shape[2] - max(-padx1, 0)]

    # setup filter
    f = f * (gain ** (f.ndim / 2))
    if not flip_filter:
        for i in range(f.ndim):
            f = jnp.flip(f, axis=i)

    # convole filter
    f = jnp.repeat(jnp.expand_dims(f, axis=(-2, -1)), repeats=C, axis=-1)
    if f.ndim == 4:
        x = jax.lax.conv_general_dilated(x,
                                         f.astype(x.dtype),
                                         window_strides=strides or (1,) * (x.ndim - 2),
                                         padding='valid',
                                         dimension_numbers=nn.linear._conv_dimension_numbers(x.shape),
                                         feature_group_count=C)
    else:
        x = jax.lax.conv_general_dilated(x,
                                         jnp.expand_dims(f, axis=0).astype(x.dtype),
                                         window_strides=strides or (1,) * (x.ndim - 2),
                                         padding='valid',
                                         dimension_numbers=nn.linear._conv_dimension_numbers(x.shape),
                                         feature_group_count=C)
        x = jax.lax.conv_general_dilated(x,
                                         jnp.expand_dims(f, axis=1).astype(x.dtype),
                                         window_strides=strides or (1,) * (x.ndim - 2),
                                         padding='valid',
                                         dimension_numbers=nn.linear._conv_dimension_numbers(x.shape),
                                         feature_group_count=C)
    x = x[:, ::down, ::down]
    return x


def upsample2d(x, f, up=2, padding=0, flip_filter=False, gain=1):
    if f.ndim == 1:
        fh, fw = f.shape[0], f.shape[0]
    elif f.ndim == 2:
        fh, fw = f.shape[0], f.shape[1]
    else:
        raise ValueError('Invalid filter shape:', f.shape)
    padx0 = padding + (fw + up - 1) // 2
    padx1 = padding + (fw - up) // 2
    pady0 = padding + (fh + up - 1) // 2
    pady1 = padding + (fh - up) // 2
    return upfirdn2d(x, f=f, up=up, padding=(padx0, padx1, pady0, pady1), flip_filter=flip_filter, gain=gain * up * up)


#------------------------------------------------------
# Linear 
#------------------------------------------------------
class LinearLayer(nn.Module):
    """
    Linear Layer.

    Attributes:
        in_features (int): Input dimension.
        out_features (int): Output dimension.
        use_bias (bool): If True, use bias.
        bias_init (int): Bias init.
        lr_multiplier (float): Learning rate multiplier.
        activation (str): Activation function: 'relu', 'lrelu', etc.
        param_dict (h5py.Group): Parameter dict with pretrained parameters.
        layer_name (str): Layer name.
        dtype (str): Data type.
        rng (jax.random.PRNGKey): Random seed for initialization.
    """
    in_features: int
    out_features: int
    use_bias: bool=True
    bias_init: int=0
    lr_multiplier: float=1
    activation: str='linear'
    param_dict: h5py.Group=None
    layer_name: str=None
    dtype: str='float32'
    rng: Any=random.PRNGKey(0)

    @nn.compact
    def __call__(self, x):
        """
        Run Linear Layer.
        
        Args:
            x (tensor): Input tensor of shape [N, in_features].
            
        Returns:
            (tensor): Output tensor of shape [N, out_features].
        """
        w_shape = [self.in_features, self.out_features]
        params = get_weight(w_shape, self.lr_multiplier, self.use_bias, self.param_dict, self.layer_name, self.rng)

        if self.use_bias:
            w, b = params
        else:
            w = params

        w = self.param(name='weight', init_fn=lambda *_ : w)
        w = equalize_lr_weight(w, self.lr_multiplier)
        x = jnp.matmul(x, w.astype(x.dtype))

        if self.use_bias:
            b = self.param(name='bias', init_fn=lambda *_ : b)
            b = equalize_lr_bias(b, self.lr_multiplier)
            x += b.astype(x.dtype)
            x += self.bias_init
        
        x = apply_activation(x, activation=self.activation)
        return x


#------------------------------------------------------
# Convolution
#------------------------------------------------------
def conv_downsample_2d(x, w, k=None, factor=2, gain=1, padding=0):
    """
    Fused downsample convolution.

    Padding is performed only once at the beginning, not between the operations.
    The fused op is considerably more efficient than performing the same calculation
    using standard TensorFlow ops. It supports gradients of arbitrary order.

    Args:
        x (tensor): Input tensor of the shape [N, H, W, C].
        w (tensor): Weight tensor of the shape [filterH, filterW, inChannels, outChannels].
                    Grouped convolution can be performed by inChannels = x.shape[0] // numGroups.
        k (tensor): FIR filter of the shape [firH, firW] or [firN].
                    The default is `[1] * factor`, which corresponds to average pooling.
        factor (int): Downsampling factor (default: 2).
        gain (float): Scaling factor for signal magnitude (default: 1.0).
        padding (int): Number of pixels to pad or crop the output on each side (default: 0).

    Returns:
        (tensor): Output of the shape [N, H // factor, W // factor, C].
    """
    assert isinstance(factor, int) and factor >= 1
    assert isinstance(padding, int)

    # Check weight shape.
    ch, cw, _inC, _outC = w.shape
    assert cw == ch

    # Setup filter kernel.
    k = setup_filter(k, gain=gain)
    assert k.shape[0] == k.shape[1]

    # Execute.
    pad0 = (k.shape[0] - factor + cw) // 2 + padding * factor
    pad1 = (k.shape[0] - factor + cw - 1) // 2 + padding * factor
    x = upfirdn2d(x=x, f=k, padding=(pad0, pad0, pad1, pad1))
    
    x = jax.lax.conv_general_dilated(x,
                                     w,
                                     window_strides=(factor, factor),
                                     padding='VALID',
                                     dimension_numbers=nn.linear._conv_dimension_numbers(x.shape))
    return x


def upsample_conv_2d(x, w, k=None, factor=2, gain=1, padding=0):
    """
    Fused upsample convolution.

    Padding is performed only once at the beginning, not between the operations.
    The fused op is considerably more efficient than performing the same calculation
    using standard TensorFlow ops. It supports gradients of arbitrary order.

    Args:
        x (tensor): Input tensor of the shape [N, H, W, C].
        w (tensor): Weight tensor of the shape [filterH, filterW, inChannels, outChannels].
                    Grouped convolution can be performed by inChannels = x.shape[0] // numGroups.
        k (tensor): FIR filter of the shape [firH, firW] or [firN].
                    The default is [1] * factor, which corresponds to nearest-neighbor upsampling.
        factor (int): Integer upsampling factor (default: 2).
        gain (float): Scaling factor for signal magnitude (default: 1.0).
        padding (int): Number of pixels to pad or crop the output on each side (default: 0).

    Returns:
        (tensor): Output of the shape [N, H * factor, W * factor, C].
    """
    assert isinstance(factor, int) and factor >= 1
    assert isinstance(padding, int)

    # Check weight shape.
    ch, cw, _inC, _outC = w.shape
    inC = w.shape[2] 
    outC = w.shape[3]
    assert cw == ch

    # Fast path for 1x1 convolution.
    if cw == 1 and ch == 1:
        x = jax.lax.conv_general_dilated(x,
                                         w,
                                         window_strides=(1, 1),
                                         padding='VALID',
                                         dimension_numbers=nn.linear._conv_dimension_numbers(x.shape))
        k = setup_filter(k, gain=gain * (factor ** 2))
        pad0 = (k.shape[0] + factor - cw) // 2 + padding
        pad1 = (k.shape[0] - factor) // 2 + padding
        x = upfirdn2d(x, f=k, up=factor, padding=(pad0, pad1, pad0, pad1))
        return x

    # Setup filter kernel.
    k = setup_filter(k, gain=gain * (factor ** 2))
    assert k.shape[0] == k.shape[1]

    # Determine data dimensions.
    stride = (factor, factor)
    output_shape = ((x.shape[1] - 1) * factor + ch, (x.shape[2] - 1) * factor + cw)
    num_groups = x.shape[3] // inC
    
    # Transpose weights.
    w = jnp.reshape(w, (ch, cw, inC, num_groups, -1))
    w = jnp.transpose(w[::-1, ::-1], (0, 1, 4, 3, 2))
    w = jnp.reshape(w, (ch, cw, -1, num_groups * inC))
    
    # Execute.
    x = gradient_based_conv_transpose(lhs=x, 
                                      rhs=w, 
                                      strides=stride,
                                      padding='VALID',
                                      output_padding=(0, 0, 0, 0),
                                      output_shape=output_shape,
                                      )

    pad0 = (k.shape[0] + factor - cw) // 2 + padding
    pad1 = (k.shape[0] - factor - cw + 3) // 2 + padding
    x = upfirdn2d(x=x, f=k, padding=(pad0, pad1, pad0, pad1))
    return x


def conv2d(x, w, up=False, down=False, resample_kernel=None, padding=0):
    assert not (up and down)
    kernel = w.shape[0]
    assert w.shape[1] == kernel
    assert kernel >= 1 and kernel % 2 == 1
    
    num_groups = x.shape[3] // w.shape[2]

    w = w.astype(x.dtype)
    if up:
        x = upsample_conv_2d(x, w, k=resample_kernel, padding=padding)
    elif down:
        x = conv_downsample_2d(x, w, k=resample_kernel, padding=padding)
    else:
        padding_mode = {0: 'SAME', -(kernel // 2): 'VALID'}[padding]
        x = jax.lax.conv_general_dilated(x,
                                         w,
                                         window_strides=(1, 1),
                                         padding=padding_mode,
                                         dimension_numbers=nn.linear._conv_dimension_numbers(x.shape),
                                         feature_group_count=num_groups)
    return x


def modulated_conv2d_layer(x, w, s, fmaps, kernel, up=False, down=False, demodulate=True, resample_kernel=None, fused_modconv=False):
    assert not (up and down)
    assert kernel >= 1 and kernel % 2 == 1
    
    # Get weight.
    wshape = (kernel, kernel, x.shape[3], fmaps)
    if x.dtype.name == 'float16' and not fused_modconv and demodulate:
        w *= jnp.sqrt(1 / np.prod(wshape[:-1])) / jnp.max(jnp.abs(w), axis=(0, 1, 2)) # Pre-normalize to avoid float16 overflow.
    ww = w[jnp.newaxis] # [BkkIO] Introduce minibatch dimension.

    # Modulate.
    if x.dtype.name == 'float16' and not fused_modconv and demodulate:
        s *= 1 / jnp.max(jnp.abs(s)) # Pre-normalize to avoid float16 overflow.
    ww *= s[:, jnp.newaxis, jnp.newaxis, :, jnp.newaxis].astype(w.dtype) # [BkkIO] Scale input feature maps.

    # Demodulate.
    if demodulate:
        d = jax.lax.rsqrt(jnp.sum(jnp.square(ww), axis=(1, 2, 3)) + 1e-8) # [BO] Scaling factor.
        ww *= d[:, jnp.newaxis, jnp.newaxis, jnp.newaxis, :] # [BkkIO] Scale output feature maps.

    # Reshape/scale input.
    if fused_modconv:
        x = jnp.transpose(x, axes=(0, 3, 1, 2))
        x = jnp.reshape(x, (1, -1, x.shape[2], x.shape[3])) # Fused => reshape minibatch to convolution groups.
        x = jnp.transpose(x, axes=(0, 2, 3, 1))
        w = jnp.reshape(jnp.transpose(ww, (1, 2, 3, 0, 4)), (ww.shape[1], ww.shape[2], ww.shape[3], -1))
    else:
        x *= s[:, jnp.newaxis, jnp.newaxis].astype(x.dtype)  # [BIhw] Not fused => scale input activations.

    # 2D convolution.
    x = conv2d(x, w.astype(x.dtype), up=up, down=down, resample_kernel=resample_kernel)

    # Reshape/scale output.
    if fused_modconv:
        x = jnp.transpose(x, axes=(0, 3, 1, 2))
        x = jnp.reshape(x, (-1, fmaps, x.shape[2], x.shape[3])) # Fused => reshape convolution groups back to minibatch.
        x = jnp.transpose(x, axes=(0, 2, 3, 1))
    elif demodulate:
        x *= d[:, jnp.newaxis, jnp.newaxis].astype(x.dtype) # [BOhw] Not fused => scale output activations.
    
    return x


def _deconv_output_length(input_length, filter_size, padding, output_padding=None, stride=0, dilation=1):
    """
    Taken from: https://github.com/google/jax/pull/5772/commits

    Determines the output length of a transposed convolution given the input length.
    Function modified from Keras.
    Arguments:
      input_length: Integer.
      filter_size: Integer.
      padding: one of `"SAME"`, `"VALID"`, or a 2-integer tuple.
      output_padding: Integer, amount of padding along the output dimension. Can
        be set to `None` in which case the output length is inferred.
      stride: Integer.
      dilation: Integer.
    Returns:
      The output length (integer).
    """
    if input_length is None:
        return None

    # Get the dilated kernel size
    filter_size = filter_size + (filter_size - 1) * (dilation - 1)

    # Infer length if output padding is None, else compute the exact length
    if output_padding is None:
        if padding == 'VALID':
            length = input_length * stride + max(filter_size - stride, 0)
        elif padding == 'SAME':
            length = input_length * stride
        else:
            length = ((input_length - 1) * stride + filter_size - padding[0] - padding[1])

    else:
        if padding == 'SAME':
            pad = filter_size // 2
            total_pad = pad * 2
        elif padding == 'VALID':
            total_pad = 0
        else:
            total_pad = padding[0] + padding[1]

    length = ((input_length - 1) * stride + filter_size - total_pad + output_padding)
    return length


def _compute_adjusted_padding(input_size, output_size, kernel_size, stride, padding, dilation=1):
    """
    Taken from: https://github.com/google/jax/pull/5772/commits

    Computes adjusted padding for desired ConvTranspose `output_size`.
    Ported from DeepMind Haiku.
    """
    kernel_size = (kernel_size - 1) * dilation + 1
    if padding == 'VALID':
        expected_input_size = (output_size - kernel_size + stride) // stride
        if input_size != expected_input_size:
            raise ValueError(f'The expected input size with the current set of input '
                             f'parameters is {expected_input_size} which doesn\'t '
                             f'match the actual input size {input_size}.')
        padding_before = 0
    elif padding == 'SAME':
        expected_input_size = (output_size + stride - 1) // stride
        if input_size != expected_input_size:
            raise ValueError(f'The expected input size with the current set of input '
                             f'parameters is {expected_input_size} which doesn\'t '
                             f'match the actual input size {input_size}.')
        padding_needed = max(0, (input_size - 1) * stride + kernel_size - output_size)
        padding_before = padding_needed // 2
    else:
        padding_before = padding[0]  # type: ignore[assignment]

    expanded_input_size = (input_size - 1) * stride + 1
    padded_out_size = output_size + kernel_size - 1
    pad_before = kernel_size - 1 - padding_before
    pad_after = padded_out_size - expanded_input_size - pad_before
    return (pad_before, pad_after)


def _flip_axes(x, axes):
    """
    Taken from: https://github.com/google/jax/blob/master/jax/_src/lax/lax.py 

    Flip ndarray 'x' along each axis specified in axes tuple.
    """
    for axis in axes:
        x = jnp.flip(x, axis)
    return x


def gradient_based_conv_transpose(lhs, 
                                  rhs, 
                                  strides,
                                  padding,
                                  output_padding,
                                  output_shape=None,
                                  dilation=None,
                                  dimension_numbers=None,
                                  transpose_kernel=True,
                                  feature_group_count=1, 
                                  precision=None):
    """
    Taken from: https://github.com/google/jax/pull/5772/commits

    Convenience wrapper for calculating the N-d transposed convolution.
    Much like `conv_transpose`, this function calculates transposed convolutions
    via fractionally strided convolution rather than calculating the gradient
    (transpose) of a forward convolution. However, the latter is more common
    among deep learning frameworks, such as TensorFlow, PyTorch, and Keras.
    This function provides the same set of APIs to help reproduce results in these frameworks.
    Args:
        lhs: a rank `n+2` dimensional input array.
        rhs: a rank `n+2` dimensional array of kernel weights.
        strides: sequence of `n` integers, amounts to strides of the corresponding forward convolution.
        padding: `"SAME"`, `"VALID"`, or a sequence of `n` integer 2-tuples that controls
                 the before-and-after padding for each `n` spatial dimension of
                 the corresponding forward convolution.
        output_padding: A sequence of integers specifying the amount of padding along
                        each spacial dimension of the output tensor, used to disambiguate the output shape of
                        transposed convolutions when the stride is larger than 1.
                        (see a detailed description at https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html)
                        The amount of output padding along a given dimension must
                        be lower than the stride along that same dimension.
                        If set to `None` (default), the output shape is inferred.
                        If both `output_padding` and `output_shape` are specified, they have to be mutually compatible.
        output_shape: Output shape of the spatial dimensions of a transpose
                      convolution. Can be `None` or an iterable of `n` integers. If a `None` value is given (default),
                      the shape is automatically calculated.
                      Similar to `output_padding`, `output_shape` is also for disambiguating the output shape
                      when stride > 1 (see also
                      https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose)
                      If both `output_padding` and `output_shape` are specified, they have to be mutually compatible.
        dilation: `None`, or a sequence of `n` integers, giving the
                   dilation factor to apply in each spatial dimension of `rhs`. Dilated convolution
                   is also known as atrous convolution.
        dimension_numbers: tuple of dimension descriptors as in lax.conv_general_dilated. Defaults to tensorflow convention.
        transpose_kernel: if `True` flips spatial axes and swaps the input/output
                          channel axes of the kernel. This makes the output of this function identical
                          to the gradient-derived functions like keras.layers.Conv2DTranspose and
                          torch.nn.ConvTranspose2d applied to the same kernel.
                          Although for typical use in neural nets this is unnecessary
                          and makes input/output channel specification confusing, you need to set this to `True`
                          in order to match the behavior in many deep learning frameworks, such as TensorFlow, Keras, and PyTorch.
        precision: Optional. Either ``None``, which means the default precision for
                   the backend, a ``lax.Precision`` enum value (``Precision.DEFAULT``,
                   ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple of two
                   ``lax.Precision`` enums indicating precision of ``lhs``` and ``rhs``.
    Returns:
        Transposed N-d convolution.
    """
    assert len(lhs.shape) == len(rhs.shape) and len(lhs.shape) >= 2
    ndims = len(lhs.shape)
    one = (1,) * (ndims - 2)
    # Set dimensional layout defaults if not specified.
    if dimension_numbers is None:
        if ndims == 2:
            dimension_numbers = ('NC', 'IO', 'NC')
        elif ndims == 3:
            dimension_numbers = ('NHC', 'HIO', 'NHC')
        elif ndims == 4:
            dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
        elif ndims == 5:
            dimension_numbers = ('NHWDC', 'HWDIO', 'NHWDC')
        else:
            raise ValueError('No 4+ dimensional dimension_number defaults.')
    dn = jax.lax.conv_dimension_numbers(lhs.shape, rhs.shape, dimension_numbers)
    k_shape = np.take(rhs.shape, dn.rhs_spec)
    k_sdims = k_shape[2:]  # type: ignore[index]
    i_shape = np.take(lhs.shape, dn.lhs_spec)
    i_sdims = i_shape[2:]  # type: ignore[index]

    # Calculate correct output shape given padding and strides.
    if dilation is None:
        dilation = (1,) * (rhs.ndim - 2)

    if output_padding is None:
        output_padding = [None] * (rhs.ndim - 2)  # type: ignore[list-item]

    if isinstance(padding, str):
        if padding in {'SAME', 'VALID'}:
            padding = [padding] * (rhs.ndim - 2)  # type: ignore[list-item]
        else:
            raise ValueError(f"`padding` must be 'VALID' or 'SAME'. Passed: {padding}.")

    inferred_output_shape = tuple(map(_deconv_output_length, i_sdims, k_sdims, padding, output_padding, strides, dilation))

    if output_shape is None:
        output_shape = inferred_output_shape  # type: ignore[assignment]
    else:
        if not output_shape == inferred_output_shape:
            raise ValueError(f'`output_padding` and `output_shape` are not compatible.'
                             f'Inferred output shape from `output_padding`: {inferred_output_shape}, '
                             f'but got `output_shape` {output_shape}')

    pads = tuple(map(_compute_adjusted_padding, i_sdims, output_shape, k_sdims, strides, padding, dilation))

    if transpose_kernel:
        # flip spatial dims and swap input / output channel axes
        rhs = _flip_axes(rhs, np.array(dn.rhs_spec)[2:])
        rhs = np.swapaxes(rhs, dn.rhs_spec[0], dn.rhs_spec[1])
    return jax.lax.conv_general_dilated(lhs, rhs, one, pads, strides, dilation, dn, feature_group_count, precision=precision)


