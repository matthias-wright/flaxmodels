import numpy as np
import jax
from jax import random
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Tuple, List, Callable
import h5py
from . import ops
from .. import utils


class FromRGBLayer(nn.Module):
    """
    From RGB Layer.

    Attributes:
        fmaps (int): Number of output channels of the convolution.
        kernel (int): Kernel size of the convolution.
        lr_multiplier (float): Learning rate multiplier.
        activation (str): Activation function: 'relu', 'lrelu', etc.
        clip_conv (float): Clip the output of convolution layers to [-clip_conv, +clip_conv], None = disable clipping.
        dtype (str): Data dtype.
    """
    fmaps: int
    kernel: int=1
    lr_multiplier: float=1
    activation: str='leaky_relu'
    clip_conv: float=None
    dtype: str='float32'

    @nn.compact
    def __call__(self, x, y):
        """
        Run From RGB Layer.

        Args:
            x (tensor): Input image of shape [N, H, W, num_channels].
            y (tensor): Input tensor of shape [N, H, W, out_channels].

        Returns:
            (tensor): Output tensor of shape [N, H, W, out_channels].
        """
        w_init = ops.get_weight_init()
        b_init = ops.get_bias_init()
        w = self.param('weight', w_init, (self.kernel, self.kernel, x.shape[3], self.fmaps))
        b = self.param('bias', b_init, (self.fmaps,))
        w = ops.equalize_lr_weight(w, self.lr_multiplier)
        b = ops.equalize_lr_bias(b, self.lr_multiplier)
        
        x = x.astype(self.dtype)
        x = ops.conv2d(x, w.astype(x.dtype))
        x += b.astype(x.dtype)
        x = ops.apply_activation(x, activation=self.activation)
        if self.clip_conv is not None:
            x = jnp.clip(x, -self.clip_conv, self.clip_conv)
        if y is not None:
            x += y
        return x


class DiscriminatorLayer(nn.Module):
    """
    Discriminator Layer.

    Attributes:
        fmaps (int): Number of output channels of the convolution.
        kernel (int): Kernel size of the convolution.
        use_bias (bool): If True, use bias.
        down (bool): If True, downsample the spatial resolution.
        resample_kernel (Tuple): Kernel that is used for FIR filter.
        activation (str): Activation function: 'relu', 'lrelu', etc.
        lr_multiplier (float): Learning rate multiplier.
        clip_conv (float): Clip the output of convolution layers to [-clip_conv, +clip_conv], None = disable clipping.
        dtype (str): Data dtype.
    """
    fmaps: int
    kernel: int=3
    use_bias: bool=True
    down: bool=False
    resample_kernel: Tuple=None
    activation: str='leaky_relu'
    lr_multiplier: float=1
    clip_conv: float=None
    dtype: str='float32'

    @nn.compact
    def __call__(self, x):
        """
        Run Discriminator Layer.

        Args:
            x (tensor): Input tensor of shape [N, H, W, C].

        Returns:
            (tensor): Output tensor of shape [N, H, W, fmaps].
        """
        w_init = ops.get_weight_init()
        w = self.param('weight', w_init, (self.kernel, self.kernel, x.shape[3], self.fmaps))
        w = ops.equalize_lr_weight(w, self.lr_multiplier)
        if self.use_bias:
            b_init = ops.get_bias_init()
            b = self.param('bias', b_init, (self.fmaps,))
            b = ops.equalize_lr_bias(b, self.lr_multiplier)

        x = x.astype(self.dtype)
        x = ops.conv2d(x, w, down=self.down, resample_kernel=self.resample_kernel)
        if self.use_bias: x += b.astype(x.dtype)
        x = ops.apply_activation(x, activation=self.activation)
        if self.clip_conv is not None:
            x = jnp.clip(x, -self.clip_conv, self.clip_conv)
        return x


class DiscriminatorBlock(nn.Module):
    """
    Discriminator Block.

    Attributes:
        fmaps (int): Number of output channels of the convolution.
        kernel (int): Kernel size of the convolution.
        resample_kernel (Tuple): Kernel that is used for FIR filter.
        activation (str): Activation function: 'relu', 'lrelu', etc.
        lr_multiplier (float): Learning rate multiplier.
        architecture (str): Architecture: 'orig', 'resnet'.
        nf (Callable): Callable that returns the number of feature maps for a given layer.
        clip_conv (float): Clip the output of convolution layers to [-clip_conv, +clip_conv], None = disable clipping.
        dtype (str): Data dtype.
    """
    res: int
    kernel: int=3
    resample_kernel: Tuple=(1, 3, 3, 1)
    activation: str='leaky_relu'
    lr_multiplier: float=1
    architecture: str='resnet'
    nf: Callable=None
    clip_conv: float=None
    dtype: str='float32'

    @nn.compact
    def __call__(self, x, feature_list=None):
        """
        Run Discriminator Block.

        Args:
            x (tensor): Input tensor of shape [N, H, W, C].
            feature_list (list): List of activations.

        Returns:
            (tensor): Output tensor of shape [N, H, W, fmaps]
        """
        x = x.astype(self.dtype)
        residual = x
        for i in range(2):
            x = DiscriminatorLayer(fmaps=self.nf(self.res - (i + 1)),
                                   kernel=self.kernel,
                                   down=i == 1,
                                   resample_kernel=self.resample_kernel if i == 1 else None,
                                   activation=self.activation,
                                   lr_multiplier=self.lr_multiplier,
                                   clip_conv=self.clip_conv,
                                   dtype=self.dtype)(x)
            if x.shape[3] == 512 and (x.shape[1] == 16 or x.shape[1] == 32):
                feature_list.append(nn.Conv(features=1, kernel_size=(3, 3), name=f'Out_{i}')(x))

        if self.architecture == 'resnet':
            residual = DiscriminatorLayer(fmaps=self.nf(self.res - 2),
                                          kernel=1,
                                          use_bias=False,
                                          down=True,
                                          resample_kernel=self.resample_kernel,
                                          activation='linear',
                                          lr_multiplier=self.lr_multiplier,
                                          dtype=self.dtype)(residual)

            x = (x + residual) * np.sqrt(0.5, dtype=x.dtype)
        return x


class Discriminator(nn.Module):
    """
    Discriminator.

    Attributes:
        resolution (int): Input resolution. Overridden based on dataset.
        num_channels (int): Number of input color channels. Overridden based on dataset.
        c_dim (int): Dimensionality of the labels (c), 0 if no labels. Overrttten based on dataset.
        fmap_base (int): Overall multiplier for the number of feature maps.
        fmap_decay (int): Log2 feature map reduction when doubling the resolution.
        fmap_min (int): Minimum number of feature maps in any layer.
        fmap_max (int): Maximum number of feature maps in any layer.
        mapping_layers (int): Number of additional mapping layers for the conditioning labels.
        mapping_fmaps (int): Number of activations in the mapping layers, None = default.
        mapping_lr_multiplier (float): Learning rate multiplier for the mapping layers.
        architecture (str): Architecture: 'orig', 'resnet'.
        activation (int): Activation function: 'relu', 'leaky_relu', etc.
        mbstd_group_size (int): Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_features (int): Number of features for the minibatch standard deviation layer, 0 = disable.
        resample_kernel (Tuple): Low-pass filter to apply when resampling activations, None = box filter.
        num_fp16_res (int): Use float16 for the 'num_fp16_res' highest resolutions.
        clip_conv (float): Clip the output of convolution layers to [-clip_conv, +clip_conv], None = disable clipping.
        dtype (str): Data type.
    """
    # Input dimensions.
    resolution: int=256
    num_channels: int=3
    c_dim: int=0

    # Capacity.
    fmap_base: int=16384
    fmap_decay: int=1
    fmap_min: int=1
    fmap_max: int=512

    # Internal details.
    mapping_layers: int=0
    mapping_fmaps: int=None
    mapping_lr_multiplier: float=0.1
    architecture: str='resnet'
    activation: str='leaky_relu'
    mbstd_group_size: int=None
    mbstd_num_features: int=1
    resample_kernel: Tuple=(1, 3, 3, 1)
    num_fp16_res: int=0
    clip_conv: float=None

    dtype: str='float32'

    def setup(self):
        assert self.architecture in ['orig', 'resnet']

    @nn.compact
    def __call__(self, x, c=None):
        """
        Run Discriminator.

        Args:
            x (tensor): Input image of shape [N, H, W, num_channels].
            c (tensor): Input labels, shape [N, c_dim].
        
        Returns:
            (tensor): Output tensor of shape [N, 1].
        """
        resolution_log2 = int(np.log2(self.resolution))
        assert self.resolution == 2**resolution_log2 and self.resolution >= 4
        def nf(stage): return np.clip(int(self.fmap_base / (2.0 ** (stage * self.fmap_decay))), self.fmap_min, self.fmap_max)
        if self.mapping_fmaps is None:
            mapping_fmaps = nf(0)
        else:
            mapping_fmaps = self.mapping_fmaps
        
        # Label embedding and mapping.
        if self.c_dim > 0:
            c = ops.LinearLayer(in_features=self.c_dim,
                                out_features=mapping_fmaps,
                                lr_multiplier=self.mapping_lr_multiplier,
                                dtype=self.dtype)(c)
        
            c = ops.normalize_2nd_moment(c)
            for i in range(self.mapping_layers):
                c = ops.LinearLayer(in_features=self.c_dim,
                                    out_features=mapping_fmaps,
                                    lr_multiplier=self.mapping_lr_multiplier,
                                    dtype=self.dtype)(c)

        # Layers for >=8x8 resolutions.
        y = None
        feature_list = []
        for res in range(resolution_log2, 2, -1):
            res_str = f'block_{2**res}x{2**res}'
            if res == resolution_log2:
                x = FromRGBLayer(fmaps=nf(res - 1),
                                 kernel=1,
                                 activation=self.activation,
                                 clip_conv=self.clip_conv,
                                 dtype=self.dtype if res >= resolution_log2 + 1 - self.num_fp16_res else 'float32')(x, y)
 
            x = DiscriminatorBlock(res=res,
                                   kernel=3,
                                   resample_kernel=self.resample_kernel,
                                   activation=self.activation,
                                   architecture=self.architecture,
                                   nf=nf,
                                   clip_conv=self.clip_conv,
                                   dtype=self.dtype if res >= resolution_log2 + 1 - self.num_fp16_res else 'float32')(x, feature_list)


        # Layers for 4x4 resolution.
        dtype = jnp.float32
        x = x.astype(dtype)
        if self.mbstd_num_features > 0:
            x = ops.minibatch_stddev_layer(x, self.mbstd_group_size, self.mbstd_num_features)
        x = DiscriminatorLayer(fmaps=nf(1),
                               kernel=3,
                               use_bias=True,
                               activation=self.activation,
                               clip_conv=self.clip_conv,
                               dtype=dtype)(x)

        # Switch to NCHW so that the pretrained weights still work after reshaping
        x = jnp.transpose(x, axes=(0, 3, 1, 2))
        x = jnp.reshape(x, newshape=(-1, x.shape[1] * x.shape[2] * x.shape[3]))

        x = ops.LinearLayer(in_features=x.shape[1],
                            out_features=nf(0),
                            activation=self.activation,
                            dtype=dtype)(x)

        # Output layer.
        x = ops.LinearLayer(in_features=x.shape[1],
                            out_features=1 if self.c_dim == 0 else mapping_fmaps,
                            dtype=dtype)(x)

        if self.c_dim > 0:
            x = jnp.sum(x * c, axis=1, keepdims=True) / jnp.sqrt(mapping_fmaps)
        feature_list.append(x)
        return feature_list



