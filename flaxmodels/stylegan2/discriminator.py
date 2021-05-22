import numpy as np
import jax
from jax import random
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Tuple, List, Callable
import h5py
from . import ops
from .. import utils


URLS = {'afhqcat': 'https://www.dropbox.com/s/qygbjkefyqyu9k9/stylegan2_discriminator_afhqcat.h5?dl=1',
        'afhqdog': 'https://www.dropbox.com/s/kmoxbp33qswz64p/stylegan2_discriminator_afhqdog.h5?dl=1',
        'afhqwild': 'https://www.dropbox.com/s/jz1hpsyt3isj6e7/stylegan2_discriminator_afhqwild.h5?dl=1',
        'brecahad': 'https://www.dropbox.com/s/h0cb89hruo6pmyj/stylegan2_discriminator_brecahad.h5?dl=1',
        'car': 'https://www.dropbox.com/s/2ghjrmxih7cic76/stylegan2_discriminator_car.h5?dl=1',
        'cat': 'https://www.dropbox.com/s/zfhjsvlsny5qixd/stylegan2_discriminator_cat.h5?dl=1',
        'church': 'https://www.dropbox.com/s/jlno7zeivkjtk8g/stylegan2_discriminator_church.h5?dl=1',
        'cifar10': 'https://www.dropbox.com/s/eldpubfkl4c6rur/stylegan2_discriminator_cifar10.h5?dl=1',
        'ffhq': 'https://www.dropbox.com/s/m42qy9951b7lq1s/stylegan2_discriminator_ffhq.h5?dl=1',
        'horse': 'https://www.dropbox.com/s/19f5pxrcdh2g8cw/stylegan2_discriminator_horse.h5?dl=1',
        'metfaces': 'https://www.dropbox.com/s/xnokaunql12glkd/stylegan2_discriminator_metfaces.h5?dl=1'}

RESOLUTION = {'metfaces': 1024,
              'ffhq': 1024,
              'church': 256,
              'cat': 256,
              'horse': 256,
              'car': 512,
              'brecahad': 512,
              'afhqwild': 512,
              'afhqdog': 512,
              'afhqcat': 512,
              'cifar10': 32}

C_DIM = {'metfaces': 0,
         'ffhq': 0,
         'church': 0,
         'cat': 0,
         'horse': 0,
         'car': 0,
         'brecahad': 0,
         'afhqwild': 0,
         'afhqdog': 0,
         'afhqcat': 0,
         'cifar10': 10}

ARCHITECTURE = {'metfaces': 'resnet',
                'ffhq': 'resnet',
                'church': 'resnet',
                'cat': 'resnet',
                'horse': 'resnet',
                'car': 'resnet',
                'brecahad': 'resnet',
                'afhqwild': 'resnet',
                'afhqdog': 'resnet',
                'afhqcat': 'resnet',
                'cifar10': 'orig'}

MBSTD_GROUP_SIZE = {'metfaces': None,
                    'ffhq': None,
                    'church': None,
                    'cat': None,
                    'horse': None,
                    'car': None,
                    'brecahad': None,
                    'afhqwild': None,
                    'afhqdog': None,
                    'afhqcat': None,
                    'cifar10': 32}


class LinearLayer(nn.Module):
    """
    Linear Layer.

    Attributes:
        in_features (int): Input dimension.
        out_features (int): Output dimension.
        use_bias (bool): If True, use bias.
        lr_multiplier (float): Learning rate multiplier.
        activation (str): Activation function: 'relu', 'lrelu', etc.
        param_dict (h5py.Group): Parameter dict with pretrained parameters.
        layer_name (str): Layer name.
        dtype (str): Data type.
        rng (jax.random.PRNGKey): Random seed.
    """
    in_features: int
    out_features: int
    use_bias: bool=True
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
        params = ops.get_weight(w_shape, self.lr_multiplier, self.use_bias, self.param_dict, self.layer_name, self.rng, self.dtype)

        if self.use_bias:
            w, b = params
        else:
            w = params
            b = None

        w = self.param(name='weight', init_fn=lambda *_ : w)
        if self.use_bias: b = self.param(name='bias', init_fn=lambda *_ : b)
        
        x = ops.linear_layer(x, w, b, self.activation) 
        return x


class FromRGBLayer(nn.Module):
    """
    From RGB Layer.

    Attributes:
        fmaps (int): Number of output channels of the convolution.
        kernel (int): Kernel size of the convolution.
        lr_multiplier (float): Learning rate multiplier.
        activation (str): Activation function: 'relu', 'lrelu', etc.
        param_dict (h5py.Group): Parameter dict with pretrained parameters. If not None, 'pretrained' will be ignored.
        dtype (str): Data dtype.
        rng (jax.random.PRNGKey): Random seed.
    """
    fmaps: int
    kernel: int=1
    lr_multiplier: float=1
    activation: str='leaky_relu'
    param_dict: h5py.Group=None
    dtype: str='float32'
    rng: Any=random.PRNGKey(0)

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
        w_shape = [self.kernel, self.kernel, x.shape[3], self.fmaps]
        w, b = ops.get_weight(w_shape, self.lr_multiplier, True, self.param_dict, 'fromrgb', self.rng, self.dtype)

        w = self.param(name='weight', init_fn=lambda *_ : w)
        b = self.param(name='bias', init_fn=lambda *_ : b)

        x = ops.conv2d(x, w)
        x += b
        x = ops.apply_activation(x, activation=self.activation)
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
        layer_name (str): Layer name.
        param_dict (h5py.Group): Parameter dict with pretrained parameters.
        lr_multiplier (float): Learning rate multiplier.
        dtype (str): Data dtype.
        rng (jax.random.PRNGKey): Random seed.       
    """
    fmaps: int
    kernel: int=3
    use_bias: bool=True
    down: bool=False
    resample_kernel: Tuple=None
    activation: str='leaky_relu'
    layer_name: str=None
    param_dict: h5py.Group=None
    lr_multiplier: float=1
    dtype: str='float32'
    rng: Any=random.PRNGKey(0)

    @nn.compact
    def __call__(self, x):
        """
        Run Discriminator Layer.

        Args:
            x (tensor): Input tensor of shape [N, H, W, C].

        Returns:
            (tensor): Output tensor of shape [N, H, W, fmaps].
        """
        w_shape = [self.kernel, self.kernel, x.shape[3], self.fmaps]
        if self.use_bias:
            w, b = ops.get_weight(w_shape, self.lr_multiplier, self.use_bias, self.param_dict, self.layer_name, self.rng, self.dtype)
        else:
            w = ops.get_weight(w_shape, self.lr_multiplier, self.use_bias, self.param_dict, self.layer_name, self.rng, self.dtype)

        w = self.param(name='weight', init_fn=lambda *_ : w)
        if self.use_bias: b = self.param(name='bias', init_fn=lambda *_ : b)

        x = ops.conv2d(x, w, down=self.down, resample_kernel=self.resample_kernel)
        if self.use_bias: x += b
        x = ops.apply_activation(x, activation=self.activation)
        return x


class DiscriminatorBlock(nn.Module):
    """
    Discriminator Block.

    Attributes:
        fmaps (int): Number of output channels of the convolution.
        kernel (int): Kernel size of the convolution.
        resample_kernel (Tuple): Kernel that is used for FIR filter.
        activation (str): Activation function: 'relu', 'lrelu', etc.
        param_dict (h5py.Group): Parameter dict with pretrained parameters.
        lr_multiplier (float): Learning rate multiplier.
        architecture (str): Architecture: 'orig', 'resnet'.
        nf (Callable): Callable that returns the number of feature maps for a given layer.
        dtype (str): Data dtype.
        rng (jax.random.PRNGKey): Random seed.       
    """
    res: int
    kernel: int=3
    resample_kernel: Tuple=(1, 3, 3, 1)
    activation: str='leaky_relu'
    param_dict: Any=None
    lr_multiplier: float=1
    architecture: str='resnet'
    nf: Callable=None
    dtype: str='float32'
    rng: Any=random.PRNGKey(0)

    @nn.compact
    def __call__(self, x):
        """
        Run Discriminator Block.

        Args:
            x (tensor): Input tensor of shape [N, H, W, C].

        Returns:
            (tensor): Output tensor of shape [N, H, W, fmaps].
        """
        residual = x
        
        for i in range(2):
            x = DiscriminatorLayer(fmaps=self.nf(self.res - (i + 1)),
                                   kernel=self.kernel,
                                   down=i == 1,
                                   resample_kernel=self.resample_kernel if i == 1 else None,
                                   activation=self.activation,
                                   layer_name=f'conv{i}',
                                   param_dict=self.param_dict,
                                   lr_multiplier=self.lr_multiplier,
                                   dtype=self.dtype,
                                   rng=self.rng)(x)

        
        if self.architecture == 'resnet':
            residual = DiscriminatorLayer(fmaps=self.nf(self.res - 2),
                                          kernel=1,
                                          use_bias=False,
                                          down=True,
                                          resample_kernel=self.resample_kernel,
                                          activation='linear',
                                          layer_name='skip',
                                          param_dict=self.param_dict,
                                          lr_multiplier=self.lr_multiplier,
                                          dtype=self.dtype,
                                          rng=self.rng)(residual)

            x = (x + residual) * (1 / np.sqrt(2))
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
        pretrained (str): Use pretrained model, None for random initialization.
        ckpt_dir (str): Directory to which the pretrained weights are downloaded. If None, a temp directory will be used.
        dtype (str): Data type.
        rng (jax.random.PRNGKey): Random seed.       
    """
    # Input dimensions.
    resolution: int=1024
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

    # Pretraining
    pretrained: str=None
    ckpt_dir: str=None

    dtype: str='float32'
    rng: Any=random.PRNGKey(0)

    def setup(self):
        self.resolution_ = self.resolution
        self.c_dim_ = self.c_dim
        self.architecture_ = self.architecture
        self.mbstd_group_size_ = self.mbstd_group_size
        if self.pretrained is not None:
            assert self.pretrained in URLS.keys(), f'Pretrained model not available: {self.pretrained}'
            ckpt_file = utils.download(self.ckpt_dir, URLS[self.pretrained])
            self.param_dict = h5py.File(ckpt_file, 'r')['discriminator']
            self.resolution_ = RESOLUTION[self.pretrained]
            self.architecture_ = ARCHITECTURE[self.pretrained]
            self.mbstd_group_size_ = MBSTD_GROUP_SIZE[self.pretrained]
            self.c_dim_ = C_DIM[self.pretrained]

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
        resolution_log2 = int(np.log2(self.resolution_))
        assert self.resolution_ == 2**resolution_log2 and self.resolution_ >= 4
        def nf(stage): return np.clip(int(self.fmap_base / (2.0 ** (stage * self.fmap_decay))), self.fmap_min, self.fmap_max)
        if self.mapping_fmaps is None:
            mapping_fmaps = nf(0)
        else:
            mapping_fmaps = self.mapping_fmaps
        
        # Label embedding and mapping.
        if self.c_dim_ > 0:
            c = LinearLayer(in_features=self.c_dim_,
                            out_features=mapping_fmaps,
                            lr_multiplier=self.mapping_lr_multiplier,
                            param_dict=self.param_dict,
                            layer_name='label_embedding',
                            dtype=self.dtype,
                            rng=self.rng)(c)
        
            c = ops.normalize_2nd_moment(c)
            for i in range(self.mapping_layers):
                c = LinearLayer(in_features=self.c_dim_,
                                out_features=mapping_fmaps,
                                lr_multiplier=self.mapping_lr_multiplier,
                                param_dict=self.param_dict,
                                layer_name=f'fc{i}',
                                dtype=self.dtype,
                                rng=self.rng)(c)

        # Layers for >=8x8 resolutions.
        y = None
        for res in range(resolution_log2, 2, -1):
            res_str = f'block_{2**res}x{2**res}'
            if res == resolution_log2:
                x = FromRGBLayer(fmaps=nf(res - 1),
                                 kernel=1,
                                 activation=self.activation,
                                 param_dict=self.param_dict[res_str] if self.param_dict is not None else None,
                                 dtype=self.dtype,
                                 rng=self.rng)(x, y)
 
            x = DiscriminatorBlock(res=res,
                                   kernel=3,
                                   resample_kernel=self.resample_kernel,
                                   activation=self.activation,
                                   param_dict=self.param_dict[res_str] if self.param_dict is not None else None,
                                   architecture=self.architecture_,
                                   nf=nf,
                                   dtype=self.dtype,
                                   rng=self.rng)(x)

        # Layers for 4x4 resolution.
        x = x.astype(self.dtype)
        if self.mbstd_num_features > 0:
            x = ops.minibatch_stddev_layer(x, self.mbstd_group_size_, self.mbstd_num_features)
        x = DiscriminatorLayer(fmaps=nf(1),
                               kernel=3,
                               use_bias=True,
                               activation=self.activation,
                               layer_name='conv0',
                               param_dict=self.param_dict['block_4x4'] if self.param_dict is not None else None,
                               dtype=self.dtype,
                               rng=self.rng)(x)

        # Switch to NCHW so that the pretrained weights still work after reshaping
        x = jnp.transpose(x, axes=(0, 3, 1, 2))
        x = jnp.reshape(x, newshape=(-1, x.shape[1] * x.shape[2] * x.shape[3]))

        x = LinearLayer(in_features=x.shape[1],
                        out_features=nf(0),
                        activation=self.activation,
                        param_dict=self.param_dict['block_4x4'] if self.param_dict is not None else None,
                        layer_name='fc0',
                        dtype=self.dtype,
                        rng=self.rng)(x)

        # Output layer.
        x = LinearLayer(in_features=x.shape[1],
                        out_features=1 if self.c_dim_ == 0 else mapping_fmaps,
                        param_dict=self.param_dict,
                        layer_name='output',
                        dtype=self.dtype,
                        rng=self.rng)(x)

        if self.c_dim_ > 0:
            x = jnp.sum(x * c, axis=1, keepdims=True) / jnp.sqrt(mapping_fmaps)
        return x



