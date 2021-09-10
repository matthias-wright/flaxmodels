import numpy as np
import jax
from jax import random
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Tuple, List
import h5py
from . import ops
from .. import utils


URLS = {'afhqcat': 'https://www.dropbox.com/s/lv1r0bwvg5ta51f/stylegan2_generator_afhqcat.h5?dl=1',
        'afhqdog': 'https://www.dropbox.com/s/px6ply9hv0vdwen/stylegan2_generator_afhqdog.h5?dl=1',
        'afhqwild': 'https://www.dropbox.com/s/p1slbtmzhcnw9q8/stylegan2_generator_afhqwild.h5?dl=1',
        'brecahad': 'https://www.dropbox.com/s/28uykhj0ku6hwg2/stylegan2_generator_brecahad.h5?dl=1',
        'car': 'https://www.dropbox.com/s/67o834b6xfg9x1q/stylegan2_generator_car.h5?dl=1',
        'cat': 'https://www.dropbox.com/s/cu9egc4e74e1nig/stylegan2_generator_cat.h5?dl=1',
        'church': 'https://www.dropbox.com/s/kwvokfwbrhsn58m/stylegan2_generator_church.h5?dl=1',
        'cifar10': 'https://www.dropbox.com/s/h1kmymjzfwwkftk/stylegan2_generator_cifar10.h5?dl=1',
        'ffhq': 'https://www.dropbox.com/s/e8de1peq7p8gq9d/stylegan2_generator_ffhq.h5?dl=1',
        'horse': 'https://www.dropbox.com/s/3e5bimv2d41bc13/stylegan2_generator_horse.h5?dl=1',
        'metfaces': 'https://www.dropbox.com/s/75klr5k6mgm7qdy/stylegan2_generator_metfaces.h5?dl=1'}

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

NUM_MAPPING_LAYERS = {'metfaces': 8,
                      'ffhq': 8,
                      'church': 8,
                      'cat': 8,
                      'horse': 8,
                      'car': 8,
                      'brecahad': 8,
                      'afhqwild': 8,
                      'afhqdog': 8,
                      'afhqcat': 8,
                      'cifar10': 2}


class MappingNetwork(nn.Module):
    """
    Mapping Network.

    Attributes:
        z_dim (int): Input latent (Z) dimensionality.
        c_dim (int): Conditioning label (C) dimensionality, 0 = no label.
        w_dim (int): Intermediate latent (W) dimensionality.
        embed_features (int): Label embedding dimensionality, None = same as w_dim.
        layer_features (int): Number of intermediate features in the mapping layers, None = same as w_dim.
        num_ws (int): Number of intermediate latents to output, None = do not broadcast.
        num_layers (int): Number of mapping layers.
        pretrained (str): Which pretrained model to use, None for random initialization.
        param_dict (h5py.Group): Parameter dict with pretrained parameters. If not None, 'pretrained' will be ignored.
        ckpt_dir (str): Directory to which the pretrained weights are downloaded. If None, a temp directory will be used.
        activation (str): Activation function: 'relu', 'lrelu', etc.
        lr_multiplier (float): Learning rate multiplier for the mapping layers.
        w_avg_beta (float): Decay for tracking the moving average of W during training, None = do not track.
        dtype (str): Data type.
    """
    # Dimensionality
    z_dim: int=512
    c_dim: int=0
    w_dim: int=512
    embed_features: int=None
    layer_features: int=512

    # Layers
    num_ws: int=18
    num_layers: int=8
    
    # Pretrained
    pretrained: str=None
    param_dict: h5py.Group=None
    ckpt_dir: str=None

    # Internal details
    activation: str='leaky_relu'
    lr_multiplier: float=0.01
    w_avg_beta: float=0.995
    dtype: str='float32'

    def setup(self):
        self.embed_features_ = self.embed_features
        self.c_dim_ = self.c_dim
        self.layer_features_ = self.layer_features
        self.num_layers_ = self.num_layers
        self.param_dict_ = self.param_dict

        if self.pretrained is not None and self.param_dict is None:
            assert self.pretrained in URLS.keys(), f'Pretrained model not available: {self.pretrained}'
            ckpt_file = utils.download(self.ckpt_dir, URLS[self.pretrained])
            self.param_dict_ = h5py.File(ckpt_file, 'r')['mapping_network']
            self.c_dim_ = C_DIM[self.pretrained]
            self.num_layers_ = NUM_MAPPING_LAYERS[self.pretrained]

        if self.embed_features_ is None:
            self.embed_features_ = self.w_dim
        if self.c_dim_ == 0:
            self.embed_features_ = 0
        if self.layer_features_ is None:
            self.layer_features_ = self.w_dim

        if self.param_dict_ is not None and 'w_avg' in self.param_dict_:
            self.w_avg = self.variable('moving_stats', 'w_avg', lambda *_ : jnp.array(self.param_dict_['w_avg']), [self.w_dim])
        else:
            self.w_avg = self.variable('moving_stats', 'w_avg', jnp.zeros, [self.w_dim])
       
    @nn.compact
    def __call__(self, z, c=None, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False, train=True):
        """
        Run Mapping Network.

        Args:
            z (tensor): Input noise, shape [N, z_dim].
            c (tensor): Input labels, shape [N, c_dim].
            truncation_psi (float): Controls truncation (trading off variation for quality). If 1, truncation is disabled.
            truncation_cutoff (int): Controls truncation. None = disable.
            skip_w_avg_update (bool): If True, updates the exponential moving average of W.
            train (bool): Training mode.

        Returns:
            (tensor): Intermediate latent W.
        """
        # Embed, normalize, and concat inputs.
        x = None
        if self.z_dim > 0:
            x = ops.normalize_2nd_moment(z.astype(jnp.float32))
        if self.c_dim_ > 0:
            # Conditioning label
            y = ops.LinearLayer(in_features=self.c_dim_,
                                out_features=self.embed_features_,
                                use_bias=True,
                                lr_multiplier=self.lr_multiplier,
                                activation='linear',
                                param_dict=self.param_dict_,
                                layer_name='label_embedding',
                                dtype=self.dtype)(c.astype(jnp.float32))

            y = ops.normalize_2nd_moment(y)
            x = jnp.concatenate((x, y), axis=1) if x is not None else y

        # Main layers.
        for i in range(self.num_layers_):
            x = ops.LinearLayer(in_features=x.shape[1],
                                out_features=self.layer_features_,
                                use_bias=True,
                                lr_multiplier=self.lr_multiplier,
                                activation=self.activation,
                                param_dict=self.param_dict_,
                                layer_name=f'fc{i}',
                                dtype=self.dtype)(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and train and not skip_w_avg_update:
            self.w_avg.value = self.w_avg_beta * self.w_avg.value + (1 - self.w_avg_beta) * jnp.mean(x, axis=0)

        # Broadcast.
        if self.num_ws is not None:
            x = jnp.repeat(jnp.expand_dims(x, axis=-2), repeats=self.num_ws, axis=-2)

        # Apply truncation.
        if truncation_psi != 1:
            assert self.w_avg_beta is not None
            if self.num_ws is None or truncation_cutoff is None:
                x = truncation_psi * x + (1 - truncation_psi) * self.w_avg.value
            else:
                x[:, :truncation_cutoff] = truncation_psi * x[:, :truncation_cutoff] + (1 - truncation_psi) * self.w_avg.value

        return x


class SynthesisLayer(nn.Module):
    """
    Synthesis Layer.

    Attributes:
        fmaps (int): Number of output channels of the modulated convolution.
        kernel (int): Kernel size of the modulated convolution.
        layer_idx (int): Layer index. Used to access the latent code for a specific layer.
        res (int): Resolution (log2) of the current layer.
        lr_multiplier (float): Learning rate multiplier.
        up (bool): If True, upsample the spatial resolution.
        activation (str): Activation function: 'relu', 'lrelu', etc.
        use_noise (bool): If True, add spatial-specific noise.
        resample_kernel (Tuple): Kernel that is used for FIR filter.
        fused_modconv (bool): If True, Perform modulation, convolution, and demodulation as a single fused operation.
        param_dict (h5py.Group): Parameter dict with pretrained parameters. If not None, 'pretrained' will be ignored.
        clip_conv (float): Clip the output of convolution layers to [-clip_conv, +clip_conv], None = disable clipping.
        dtype (str): Data dtype.
        rng (jax.random.PRNGKey): Random seed for noise const initialization.
    """
    fmaps: int
    kernel: int
    layer_idx: int
    res: int
    lr_multiplier: float=1
    up: bool=False
    activation: str='leaky_relu'
    use_noise: bool=True
    resample_kernel: Tuple=(1, 3, 3, 1)
    fused_modconv: bool=False
    param_dict: h5py.Group=None
    clip_conv: float=None
    dtype: str='float32'
    rng: Any=random.PRNGKey(0)

    def setup(self):
        if self.param_dict is not None:
            noise_const = jnp.array(self.param_dict['noise_const'], dtype=self.dtype)
        else:
            noise_const = random.normal(self.rng, shape=(1, 2 ** self.res, 2 ** self.res, 1), dtype=self.dtype)
        self.noise_const = self.variable('noise_consts', 'noise_const', lambda *_: noise_const)

    @nn.compact
    def __call__(self, x, dlatents, noise_mode='random', rng=random.PRNGKey(0)):
        """
        Run Synthesis Layer.

        Args:
            x (tensor): Input tensor of the shape [N, H, W, C].
            dlatents (tensor): Intermediate latents (W) of shape [N, num_ws, w_dim].
            noise_mode (str): Noise type.
                              - 'const': Constant noise.
                              - 'random': Random noise.
                              - 'none': No noise.
            rng (jax.random.PRNGKey): Random seed for spatialwise noise.

        Returns:
            (tensor): Output tensor of shape [N, H', W', fmaps].
        """
        assert noise_mode in ['const', 'random', 'none']

        # Affine transformation to obtain style variable.
        s = ops.LinearLayer(in_features=dlatents[:, self.layer_idx].shape[1],
                            out_features=x.shape[3],
                            use_bias=True,
                            bias_init=1,
                            lr_multiplier=self.lr_multiplier,
                            param_dict=self.param_dict,
                            layer_name='affine',
                            dtype=self.dtype)(dlatents[:, self.layer_idx])

        # Noise variables.
        if self.param_dict is None:
            noise_strength = jnp.zeros(())
        else:
            noise_strength = jnp.array(self.param_dict['noise_strength']) 
        noise_strength = self.param(name='noise_strength', init_fn=lambda *_ : noise_strength)

        # Weight and bias for convolution operation.
        w_init = ops.get_weight_init(self.param_dict, layer_name='conv')
        b_init = ops.get_bias_init(self.param_dict, layer_name='conv')
        w = self.param('weight', w_init, (self.kernel, self.kernel, x.shape[3], self.fmaps))
        b = self.param('bias', b_init, (self.fmaps,))
        w = ops.equalize_lr_weight(w, self.lr_multiplier)
        b = ops.equalize_lr_bias(b, self.lr_multiplier)

        x = ops.modulated_conv2d_layer(x=x, 
                                       w=w, 
                                       s=s, 
                                       fmaps=self.fmaps, 
                                       kernel=self.kernel, 
                                       up=self.up, 
                                       resample_kernel=self.resample_kernel, 
                                       fused_modconv=self.fused_modconv)
        
        if self.use_noise and noise_mode != 'none':
            if noise_mode == 'const':
                noise = self.noise_const.value
            elif noise_mode == 'random':
                noise = random.normal(rng, shape=(x.shape[0], x.shape[1], x.shape[2], 1), dtype=self.dtype)
            x += noise * noise_strength.astype(self.dtype)
        x += b.astype(x.dtype)
        x = ops.apply_activation(x, activation=self.activation)
        if self.clip_conv is not None:
            x = jnp.clip(x, -self.clip_conv, self.clip_conv)
        return x


class ToRGBLayer(nn.Module):
    """
    To RGB Layer.

    Attributes:
        fmaps (int): Number of output channels of the modulated convolution.
        layer_idx (int): Layer index. Used to access the latent code for a specific layer.
        kernel (int): Kernel size of the modulated convolution.
        lr_multiplier (float): Learning rate multiplier.
        fused_modconv (bool): If True, Perform modulation, convolution, and demodulation as a single fused operation.
        param_dict (h5py.Group): Parameter dict with pretrained parameters. If not None, 'pretrained' will be ignored.
        clip_conv (float): Clip the output of convolution layers to [-clip_conv, +clip_conv], None = disable clipping.
        dtype (str): Data dtype.
    """
    fmaps: int
    layer_idx: int
    kernel: int=1
    lr_multiplier: float=1
    fused_modconv: bool=False
    param_dict: h5py.Group=None
    clip_conv: float=None
    dtype: str='float32'
    
    @nn.compact
    def __call__(self, x, y, dlatents):
        """
        Run To RGB Layer.

        Args:
            x (tensor): Input tensor of shape [N, H, W, C].
            y (tensor): Image of shape [N, H', W', fmaps]. 
            dlatents (tensor): Intermediate latents (W) of shape [N, num_ws, w_dim].

        Returns:
            (tensor): Output tensor of shape [N, H', W', fmaps].
        """
        # Affine transformation to obtain style variable.
        s = ops.LinearLayer(in_features=dlatents[:, self.layer_idx].shape[1],
                            out_features=x.shape[3],
                            use_bias=True,
                            bias_init=1,
                            lr_multiplier=self.lr_multiplier,
                            param_dict=self.param_dict,
                            layer_name='affine',
                            dtype=self.dtype)(dlatents[:, self.layer_idx])

        # Weight and bias for convolution operation.
        w_init = ops.get_weight_init(self.param_dict, layer_name='conv')
        b_init = ops.get_bias_init(self.param_dict, layer_name='conv')
        w = self.param('weight', w_init, (self.kernel, self.kernel, x.shape[3], self.fmaps))
        b = self.param('bias', b_init, (self.fmaps,))
        w = ops.equalize_lr_weight(w, self.lr_multiplier)
        b = ops.equalize_lr_bias(b, self.lr_multiplier)
        
        x = ops.modulated_conv2d_layer(x, w, s, fmaps=self.fmaps, kernel=self.kernel, demodulate=False, fused_modconv=self.fused_modconv)
        x += b.astype(x.dtype)
        x = ops.apply_activation(x, activation='linear')
        if self.clip_conv is not None:
            x = jnp.clip(x, -self.clip_conv, self.clip_conv)
        if y is not None:
            x += y.astype(jnp.float32)
        return x


class SynthesisBlock(nn.Module):
    """
    Synthesis Block.

    Attributes:
        fmaps (int): Number of output channels of the modulated convolution.
        res (int): Resolution (log2) of the current block.
        num_layers (int): Number of layers in the current block.
        num_channels (int): Number of output color channels.
        lr_multiplier (float): Learning rate multiplier.
        activation (str): Activation function: 'relu', 'lrelu', etc.
        use_noise (bool): If True, add spatial-specific noise.
        resample_kernel (Tuple): Kernel that is used for FIR filter.
        fused_modconv (bool): If True, Perform modulation, convolution, and demodulation as a single fused operation.
        param_dict (h5py.Group): Parameter dict with pretrained parameters. If not None, 'pretrained' will be ignored.
        clip_conv (float): Clip the output of convolution layers to [-clip_conv, +clip_conv], None = disable clipping.
        dtype (str): Data dtype.
        rng (jax.random.PRNGKey): Random seed for noise const initialization.
    """
    fmaps: int
    res: int
    num_layers: int=2
    num_channels: int=3
    lr_multiplier: float=1
    activation: str='leaky_relu'
    use_noise: bool=True
    resample_kernel: Tuple=(1, 3, 3, 1)
    fused_modconv: bool=False
    param_dict: h5py.Group=None
    clip_conv: float=None
    dtype: str='float32'
    rng: Any=random.PRNGKey(0)

    @nn.compact
    def __call__(self, x, y, dlatents_in, noise_mode='random', rng=random.PRNGKey(0)):
        """
        Run Synthesis Block.

        Args:
            x (tensor): Input tensor of shape [N, H, W, C].
            y (tensor): Image of shape [N, H', W', fmaps]. 
            dlatents (tensor): Intermediate latents (W) of shape [N, num_ws, w_dim].
            noise_mode (str): Noise type.
                              - 'const': Constant noise.
                              - 'random': Random noise.
                              - 'none': No noise.
            rng (jax.random.PRNGKey): Random seed for spatialwise noise.

        Returns:
            (tensor): Output tensor of shape [N, H', W', fmaps].
        """
        x = x.astype(self.dtype)
        for i in range(self.num_layers):
            x = SynthesisLayer(fmaps=self.fmaps, 
                               kernel=3,
                               layer_idx=self.res * 2 - (5 - i) if self.res > 2 else 0,
                               res=self.res,
                               lr_multiplier=self.lr_multiplier,
                               up=i == 0 and self.res != 2,
                               activation=self.activation,
                               use_noise=self.use_noise,
                               resample_kernel=self.resample_kernel,
                               fused_modconv=self.fused_modconv,
                               param_dict=self.param_dict[f'layer{i}'] if self.param_dict is not None else None,
                               dtype=self.dtype,
                               rng=self.rng)(x, dlatents_in, noise_mode, rng)

        if self.num_layers == 2:
            k = ops.setup_filter(self.resample_kernel)
            y = ops.upsample2d(y, f=k, up=2)
        
        y = ToRGBLayer(fmaps=self.num_channels, 
                       layer_idx=self.res * 2 - 3, 
                       lr_multiplier=self.lr_multiplier,
                       param_dict=self.param_dict['torgb'] if self.param_dict is not None else None, 
                       dtype=self.dtype)(x, y, dlatents_in)
        return x, y


class SynthesisNetwork(nn.Module):
    """
    Synthesis Network.

    Attributes:
        resolution (int): Output resolution.
        num_channels (int): Number of output color channels.
        w_dim (int): Input latent (Z) dimensionality.
        fmap_base (int): Overall multiplier for the number of feature maps.
        fmap_decay (int): Log2 feature map reduction when doubling the resolution.
        fmap_min (int): Minimum number of feature maps in any layer.
        fmap_max (int): Maximum number of feature maps in any layer.
        fmap_const (int): Number of feature maps in the constant input layer. None = default.
        pretrained (str): Which pretrained model to use, None for random initialization.
        param_dict (h5py.Group): Parameter dict with pretrained parameters. If not None, 'pretrained' will be ignored.
        ckpt_dir (str): Directory to which the pretrained weights are downloaded. If None, a temp directory will be used.
        activation (str): Activation function: 'relu', 'lrelu', etc.
        use_noise (bool): If True, add spatial-specific noise.
        resample_kernel (Tuple): Kernel that is used for FIR filter.
        fused_modconv (bool): If True, Perform modulation, convolution, and demodulation as a single fused operation.
        num_fp16_res (int): Use float16 for the 'num_fp16_res' highest resolutions.
        clip_conv (float): Clip the output of convolution layers to [-clip_conv, +clip_conv], None = disable clipping.
        dtype (str): Data type.
        rng (jax.random.PRNGKey): Random seed for noise const initialization.
    """
    # Dimensionality
    resolution: int=1024
    num_channels: int=3
    w_dim: int=512

    # Capacity
    fmap_base: int=16384
    fmap_decay: int=1
    fmap_min: int=1
    fmap_max: int=512
    fmap_const: int=None

    # Pretraining
    pretrained: str=None
    param_dict: h5py.Group=None
    ckpt_dir: str=None

    # Internal details
    activation: str='leaky_relu'
    use_noise: bool=True
    resample_kernel: Tuple=(1, 3, 3, 1)
    fused_modconv: bool=False
    num_fp16_res: int=0
    clip_conv: float=None
    dtype: str='float32'
    rng: Any=random.PRNGKey(0)

    def setup(self):
        self.resolution_ = self.resolution
        self.param_dict_ = self.param_dict
        if self.pretrained is not None and self.param_dict is None:
            assert self.pretrained in URLS.keys(), f'Pretrained model not available: {self.pretrained}'
            ckpt_file = utils.download(self.ckpt_dir, URLS[self.pretrained])
            self.param_dict_ = h5py.File(ckpt_file, 'r')['synthesis_network']
            self.resolution_ = RESOLUTION[self.pretrained]
        
    @nn.compact
    def __call__(self, dlatents_in, noise_mode='random', rng=random.PRNGKey(0)):
        """
        Run Synthesis Network.

        Args:
            dlatents_in (tensor): Intermediate latents (W) of shape [N, num_ws, w_dim].
            noise_mode (str): Noise type.
                              - 'const': Constant noise.
                              - 'random': Random noise.
                              - 'none': No noise.
            rng (jax.random.PRNGKey): Random seed for spatialwise noise.

        Returns:
            (tensor): Image of shape [N, H, W, num_channels].
        """
        resolution_log2 = int(np.log2(self.resolution_))
        assert self.resolution_ == 2 ** resolution_log2 and self.resolution_ >= 4

        def nf(stage): return np.clip(int(self.fmap_base / (2.0 ** (stage * self.fmap_decay))), self.fmap_min, self.fmap_max)
        num_layers = resolution_log2 * 2 - 2
        
        fmaps = self.fmap_const if self.fmap_const is not None else nf(1)
        
        if self.param_dict_ is None:
            def const_init(key, shape, dtype=self.dtype):
                return random.normal(key, shape, dtype=dtype)
        else:
            const_init = lambda *_ : jnp.array(self.param_dict_['const'], dtype=self.dtype)
        x = self.param('const', const_init, (1, 4, 4, fmaps))
        x = jnp.repeat(x, repeats=dlatents_in.shape[0], axis=0)

        y = None

        dlatents_in = dlatents_in.astype(jnp.float32)

        for res in range(2, resolution_log2 + 1):
            x, y = SynthesisBlock(fmaps=nf(res - 1),
                                  res=res,
                                  num_layers=1 if res == 2 else 2,
                                  num_channels=self.num_channels,
                                  activation=self.activation,
                                  use_noise=self.use_noise,
                                  resample_kernel=self.resample_kernel,
                                  fused_modconv=self.fused_modconv,
                                  param_dict=self.param_dict_[f'block_{2 ** res}x{2 ** res}'] if self.param_dict_ is not None else None,
                                  clip_conv=self.clip_conv,
                                  dtype=self.dtype if res > resolution_log2 - self.num_fp16_res else 'float32',
                                  rng=self.rng)(x, y, dlatents_in, noise_mode, rng)

        return y


class Generator(nn.Module):
    """
    Generator.

    Attributes:
        resolution (int): Output resolution.
        num_channels (int): Number of output color channels.
        z_dim (int): Input latent (Z) dimensionality.
        c_dim (int): Conditioning label (C) dimensionality, 0 = no label.
        w_dim (int): Intermediate latent (W) dimensionality.
        mapping_layer_features (int): Number of intermediate features in the mapping layers, None = same as w_dim.
        mapping_embed_features (int): Label embedding dimensionality, None = same as w_dim.
        num_ws (int): Number of intermediate latents to output, None = do not broadcast.
        num_mapping_layers (int): Number of mapping layers.
        fmap_base (int): Overall multiplier for the number of feature maps.
        fmap_decay (int): Log2 feature map reduction when doubling the resolution.
        fmap_min (int): Minimum number of feature maps in any layer.
        fmap_max (int): Maximum number of feature maps in any layer.
        fmap_const (int): Number of feature maps in the constant input layer. None = default.
        pretrained (str): Which pretrained model to use, None for random initialization.
        ckpt_dir (str): Directory to which the pretrained weights are downloaded. If None, a temp directory will be used.
        use_noise (bool): If True, add spatial-specific noise.
        activation (str): Activation function: 'relu', 'lrelu', etc.
        w_avg_beta (float): Decay for tracking the moving average of W during training, None = do not track.
        mapping_lr_multiplier (float): Learning rate multiplier for the mapping network.
        resample_kernel (Tuple): Kernel that is used for FIR filter.
        fused_modconv (bool): If True, Perform modulation, convolution, and demodulation as a single fused operation.
        num_fp16_res (int): Use float16 for the 'num_fp16_res' highest resolutions.
        clip_conv (float): Clip the output of convolution layers to [-clip_conv, +clip_conv], None = disable clipping.
        dtype (str): Data type.
        rng (jax.random.PRNGKey): Random seed for noise const initialization.
    """
    # Dimensionality
    resolution: int=1024
    num_channels: int=3
    z_dim: int=512
    c_dim: int=0
    w_dim: int=512
    mapping_layer_features: int=512
    mapping_embed_features: int=None

    # Layers
    num_ws: int=18
    num_mapping_layers: int=8

    # Capacity
    fmap_base: int=16384
    fmap_decay: int=1
    fmap_min: int=1
    fmap_max: int=512
    fmap_const: int=None

    # Pretraining
    pretrained: str=None
    ckpt_dir: str=None

    # Internal details
    use_noise: bool=True
    activation: str='leaky_relu'
    w_avg_beta: float=0.995
    mapping_lr_multiplier: float=0.01
    resample_kernel: Tuple=(1, 3, 3, 1)
    fused_modconv: bool=False
    num_fp16_res: int=0
    clip_conv: float=None
    dtype: str='float32'
    rng: Any=random.PRNGKey(0)

    def setup(self):
        self.resolution_ = self.resolution
        self.c_dim_ = self.c_dim
        self.num_mapping_layers_ = self.num_mapping_layers
        if self.pretrained is not None:
            assert self.pretrained in URLS.keys(), f'Pretrained model not available: {self.pretrained}'
            ckpt_file = utils.download(self.ckpt_dir, URLS[self.pretrained])
            self.param_dict = h5py.File(ckpt_file, 'r')
            self.resolution_ = RESOLUTION[self.pretrained]
            self.c_dim_ = C_DIM[self.pretrained]
            self.num_mapping_layers_ = NUM_MAPPING_LAYERS[self.pretrained]
        else:
            self.param_dict = None
       
    @nn.compact
    def __call__(self, z, c=None, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False, train=True, noise_mode='random', rng=random.PRNGKey(0)):
        """
        Run Generator.

        Args:
            z (tensor): Input noise, shape [N, z_dim].
            c (tensor): Input labels, shape [N, c_dim].
            truncation_psi (float): Controls truncation (trading off variation for quality). If 1, truncation is disabled.
            truncation_cutoff (int): Controls truncation. None = disable.
            skip_w_avg_update (bool): If True, updates the exponential moving average of W.
            train (bool): Training mode.
            noise_mode (str): Noise type.
                              - 'const': Constant noise.
                              - 'random': Random noise.
                              - 'none': No noise.
            rng (jax.random.PRNGKey): Random seed for spatialwise noise.

        Returns:
            (tensor): Image of shape [N, H, W, num_channels].
        """
        dlatents_in = MappingNetwork(z_dim=self.z_dim,
                                     c_dim=self.c_dim_,
                                     w_dim=self.w_dim,
                                     num_ws=self.num_ws,
                                     num_layers=self.num_mapping_layers_,
                                     embed_features=self.mapping_embed_features,
                                     layer_features=self.mapping_layer_features,
                                     activation=self.activation,
                                     lr_multiplier=self.mapping_lr_multiplier,
                                     w_avg_beta=self.w_avg_beta,
                                     param_dict=self.param_dict['mapping_network'] if self.param_dict is not None else None,
                                     dtype=self.dtype,
                                     name='mapping_network')(z, c, truncation_psi, truncation_cutoff, skip_w_avg_update, train)

        x = SynthesisNetwork(resolution=self.resolution_,
                             num_channels=self.num_channels,
                             w_dim=self.w_dim,
                             fmap_base=self.fmap_base,
                             fmap_decay=self.fmap_decay,
                             fmap_min=self.fmap_min,
                             fmap_max=self.fmap_max,
                             fmap_const=self.fmap_const,
                             param_dict=self.param_dict['synthesis_network'] if self.param_dict is not None else None,
                             activation=self.activation,
                             use_noise=self.use_noise,
                             resample_kernel=self.resample_kernel,
                             fused_modconv=self.fused_modconv,
                             num_fp16_res=self.num_fp16_res,
                             clip_conv=self.clip_conv,
                             dtype=self.dtype,
                             rng=self.rng,
                             name='synthesis_network')(dlatents_in, noise_mode, rng)

        return x

