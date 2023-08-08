import numpy as np
import jax
from jax import random
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Tuple, List
import h5py
from dataclasses import  field
from . import ops
from .. import utils


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
    
    # Internal details
    activation: str='leaky_relu'
    lr_multiplier: float=0.01
    w_avg_beta: float=0.995
    dtype: str='float32'

    def setup(self):
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
        if self.c_dim > 0:
            # Conditioning label
            y = ops.LinearLayer(in_features=self.c_dim,
                                out_features=self.embed_features,
                                use_bias=True,
                                lr_multiplier=self.lr_multiplier,
                                activation='linear',
                                dtype=self.dtype)(c.astype(jnp.float32))

            y = ops.normalize_2nd_moment(y)
            x = jnp.concatenate((x, y), axis=1) if x is not None else y

        # Main layers.
        for i in range(self.num_layers):
            x = ops.LinearLayer(in_features=x.shape[1],
                                out_features=self.layer_features,
                                use_bias=True,
                                lr_multiplier=self.lr_multiplier,
                                activation=self.activation,
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
        clip_conv (float): Clip the output of convolution layers to [-clip_conv, +clip_conv], None = disable clipping.
        dtype (str): Data dtype.
        rng (jax.random.PRNGKey): Random PRNG for noise const initialization.
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
    clip_conv: float=None
    dtype: str='float32'
    rng: Any=field(default_factory=lambda : random.PRNGKey(0))

    def setup(self):
        noise_const = random.normal(self.rng, shape=(1, 2 ** self.res, 2 ** self.res, 1), dtype=self.dtype)
        self.noise_const = self.variable('noise_consts', 'noise_const', lambda *_: noise_const)

    @nn.compact
    def __call__(self, x, dlatents, noise_mode='random', rng=None):
        """
        Run Synthesis Layer.

        Args:
            x (tensor): Input tensor of the shape [N, H, W, C].
            dlatents (tensor): Intermediate latents (W) of shape [N, num_ws, w_dim].
            noise_mode (str): Noise type.
                              - 'const': Constant noise.
                              - 'random': Random noise.
                              - 'none': No noise.
            rng (jax.random.PRNGKey): Random PRNG for spatialwise noise.

        Returns:
            (tensor): Output tensor of shape [N, H', W', fmaps].
        """
        assert noise_mode in ['const', 'random', 'none']
        if rng is None:
            rng = random.PRNGKey(0)
    
        # Affine transformation to obtain style variable.
        s = ops.LinearLayer(in_features=dlatents[:, self.layer_idx].shape[1],
                            out_features=x.shape[3],
                            use_bias=True,
                            bias_init=1,
                            lr_multiplier=self.lr_multiplier,
                            dtype=self.dtype)(dlatents[:, self.layer_idx])

        # Noise variables.
        noise_strength = self.param(name='noise_strength', init_fn=lambda *_ : jnp.zeros(()))

        # Weight and bias for convolution operation.
        w_init = ops.get_weight_init()
        b_init = ops.get_bias_init()
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
        clip_conv (float): Clip the output of convolution layers to [-clip_conv, +clip_conv], None = disable clipping.
        dtype (str): Data dtype.
    """
    fmaps: int
    layer_idx: int
    kernel: int=1
    lr_multiplier: float=1
    fused_modconv: bool=False
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
                            dtype=self.dtype)(dlatents[:, self.layer_idx])

        # Weight and bias for convolution operation.
        w_init = ops.get_weight_init()
        b_init = ops.get_bias_init()
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
        clip_conv (float): Clip the output of convolution layers to [-clip_conv, +clip_conv], None = disable clipping.
        dtype (str): Data dtype.
        rng (jax.random.PRNGKey): Random PRNG for noise const initialization.
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
    clip_conv: float=None
    dtype: str='float32'
    rng: Any=field(default_factory=lambda : random.PRNGKey(0))

    @nn.compact
    def __call__(self, x, y, dlatents_in, noise_mode='random', feature_list=None, rng=None):
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
            feature_list (list): List of activations.
            rng (jax.random.PRNGKey): Random PRNG for spatialwise noise.

        Returns:
            (tensor): Output tensor of shape [N, H', W', fmaps].
        """
        if rng is None:
            rng = random.PRNGKey(0)
        
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
                               dtype=self.dtype,
                               rng=self.rng)(x, dlatents_in, noise_mode, rng)
            feature_list.append(x)

        if self.num_layers == 2:
            k = ops.setup_filter(self.resample_kernel)
            y = ops.upsample2d(y, f=k, up=2)
        
        y = ToRGBLayer(fmaps=self.num_channels, 
                       layer_idx=self.res * 2 - 3, 
                       lr_multiplier=self.lr_multiplier,
                       dtype=self.dtype)(x, y, dlatents_in)
        return x, y, feature_list


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
        activation (str): Activation function: 'relu', 'lrelu', etc.
        use_noise (bool): If True, add spatial-specific noise.
        resample_kernel (Tuple): Kernel that is used for FIR filter.
        fused_modconv (bool): If True, Perform modulation, convolution, and demodulation as a single fused operation.
        num_fp16_res (int): Use float16 for the 'num_fp16_res' highest resolutions.
        clip_conv (float): Clip the output of convolution layers to [-clip_conv, +clip_conv], None = disable clipping.
        dtype (str): Data type.
        rng (jax.random.PRNGKey): Random PRNG for noise const initialization.
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

    # Internal details
    activation: str='leaky_relu'
    use_noise: bool=True
    resample_kernel: Tuple=(1, 3, 3, 1)
    fused_modconv: bool=False
    num_fp16_res: int=0
    clip_conv: float=None
    dtype: str='float32'
    rng: Any=field(default_factory=lambda : random.PRNGKey(0))

    @nn.compact
    def __call__(self, dlatents_in, noise_mode='random', rng=None):
        """
        Run Synthesis Network.

        Args:
            dlatents_in (tensor): Intermediate latents (W) of shape [N, num_ws, w_dim].
            noise_mode (str): Noise type.
                              - 'const': Constant noise.
                              - 'random': Random noise.
                              - 'none': No noise.
            return_features (bool): If True, additionally return a list of activations for each synthesis layer.
            rng (jax.random.PRNGKey): Random PRNG for spatialwise noise.

        Returns:
            (tensor): Image of shape [N, H, W, num_channels].
        """
        if rng is None:
            rng = random.PRNGKey(0)
        resolution_log2 = int(np.log2(self.resolution))
        assert self.resolution == 2 ** resolution_log2 and self.resolution >= 4

        def nf(stage): return np.clip(int(self.fmap_base / (2.0 ** (stage * self.fmap_decay))), self.fmap_min, self.fmap_max)
        num_layers = resolution_log2 * 2 - 2
        
        fmaps = self.fmap_const if self.fmap_const is not None else nf(1)
        
        def const_init(key, shape, dtype=self.dtype):
            return random.normal(key, shape, dtype=dtype)
        x = self.param('const', const_init, (1, 4, 4, fmaps))
        x = jnp.repeat(x, repeats=dlatents_in.shape[0], axis=0)

        y = None

        dlatents_in = dlatents_in.astype(jnp.float32)
        
        feature_list = []
        for res in range(2, resolution_log2 + 1):
            x, y, feature_list = SynthesisBlock(fmaps=nf(res - 1),
                                                res=res,
                                                num_layers=1 if res == 2 else 2,
                                                num_channels=self.num_channels,
                                                activation=self.activation,
                                                use_noise=self.use_noise,
                                                resample_kernel=self.resample_kernel,
                                                fused_modconv=self.fused_modconv,
                                                clip_conv=self.clip_conv,
                                                dtype=self.dtype if res > resolution_log2 - self.num_fp16_res else 'float32',
                                                rng=self.rng)(x, y, dlatents_in, noise_mode, feature_list, rng)
        
        return y, feature_list


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
        use_noise (bool): If True, add spatial-specific noise.
        activation (str): Activation function: 'relu', 'lrelu', etc.
        w_avg_beta (float): Decay for tracking the moving average of W during training, None = do not track.
        mapping_lr_multiplier (float): Learning rate multiplier for the mapping network.
        resample_kernel (Tuple): Kernel that is used for FIR filter.
        fused_modconv (bool): If True, Perform modulation, convolution, and demodulation as a single fused operation.
        num_fp16_res (int): Use float16 for the 'num_fp16_res' highest resolutions.
        clip_conv (float): Clip the output of convolution layers to [-clip_conv, +clip_conv], None = disable clipping.
        dtype (str): Data type.
        rng (jax.random.PRNGKey): Random PRNG for noise const initialization.
    """
    # Dimensionality
    resolution: int=256
    num_channels: int=3
    z_dim: int=512
    c_dim: int=0
    w_dim: int=512
    mapping_layer_features: int=512
    mapping_embed_features: int=None

    # Layers
    num_ws: int=13
    num_mapping_layers: int=8

    # Capacity
    fmap_base: int=16384
    fmap_decay: int=1
    fmap_min: int=1
    fmap_max: int=512
    fmap_const: int=None

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
    rng: Any=field(default_factory=lambda : random.PRNGKey(0))

    @nn.compact
    def __call__(self,
                 z,
                 c=None,
                 truncation_psi=1,
                 truncation_cutoff=None,
                 skip_w_avg_update=False,
                 train=True,
                 noise_mode='random',
                 rng=None):
        """
        Run Generator.

        Args:
            z (tensor): Input noise, shape [N, z_dim].
            c (tensor): Input labels, shape [N, c_dim].
            return_features (bool): If True, additionally return a list of activations for each synthesis layer.
            truncation_psi (float): Controls truncation (trading off variation for quality). If 1, truncation is disabled.
            truncation_cutoff (int): Controls truncation. None = disable.
            skip_w_avg_update (bool): If True, updates the exponential moving average of W.
            train (bool): Training mode.
            noise_mode (str): Noise type.
                              - 'const': Constant noise.
                              - 'random': Random noise.
                              - 'none': No noise.
            rng (jax.random.PRNGKey): Random PRNG for spatialwise noise.

        Returns:
            (tensor): Image of shape [N, H, W, num_channels].
        """
        if rng is None:
            rng = random.PRNGKey(0)
        dlatents_in = MappingNetwork(z_dim=self.z_dim,
                                     c_dim=self.c_dim,
                                     w_dim=self.w_dim,
                                     num_ws=self.num_ws,
                                     num_layers=self.num_mapping_layers,
                                     embed_features=self.mapping_embed_features,
                                     layer_features=self.mapping_layer_features,
                                     activation=self.activation,
                                     lr_multiplier=self.mapping_lr_multiplier,
                                     w_avg_beta=self.w_avg_beta,
                                     dtype=self.dtype,
                                     name='mapping_network')(z, c, truncation_psi, truncation_cutoff, skip_w_avg_update, train)
        
        synthesis_out = SynthesisNetwork(resolution=self.resolution,
                                         num_channels=self.num_channels,
                                         w_dim=self.w_dim,
                                         fmap_base=self.fmap_base,
                                         fmap_decay=self.fmap_decay,
                                         fmap_min=self.fmap_min,
                                         fmap_max=self.fmap_max,
                                         fmap_const=self.fmap_const,
                                         activation=self.activation,
                                         use_noise=self.use_noise,
                                         resample_kernel=self.resample_kernel,
                                         fused_modconv=self.fused_modconv,
                                         num_fp16_res=self.num_fp16_res,
                                         clip_conv=self.clip_conv,
                                         dtype=self.dtype,
                                         rng=self.rng,
                                         name='synthesis_network')(dlatents_in, noise_mode, rng)

        x, feature_list = synthesis_out
        return x, feature_list

