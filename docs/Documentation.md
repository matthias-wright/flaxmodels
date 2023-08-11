## Documentation

##### Table of Contents  
* [1. Checkpoints](#ckpts)
  * [1.1 Preprocessing](#prepro)
  * [1.2 Structure](#structure)
* [2. Testing](#tests)
* [3. Models](#models)
  * [3.1 GPT2](#gpt2)
  * [3.2 StyleGAN2](#stylegan2)
  * [3.3 ResNet{18, 34, 50, 101, 152}](#resnet)
  * [3.4 VGG{16, 19}](#vgg)
  * [3.5 FewShotGanAdaption](#few_shot_gan_adaption)


<a name="ckpts"></a>
## 1. Checkpoints

<a name="prepro"></a>
### 1.1 Preprocessing
The checkpoints are taken from the repositories that are referenced on the model pages. The parameter values of the checkpoints are not altered. However, in some cases the parameters are reshaped. This is because there does not exist a canonical format among deep learning frameworks. For example, the weights of a 2D convolution in PyTorch have shape `[out_channels, in_channels, kernel_height, kernel_width]`, whereas in TensorFlow and Flax the shape is `[kernel_height, kernel_width, in_channels, out_channels]`.  
Furthermore, the structure of the checkpoints is brought into a canonical form that follows the structure of the corresponding model and is consistent across all models.

<a name="structure"></a>
### 1.2 Structure
The checkpoints are stored in a <a href="https://docs.h5py.org/en/stable/quick.html">HDF5 file</a>. The weights of each component of a model are stored in a dedicated group. It is probably easiest to illustrate this with an example.  
Consider a simplified GPT2 with only two blocks and without the language model head.

<div align="center"><img src="img/gpt2_diagram.png" alt="GPT2" width="300"></div>

The corresponding checkpoint file will have the following structure. The groups contain the parameters which are used in the corresponding module.
```
gpt2
-> token_embedding
-> position_embedding
-> block0
   -> layer_norm1
   -> attention
   -> layer_norm2
   -> mlp
-> block1
   -> layer_norm1
   -> attention
   -> layer_norm2
   -> mlp
-> layer_norm
```

<a name="tests"></a>
## 2. Testing
The functionality of each model is tested with several black-box tests. The majority of these tests compare the output of a model to the output of the original model (contained in the referenced repository) for the same input.  
For example, for the resnet networks, the logit outputs from the original models for [this](https://github.com/matthias-wright/flaxmodels/blob/main/tests/aux_files/elefant.jpg) image are saved [here](https://github.com/matthias-wright/flaxmodels/tree/main/tests/resnet/aux_files).  
In order to test the resnet networks, we compute the logit outputs for the aforementioned image and compare them to the saved outputs of the original models.  
Similar tests are performed for the other models.

<a name="models"></a>
## 3. Models
* [3.1 GPT2](#gpt2)
* [3.2 StyleGAN2](#stylegan2)
* [3.3 ResNet{18, 34, 50, 101, 152}](#resnet)
* [3.4 VGG{16, 19}](#vgg)

<a name="gpt2"></a>
### 3.1 GPT2
* [GPT2LMHeadModel](#gpt2_lmhead)
* [GPT2Model](#gpt2_model)

<a name="gpt2_lmhead"></a>
### GPT2LMHeadModel
flaxmodels.gpt2.GPT2LMHeadModel(*config=None, pretrained=None, ckpt_dir=None*)


#### Attributes
* **config (types.SimpleNamespace)** - Configuration file.
* **pretrained (str)** - Which pretrained model to use, None for random initialization. Options:
  * 'gpt2'
  * 'gpt2-medium'
  * 'gpt2-large'
  * 'gpt2-xl'
* **ckpt_dir (str)** - The directory to which the pretrained weights are downloaded. Only relevant if a pretrained model is used. If this argument is None, the weights will be saved to a temp directory.
* **rng (jax.numpy.ndarray)** - Random PRNG.

#### Methods
apply(*input_ids=None, past_key_values=None, input_embds=None, labels=None, position_ids=None, attn_mask=None, head_mask=None, use_cache=False, training=False*)


##### Parameters
* **input_ids (jax.numpy.ndarray)** - Input token ids, shape [B, seq_len].
* **past_key_values (Tuple of Tuples)** - Precomputed hidden keys and values.
* **input_embds (jax.numpy.ndarray)** - Input embeddings, shape [B, seq_len, embd_dim].
* **labels (jax.numpy.ndarray)** - Labels for language modeling, shape [B, seq_len]. Will be shifted inside the model. Ignore label = -100.
* **position_ids (bool)** - Indices of positions of each input sequence tokens in the position embeddings, shape [B, seq_len].
* **attn_mask (jax.numpy.ndarray)** - Mask to avoid performing attention on padding token indices, shape [B, seq_len].
* **head_mask (jax.numpy.ndarray)** - Mask to nullify selected heads of the self-attention modules, shape [num_heads] or [num_layers, num_heads].
* **use_cache (bool)** - If True, keys and values are returned (past_key_values).
* **training (bool)** - If True, training mode on.

<a name="gpt2_model"></a>
### GPT2Model
flaxmodels.gpt2.GPT2Model(*config=None, pretrained=None, ckpt_dir=None, param_dict=None*)

#### Attributes
* **config (types.SimpleNamespace)** - Configuration file.
* **pretrained (str)** - Which pretrained model to use, None for random initialization. Options:
  * 'gpt2'
  * 'gpt2-medium'
  * 'gpt2-large'
  * 'gpt2-xl'
* **ckpt_dir (str)** - The directory to which the pretrained weights are downloaded. Only relevant if a pretrained model is used. If this argument is None, the weights will be saved to a temp directory.
* **param_dict (dict)** - Parameter dict with pretrained parameters. If not None, 'pretrained' will be ignored.


#### Methods
apply(*input_ids=None, past_key_values=None, input_embds=None, labels=None, position_ids=None, attn_mask=None, head_mask=None, use_cache=False, training=False*)

##### Parameters
* **input_ids (jax.numpy.ndarray)** - Input token ids, shape [B, seq_len].
* **past_key_values (Tuple of Tuples)** - Precomputed hidden keys and values.
* **input_embds (jax.numpy.ndarray)** - Input embeddings, shape [B, seq_len, embd_dim].
* **labels (jax.numpy.ndarray)** - Labels for language modeling, shape [B, seq_len]. Will be shifted inside the model. Ignore label = -100.
* **position_ids (bool)** - Indices of positions of each input sequence tokens in the position embeddings, shape [B, seq_len].
* **attn_mask (jax.numpy.ndarray)** - Mask to avoid performing attention on padding token indices, shape [B, seq_len].
* **head_mask (jax.numpy.ndarray)** - Mask to nullify selected heads of the self-attention modules, shape [num_heads] or [num_layers, num_heads].
* **use_cache (bool)** - If True, keys and values are returned (past_key_values).
* **training (bool)** - If True, training mode on.


<a name="stylegan2"></a>
### 3.2 StyleGAN2
* [Generator](#stylegan2_generator)
* [SynthesisNetwork](#stylegan2_syn)
* [MappingNetwork](#stylegan2_map)
* [Discriminator](#stylegan2_discriminator)

<a name="stylegan2_generator"></a>
### Generator

flaxmodels.stylegan2.Generator(*resolution=1024, num_channels=3, z_dim=512, c_dim=0, w_dim=512, mapping_layer_features=512, mapping_embed_features=None, num_ws=18, num_mapping_layers=8, fmap_base=16384, fmap_decay=1, fmap_min=1, fmap_max=512, fmap_const=None, pretrained=None, ckpt_dir=None, use_noise=True, randomize_noise=True, activation='leaky_relu', w_avg_beta=0.995, mapping_lr_multiplier=0.01, resample_kernel=[1, 3, 3, 1], fused_modconv=False, dtype='float32', rng=jax.random.PRNGKey(0)*)

#### Attributes
* **resolution (int)** - Output resolution.
* **num_channels (int)** - Number of output color channels.
* **z_dim (int)** - Input latent (W) dimensionality.
* **c_dim (int)** - Conditioning label (C) dimensionality, 0 = no label.
* **w_dim (int)** - Input latent (Z) dimensionality.
* **mapping_layer_features (int)** - Number of intermediate features in the mapping layers, None = same as w_dim.
* **mapping_embed_features (int)** - Label embedding dimensionality, None = same as w_dim.
* **num_ws (int)** - Number of intermediate latents to output, None = do not broadcast.
* **num_mapping_layers (int)** - Number of mapping layers.
* **fmap_base (int)** - Overall multiplier for the number of feature maps.
* **fmap_decay (int)** - Log2 feature map reduction when doubling the resolution.
* **fmap_min (int)** - Minimum number of feature maps in any layer.
* **fmap_max (int)** - Maximum number of feature maps in any layer.
* **fmap_const (int)** - Number of feature maps in the constant input layer. None = default.
* **pretrained (str)** - Which pretrained model to use, None for random initialization. Options:
  * 'afhqcat': AFHQ Cat dataset, resolution 512x512.
  * 'afhqdog': AFHQ Dog dataset, resolution 512x512.
  * 'afhqwild': AFHQ Wild dataset, resolution 512x512.
  * 'brecahad': BreCaHAD dataset, resolution 512x512.
  * 'car': LSUN Car dataset, resolution 512x512.
  * 'cat': LSUN Cat dataset, resolution 256x256.
  * 'church': LSUN Church dataset, resolution 256x256.
  * 'cifar10': CIFAR-10 dataset, resolution 32x32.
  * 'ffhq': FFHQ dataset, resolution 1024x1024.
  * 'horse': LSUN Horse dataset, resolution 256x256.
  * 'metfaces': MetFaces dataset, resolution 1024x1024.
* **ckpt_dir (str)** - The directory to which the pretrained weights are downloaded. Only relevant if a pretrained model is used. If this argument is None, the weights will be saved to a temp directory.
* **use_noise (bool)** - Inject noise in synthesis layers.
* **randomize_noise (bool)** - Use random noise.
* **activation (str)** - Activation function. Options:
  * 'relu'
  * 'leaky_relu'
  * 'linear'
* **w_avg_beta (float)** - Decay for tracking the moving average of W during training, None = do not track.
* **mapping_lr_multiplier (float)** - Learning rate multiplier for mapping network.
* **resample_kernel (list or tuple)** - Low-pass filter to apply when resampling activations, None = box filter.
* **fused_modconv (bool)** - Implement modulated_conv2d_layer() using grouped convolution?
* **dtype (str)** - Data dtype.
* **rng (jax.numpy.ndarray)** - PRNG for initialization.

#### Methods
apply(*z, c=None, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False, train=True*)


##### Parameters
* **z (jax.numpy.ndarray)** - Noise inputs, shape [batch, z_dim].
* **c (jax.numpy.ndarray)** - Conditional input, shape [batch, c_dim].
* **truncation_psi (float)** - Parameter that controls the linear interpolation for the truncation trick. 1 = no truncation.
* **truncation_cutoff (int)** - Number of layers for which to apply the truncation trick. None = disable.
* **skip_w_avg_update (bool)** - Don't update moving average for latent variable w.
* **train (bool)** - Training mode.


<a name="stylegan2_syn"></a>
### SynthesisNetwork

flaxmodels.stylegan2.SynthesisNetwork(*resolution=1024, num_channels=3, w_dim=512, fmap_base=16384, fmap_decay=1, fmap_min=1, fmap_max=512, fmap_const=None, pretrained=None, param_dict=None, ckpt_dir=None, activation='leaky_relu', use_noise=True, resample_kernel=[1, 3, 3, 1], fused_modconv=False, num_fp16_res=0, clip_conv=None, dtype='float32', rng=jax.random.PRNGKey(0)*)

#### Attributes
* **resolution (int)** - Output resolution.
* **num_channels (int)** - Number of output color channels.
* **w_dim (int)** - Input latent (W) dimensionality.
* **fmap_base (int)** - Overall multiplier for the number of feature maps.
* **fmap_decay (int)** - Log2 feature map reduction when doubling the resolution.
* **fmap_min (int)** - Minimum number of feature maps in any layer.
* **fmap_max (int)** - Maximum number of feature maps in any layer.
* **fmap_const (int)** - Number of feature maps in the constant input layer. None = default.
* **pretrained (str)** - Which pretrained model to use, None for random initialization. Options:
  * 'afhqcat': AFHQ Cat dataset, resolution 512x512.
  * 'afhqdog': AFHQ Dog dataset, resolution 512x512.
  * 'afhqwild': AFHQ Wild dataset, resolution 512x512.
  * 'brecahad': BreCaHAD dataset, resolution 512x512.
  * 'car': LSUN Car dataset, resolution 512x512.
  * 'cat': LSUN Cat dataset, resolution 256x256.
  * 'church': LSUN Church dataset, resolution 256x256.
  * 'cifar10': CIFAR-10 dataset, resolution 32x32.
  * 'ffhq': FFHQ dataset, resolution 1024x1024.
  * 'horse': LSUN Horse dataset, resolution 256x256.
  * 'metfaces': MetFaces dataset, resolution 1024x1024.
* **param_dict (h5py.Group)** - Parameter dict with pretrained parameters. If not None, 'pretrained' will be ignored.
* **ckpt_dir (str)** - The directory to which the pretrained weights are downloaded. Only relevant if a pretrained model is used. If this argument is None, the weights will be saved to a temp directory.
* **activation (str)** - Activation function. Options:
  * 'relu'
  * 'leaky_relu'
  * 'linear'
* **use_noise (bool)** - Inject noise in synthesis layers.
* **resample_kernel (list or tuple)** - Low-pass filter to apply when resampling activations, None = box filter.
* **fused_modconv (bool)** - Implement modulated_conv2d_layer() using grouped convolution?
* **num_fp16_res (int)** - Use float16 for the 'num_fp16_res' highest resolutions.
* **clip_conv (float)** - Clip the output of convolution layers to [-clip_conv, +clip_conv], None = disable clipping.
* **dtype (str)** - Data dtype.
* **rng (jax.numpy.ndarray)** - PRNG for initialization.

#### Methods
apply(*dlatents_in, noise_mode='random', rng=None)

##### Parameters
* **dlatents_in (jax.numpy.ndarray)** - Latent input W, shape [batch, w_dim].
* **noise_mode (str)** - Noise type:
  * 'const': Constant noise.
  * 'random': Random noise.
  * 'none': No noise.
* **rng (jax.random.PRNGKey)** - Random PRNG for spatialwise noise.


<a name="stylegan2_map"></a>
### MappingNetwork
flaxmodels.stylegan2.MappingNetwork(*z_dim=512, c_dim=0, w_dim=512, embed_features=None, layer_features=512, num_ws=18, num_layers=8, pretrained=None, param_dict=None, ckpt_dir=None, activation='leaky_relu', lr_multiplier=0.01, w_avg_beta=0.995, dtype='float32')

#### Attributes
* **z_dim (int)** - Input latent (Z) dimensionality.
* **c_dim (int)** - Input latent (C) dimensionality, 0 = no label.
* **w_dim (int)** - Input latent (W) dimensionality.
* **embed_features (int)** - Label embedding dimensionality, None = same as w_dim.
* **layer_features (int)** - Number of intermediate features in the mapping layers, None = same as w_dim.
* **num_ws (int)** - Number of intermediate latents to output, None = do not broadcast.
* **num_layers (int)** - Number of mapping layers.
* **pretrained (str)** - Which pretrained model to use, None for random initialization. Options:
  * 'afhqcat': AFHQ Cat dataset, resolution 512x512.
  * 'afhqdog': AFHQ Dog dataset, resolution 512x512.
  * 'afhqwild': AFHQ Wild dataset, resolution 512x512.
  * 'brecahad': BreCaHAD dataset, resolution 512x512.
  * 'car': LSUN Car dataset, resolution 512x512.
  * 'cat': LSUN Cat dataset, resolution 256x256.
  * 'church': LSUN Church dataset, resolution 256x256.
  * 'cifar10': CIFAR-10 dataset, resolution 32x32.
  * 'ffhq': FFHQ dataset, resolution 1024x1024.
  * 'horse': LSUN Horse dataset, resolution 256x256.
  * 'metfaces': MetFaces dataset, resolution 1024x1024.
* **param_dict (h5py.Group)** - Parameter dict with pretrained parameters. If not None, 'pretrained' will be ignored.
* **ckpt_dir (str)** - The directory to which the pretrained weights are downloaded. Only relevant if a pretrained model is used. If this argument is None, the weights will be saved to a temp directory.
* **activation (str)** - Activation function. Options:
  * 'relu'
  * 'leaky_relu'
  * 'linear'
* **lr_multiplier (float)** - Learning rate multiplier for the mapping layers.
* **w_avg_beta (float)** - Decay for tracking the moving average of W during training, None = do not track.
* **dtype (str)** - Data dtype.

#### Methods
apply(*z, c=None, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False, train=True*)

##### Parameters
* **z (jax.numpy.ndarray)** - Noise inputs, shape [batch, z_dim].
* **c (jax.numpy.ndarray)** - Conditional input, shape [batch, c_dim].
* **truncation_psi (float)** - Parameter that controls the linear interpolation for the truncation trick. 1 = no truncation.
* **truncation_cutoff (int)** - Number of layers for which to apply the truncation trick. None = disable.
* **skip_w_avg_update (bool)** - Don't update moving average for latent variable w.
* **train (bool)** - Training mode.


<a name="stylegan2_discriminator"></a>
### Discriminator
flaxmodels.stylegan2.Discriminator(*resolution=3, num_channels=3, c_dim=0, fmap_base=16384, fmap_decay=1, fmap_min=1, fmap_max=512, mapping_layers=0, mapping_fmaps=None, mapping_lr_multiplier=0.1, architecture='resnet', activation='leaky_relu', mbstd_group_size=None, mbstd_num_features=1, resample_kernel=[1, 3, 3, 1], pretrained=None, ckpt_dir=None, dtype='float32'*)

#### Attributes
* **resolution (int)** - Output resolution.
* **num_channels (int)** - Number of output color channels.
* **c_dim (int)** - Dimensionality of the labels (c), 0 if no labels. Overritten based on dataset.
* **fmap_base (int)** - Overall multiplier for the number of feature maps.
* **fmap_decay (int)** - Log2 feature map reduction when doubling the resolution.
* **fmap_min (int)** - Minimum number of feature maps in any layer.
* **fmap_max (int)** - Maximum number of feature maps in any layer.
* **mapping_layers (int)** - Number of additional mapping layers for the conditioning labels.
* **mapping_fmaps (int)** - Number of activations in the mapping layers, None = default.
* **mapping_lr_multiplier (int)** - Learning rate multiplier for the mapping layers.
* **architecture (str)** - Architecture. Options:
  * 'orig'
  * 'resnet'
* **activation (str)** - Activation function. Options:
  * 'relu'
  * 'leaky_relu'
  * 'linear'
* **mbstd_group_size (int)** - Group size for the minibatch standard deviation layer, None = entire minibatch.
* **mbstd_num_features (int)** - Number of features for the minibatch standard deviation layer, 0 = disable.
* **resample_kernel (list or tuple)** - Low-pass filter to apply when resampling activations, None = box filter.
* **pretrained (str)** - Which pretrained model to use, None for random initialization. Options:
  * 'afhqcat': AFHQ Cat dataset, resolution 512x512.
  * 'afhqdog': AFHQ Dog dataset, resolution 512x512.
  * 'afhqwild': AFHQ Wild dataset, resolution 512x512.
  * 'brecahad': BreCaHAD dataset, resolution 512x512.
  * 'car': LSUN Car dataset, resolution 512x512.
  * 'cat': LSUN Cat dataset, resolution 256x256.
  * 'church': LSUN Church dataset, resolution 256x256.
  * 'cifar10': CIFAR-10 dataset, resolution 32x32.
  * 'ffhq': FFHQ dataset, resolution 1024x1024.
  * 'horse': LSUN Horse dataset, resolution 256x256.
  * 'metfaces': MetFaces dataset, resolution 1024x1024.
* **ckpt_dir (str)** - The directory to which the pretrained weights are downloaded. Only relevant if a pretrained model is used. If this argument is None, the weights will be saved to a temp directory.
* **dtype (str)** - Data dtype.

#### Methods
apply(*z, c=None*)

##### Parameters
* **z (jax.numpy.ndarray)** - Noise inputs, shape [batch, z_dim].
* **c (jax.numpy.ndarray)** - Conditional input, shape [batch, c_dim].


<a name="resnet"></a>
### 3.3 ResNet{18, 34, 50, 101, 152}
* [ResNet18](#resnet18)
* [ResNet34](#resnet34)
* [ResNet50](#resnet50)
* [ResNet101](#resnet101)
* [ResNet152](#resnet152)


<a name="resnet18"></a>
### ResNet18
flaxmodels.ResNet18(*output='softmax', pretrained='imagenet', normalize=True, kernel_init=flax.linen.initializers.lecun_normal(), bias_init=flax.linen.initializers.zeros, ckpt_dir=None, dtype='float32'*) -> flax.linen.Module

#### Parameters
* **output (str)** - Output of the network. Options:
   * 'softmax': Output is a softmax tensor of shape [N, 1000].
   * 'logits': Output is a tensor of shape [N, 1000].
   * 'activations': Output is a dictionary containing the ResNet activations.
* **pretrained (str)** - Which pretrained weights to load. Options:
  * 'imagenet': Loads the network parameters trained on ImageNet.
  * None: Parameters of the module are initialized randomly.
* **normalize (bool)** - If True, the input will be normalized with the ImageNet statistics.
* **kernel_init (callable)** - A function that takes in a shape and returns a tensor for initializing the weights.
* **kernel_init (callable)** - A function that takes in a shape and returns a tensor for initializing the biases.
* **ckpt_dir (str)** - The directory to which the pretrained weights are downloaded. Only relevant if a pretrained model is used. If this argument is None, the weights will be saved to a temp directory.
* **dtype (str)** - Data type.


<a name="resnet34"></a>
### ResNet34
flaxmodels.ResNet34(*output='softmax', pretrained='imagenet', normalize=True, include_head=True, kernel_init=flax.linen.initializers.lecun_normal(), bias_init=flax.linen.initializers.zeros, ckpt_dir=None, dtype='float32'*) -> flax.linen.Module

#### Parameters
* **output (str)** - Output of the network. Options:
   * 'softmax': Output is a softmax tensor of shape [N, 1000].
   * 'logits': Output is a tensor of shape [N, 1000].
   * 'activations': Output is a dictionary containing the ResNet activations.
* **pretrained (str)** - Which pretrained weights to load. Options:
  * 'imagenet': Loads the network parameters trained on ImageNet.
  * None: Parameters of the module are initialized randomly.
* **normalize (bool)** - If True, the input will be normalized with the ImageNet statistics.
* **kernel_init (callable)** - A function that takes in a shape and returns a tensor for initializing the weights.
* **kernel_init (callable)** - A function that takes in a shape and returns a tensor for initializing the biases.
* **ckpt_dir (str)** - The directory to which the pretrained weights are downloaded. Only relevant if a pretrained model is used. If this argument is None, the weights will be saved to a temp directory.
* **dtype (str)** - Data type.


<a name="resnet50"></a>
### ResNet50
flaxmodels.ResNet50(*output='softmax', pretrained='imagenet', normalize=True, include_head=True, kernel_init=flax.linen.initializers.lecun_normal(), bias_init=flax.linen.initializers.zeros, ckpt_dir=None, dtype='float32'*) -> flax.linen.Module

#### Parameters
* **output (str)** - Output of the network. Options:
   * 'softmax': Output is a softmax tensor of shape [N, 1000].
   * 'logits': Output is a tensor of shape [N, 1000].
   * 'activations': Output is a dictionary containing the ResNet activations.
* **pretrained (str)** - Which pretrained weights to load. Options:
  * 'imagenet': Loads the network parameters trained on ImageNet.
  * None: Parameters of the module are initialized randomly.
* **normalize (bool)** - If True, the input will be normalized with the ImageNet statistics.
* **kernel_init (callable)** - A function that takes in a shape and returns a tensor for initializing the weights.
* **kernel_init (callable)** - A function that takes in a shape and returns a tensor for initializing the biases.
* **ckpt_dir (str)** - The directory to which the pretrained weights are downloaded. Only relevant if a pretrained model is used. If this argument is None, the weights will be saved to a temp directory.
* **dtype (str)** - Data type.


<a name="resnet101"></a>
### ResNet101
flaxmodels.ResNet101(*output='softmax', pretrained='imagenet', normalize=True, include_head=True, kernel_init=flax.linen.initializers.lecun_normal(), bias_init=flax.linen.initializers.zeros, ckpt_dir=None, dtype='float32'*) -> flax.linen.Module

#### Parameters
* **output (str)** - Output of the network. Options:
   * 'softmax': Output is a softmax tensor of shape [N, 1000].
   * 'logits': Output is a tensor of shape [N, 1000].
   * 'activations': Output is a dictionary containing the ResNet activations.
* **pretrained (str)** - Which pretrained weights to load. Options:
  * 'imagenet': Loads the network parameters trained on ImageNet.
  * None: Parameters of the module are initialized randomly.
* **normalize (bool)** - If True, the input will be normalized with the ImageNet statistics.
* **kernel_init (callable)** - A function that takes in a shape and returns a tensor for initializing the weights.
* **kernel_init (callable)** - A function that takes in a shape and returns a tensor for initializing the biases.
* **ckpt_dir (str)** - The directory to which the pretrained weights are downloaded. Only relevant if a pretrained model is used. If this argument is None, the weights will be saved to a temp directory.
* **dtype (str)** - Data type.


<a name="resnet152"></a>
### ResNet152
flaxmodels.ResNet152(*output='softmax', pretrained='imagenet', normalize=True, include_head=True, kernel_init=flax.linen.initializers.lecun_normal(), bias_init=flax.linen.initializers.zeros, ckpt_dir=None, dtype='float32'*) -> flax.linen.Module

#### Parameters
* **output (str)** - Output of the network. Options:
   * 'softmax': Output is a softmax tensor of shape [N, 1000].
   * 'logits': Output is a tensor of shape [N, 1000].
   * 'activations': Output is a dictionary containing the ResNet activations.
* **pretrained (str)** - Which pretrained weights to load. Options:
  * 'imagenet': Loads the network parameters trained on ImageNet.
  * None: Parameters of the module are initialized randomly.
* **normalize (bool)** - If True, the input will be normalized with the ImageNet statistics.
* **kernel_init (callable)** - A function that takes in a shape and returns a tensor for initializing the weights.
* **kernel_init (callable)** - A function that takes in a shape and returns a tensor for initializing the biases.
* **ckpt_dir (str)** - The directory to which the pretrained weights are downloaded. Only relevant if a pretrained model is used. If this argument is None, the weights will be saved to a temp directory.
* **dtype (str)** - Data type.


<a name="vgg"></a>
### 3.4 VGG{16, 19}
* [VGG16](#vgg16)
* [VGG19](#vgg19)

<a name="vgg16"></a>
### VGG16
flaxmodels.VGG16(*output='softmax', pretrained='imagenet', normalize=True, include_head=True, kernel_init=flax.linen.initializers.lecun_normal(), bias_init=flax.linen.initializers.zeros, ckpt_dir=None, dtype='float32'*) -> flax.linen.Module

#### Parameters
* **output (str)** - Output of the network. Options:
   * 'softmax': Output is a softmax tensor of shape [N, 1000].
   * 'logits': Output is a tensor of shape [N, 1000].
   * 'activations': Output is a dictionary containing the VGG activations.
* **pretrained (str)** - Which pretrained weights to load. Options:
  * 'imagenet': Loads the network parameters trained on ImageNet.
  * None: Parameters of the module are initialized randomly.
* **normalize (bool)** - If True, the input will be normalized with the ImageNet statistics.
* **include_head (bool)** - If True, include the three fully-connected layers at the top of the network. This option is useful when you want to obtain activations for images whose size is different than 224x224.
* **kernel_init (callable)** - A function that takes in a shape and returns a tensor for initializing the weights.
* **kernel_init (callable)** - A function that takes in a shape and returns a tensor for initializing the biases.
* **ckpt_dir (str)** - The directory to which the pretrained weights are downloaded. Only relevant if a pretrained model is used. If this argument is None, the weights will be saved to a temp directory.
* **dtype (str)** - Data type.


<a name="vgg19"></a>
### VGG19
flaxmodels.VGG19(*output='softmax', pretrained='imagenet', normalize=True, include_head=True, kernel_init=flax.linen.initializers.lecun_normal(), bias_init=flax.linen.initializers.zeros, ckpt_dir=None, dtype='float32'*) -> flax.linen.Module

#### Parameters
* **output (str)** - Output of the network. Options:
   * 'softmax': Output is a softmax tensor of shape [N, 1000].
   * 'logits': Output is a tensor of shape [N, 1000].
   * 'activations': Output is a dictionary containing the VGG activations.
* **pretrained (str)** - Which pretrained weights to load. Options:
  * 'imagenet': Loads the network parameters trained on ImageNet.
  * None: Parameters of the module are initialized randomly.
* **normalize (bool)** - If True, the input will be normalized with the ImageNet statistics.
* **include_head (bool)** - If True, include the three fully-connected layers at the top of the network. This option is useful when you want to obtain activations for images whose size is different than 224x224.
* **kernel_init (callable)** - A function that takes in a shape and returns a tensor for initializing the weights.
* **kernel_init (callable)** - A function that takes in a shape and returns a tensor for initializing the biases.
* **ckpt_dir (str)** - The directory to which the pretrained weights are downloaded. Only relevant if a pretrained model is used. If this argument is None, the weights will be saved to a temp directory.
* **dtype (str)** - Data type.


<a name="few_shot_gan_adaption"></a>
### 3.2 FewShotGanAdaption
* [Generator](#few_shot_gan_adaption_generator)
* [SynthesisNetwork](#few_shot_gan_adaption_syn)
* [MappingNetwork](#stylegan2_map)
* [Discriminator](#stylegan2_discriminator)

<a name="few_shot_gan_adaption_generator"></a>
### Generator

flaxmodels.few_shot_gan_adaption.Generator(*resolution=1024, num_channels=3, z_dim=512, c_dim=0, w_dim=512, mapping_layer_features=512, mapping_embed_features=None, num_ws=18, num_mapping_layers=8, fmap_base=16384, fmap_decay=1, fmap_min=1, fmap_max=512, fmap_const=None, use_noise=True, randomize_noise=True, activation='leaky_relu', w_avg_beta=0.995, mapping_lr_multiplier=0.01, resample_kernel=[1, 3, 3, 1], fused_modconv=False, dtype='float32', rng=jax.random.PRNGKey(0)*)

#### Attributes
* **resolution (int)** - Output resolution.
* **num_channels (int)** - Number of output color channels.
* **z_dim (int)** - Input latent (W) dimensionality.
* **c_dim (int)** - Conditioning label (C) dimensionality, 0 = no label.
* **w_dim (int)** - Input latent (Z) dimensionality.
* **mapping_layer_features (int)** - Number of intermediate features in the mapping layers, None = same as w_dim.
* **mapping_embed_features (int)** - Label embedding dimensionality, None = same as w_dim.
* **num_ws (int)** - Number of intermediate latents to output, None = do not broadcast.
* **num_mapping_layers (int)** - Number of mapping layers.
* **fmap_base (int)** - Overall multiplier for the number of feature maps.
* **fmap_decay (int)** - Log2 feature map reduction when doubling the resolution.
* **fmap_min (int)** - Minimum number of feature maps in any layer.
* **fmap_max (int)** - Maximum number of feature maps in any layer.
* **fmap_const (int)** - Number of feature maps in the constant input layer. None = default.
* **use_noise (bool)** - Inject noise in synthesis layers.
* **randomize_noise (bool)** - Use random noise.
* **activation (str)** - Activation function. Options:
  * 'relu'
  * 'leaky_relu'
  * 'linear'
* **w_avg_beta (float)** - Decay for tracking the moving average of W during training, None = do not track.
* **mapping_lr_multiplier (float)** - Learning rate multiplier for mapping network.
* **resample_kernel (list or tuple)** - Low-pass filter to apply when resampling activations, None = box filter.
* **fused_modconv (bool)** - Implement modulated_conv2d_layer() using grouped convolution?
* **dtype (str)** - Data dtype.
* **rng (jax.numpy.ndarray)** - PRNG for initialization.

#### Methods
apply(*z, c=None, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False, train=True*)


##### Parameters
* **z (jax.numpy.ndarray)** - Noise inputs, shape [batch, z_dim].
* **c (jax.numpy.ndarray)** - Conditional input, shape [batch, c_dim].
* **truncation_psi (float)** - Parameter that controls the linear interpolation for the truncation trick. 1 = no truncation.
* **truncation_cutoff (int)** - Number of layers for which to apply the truncation trick. None = disable.
* **skip_w_avg_update (bool)** - Don't update moving average for latent variable w.
* **train (bool)** - Training mode.


<a name="few_shot_gan_adaption_syn"></a>
### SynthesisNetwork

flaxmodels.few_shot_gan_adaption.SynthesisNetwork(*resolution=1024, num_channels=3, w_dim=512, fmap_base=16384, fmap_decay=1, fmap_min=1, fmap_max=512, fmap_const=None, activation='leaky_relu', use_noise=True, resample_kernel=[1, 3, 3, 1], fused_modconv=False, num_fp16_res=0, clip_conv=None, dtype='float32', rng=jax.random.PRNGKey(0)*)

#### Attributes
* **resolution (int)** - Output resolution.
* **num_channels (int)** - Number of output color channels.
* **w_dim (int)** - Input latent (W) dimensionality.
* **fmap_base (int)** - Overall multiplier for the number of feature maps.
* **fmap_decay (int)** - Log2 feature map reduction when doubling the resolution.
* **fmap_min (int)** - Minimum number of feature maps in any layer.
* **fmap_max (int)** - Maximum number of feature maps in any layer.
* **fmap_const (int)** - Number of feature maps in the constant input layer. None = default.
* **activation (str)** - Activation function. Options:
  * 'relu'
  * 'leaky_relu'
  * 'linear'
* **use_noise (bool)** - Inject noise in synthesis layers.
* **resample_kernel (list or tuple)** - Low-pass filter to apply when resampling activations, None = box filter.
* **fused_modconv (bool)** - Implement modulated_conv2d_layer() using grouped convolution?
* **num_fp16_res (int)** - Use float16 for the 'num_fp16_res' highest resolutions.
* **clip_conv (float)** - Clip the output of convolution layers to [-clip_conv, +clip_conv], None = disable clipping.
* **dtype (str)** - Data dtype.
* **rng (jax.numpy.ndarray)** - PRNG for initialization.

#### Methods
apply(*dlatents_in, noise_mode='random', rng=None*)

##### Parameters
* **dlatents_in (jax.numpy.ndarray)** - Latent input W, shape [batch, w_dim].
* **noise_mode (str)** - Noise type:
  * 'const': Constant noise.
  * 'random': Random noise.
  * 'none': No noise.
* **rng (jax.random.PRNGKey)** - Random PRNG for spatialwise noise.


<a name="few_shot_gan_adaption_map"></a>
### MappingNetwork
flaxmodels.few_shot_gan_adaption.MappingNetwork(*z_dim=512, c_dim=0, w_dim=512, embed_features=None, layer_features=512, num_ws=18, num_layers=8, activation='leaky_relu', lr_multiplier=0.01, w_avg_beta=0.995, dtype='float32')

#### Attributes
* **z_dim (int)** - Input latent (Z) dimensionality.
* **c_dim (int)** - Input latent (C) dimensionality, 0 = no label.
* **w_dim (int)** - Input latent (W) dimensionality.
* **embed_features (int)** - Label embedding dimensionality, None = same as w_dim.
* **layer_features (int)** - Number of intermediate features in the mapping layers, None = same as w_dim.
* **num_ws (int)** - Number of intermediate latents to output, None = do not broadcast.
* **num_layers (int)** - Number of mapping layers.
* **activation (str)** - Activation function. Options:
  * 'relu'
  * 'leaky_relu'
  * 'linear'
* **lr_multiplier (float)** - Learning rate multiplier for the mapping layers.
* **w_avg_beta (float)** - Decay for tracking the moving average of W during training, None = do not track.
* **dtype (str)** - Data dtype.

#### Methods
apply(*z, c=None, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False, train=True*)

##### Parameters
* **z (jax.numpy.ndarray)** - Noise inputs, shape [batch, z_dim].
* **c (jax.numpy.ndarray)** - Conditional input, shape [batch, c_dim].
* **truncation_psi (float)** - Parameter that controls the linear interpolation for the truncation trick. 1 = no truncation.
* **truncation_cutoff (int)** - Number of layers for which to apply the truncation trick. None = disable.
* **skip_w_avg_update (bool)** - Don't update moving average for latent variable w.
* **train (bool)** - Training mode.


<a name="few_shot_gan_adaption_discriminator"></a>
### Discriminator
flaxmodels.few_shot_gan_adaption.Discriminator(*resolution=3, num_channels=3, c_dim=0, fmap_base=16384, fmap_decay=1, fmap_min=1, fmap_max=512, mapping_layers=0, mapping_fmaps=None, mapping_lr_multiplier=0.1, architecture='resnet', activation='leaky_relu', mbstd_group_size=None, mbstd_num_features=1, resample_kernel=[1, 3, 3, 1], dtype='float32'*)

#### Attributes
* **resolution (int)** - Output resolution.
* **num_channels (int)** - Number of output color channels.
* **c_dim (int)** - Dimensionality of the labels (c), 0 if no labels. Overritten based on dataset.
* **fmap_base (int)** - Overall multiplier for the number of feature maps.
* **fmap_decay (int)** - Log2 feature map reduction when doubling the resolution.
* **fmap_min (int)** - Minimum number of feature maps in any layer.
* **fmap_max (int)** - Maximum number of feature maps in any layer.
* **mapping_layers (int)** - Number of additional mapping layers for the conditioning labels.
* **mapping_fmaps (int)** - Number of activations in the mapping layers, None = default.
* **mapping_lr_multiplier (int)** - Learning rate multiplier for the mapping layers.
* **architecture (str)** - Architecture. Options:
  * 'orig'
  * 'resnet'
* **activation (str)** - Activation function. Options:
  * 'relu'
  * 'leaky_relu'
  * 'linear'
* **mbstd_group_size (int)** - Group size for the minibatch standard deviation layer, None = entire minibatch.
* **mbstd_num_features (int)** - Number of features for the minibatch standard deviation layer, 0 = disable.
* **resample_kernel (list or tuple)** - Low-pass filter to apply when resampling activations, None = box filter.
* **dtype (str)** - Data dtype.

#### Methods
apply(*z, c=None*)

##### Parameters
* **z (jax.numpy.ndarray)** - Noise inputs, shape [batch, z_dim].
* **c (jax.numpy.ndarray)** - Conditional input, shape [batch, c_dim].


