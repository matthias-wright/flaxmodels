# Very Deep Convolutional Networks for Large-Scale Image Recognition

<b>Paper:</b> <a href="https://arxiv.org/abs/1409.1556">https://arxiv.org/abs/1409.1556</a>  
<b>Project Page:</b> <a href="https://www.robots.ox.ac.uk/~vgg/research/very_deep/">https://www.robots.ox.ac.uk/~vgg/research/very_deep/</a>  
<b>Repository:</b> <a href="https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py">https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py</a>

##### Table of Contents 
* [1. Important Note](#note)
* [2. Example usages](#usages)
  * [2.1 Get classifier scores](#get_class_scores)
  * [2.2 Get activations (including classifier scores)](#get_activations)
  * [2.3 Get activations for image of arbitrary size](#get_activations_arb)
* [3. Documentation](#documentation)
  * [3.1 VGG16](#vgg16)
  * [3.2 VGG19](#vgg19)
* [4. License](#license)


<a name="note"></a>
## 1. Important Note
Images must be in range [0, 1]. If the pretrained ImageNet weights are selected, the images are internally normalized with the ImageNet mean and standard deviation.

<a name="usages"></a>
## 2. Example usages

<a name="get_class_scores"></a>
### 2.1 Get classifier scores
```python
from PIL import Image
import jax
import jax.numpy as jnp
import flaxmodels as fm

key = jax.random.PRNGKey(0)

# Load image
img = Image.open('example.jpg')
# Image must be 224x224 if classification head is included
img = img.resize((224, 224))
# Image should be in range [0, 1]
x = jnp.array(img, dtype=jnp.float32) / 255.0
# Add batch dimension
x = jnp.expand_dims(x, axis=0)

vgg16 = fm.VGG16(output='logits', pretrained='imagenet')
params = vgg16.init(key, x)
out = vgg16.apply(params, x)

```

<a name="get_activations"></a>
### 2.2 Get activations (including classifier scores)
```python
from PIL import Image
import jax
import jax.numpy as jnp
import flaxmodels as fm

key = jax.random.PRNGKey(0)

# Load image
img = Image.open('example.jpg')
# Image must be 224x224 if classification head is included
img = img.resize((224, 224))
# Image should be in range [0, 1]
x = jnp.array(img, dtype=jnp.float32) / 255.0
# Add batch dimension
x = jnp.expand_dims(x, axis=0)

vgg16 = fm.VGG16(output='activations', pretrained='imagenet')
params = vgg16.init(key, x)
out = vgg16.apply(params, x)

```

<a name="get_activations_arb"></a>
### 2.3 Get activations for image of arbitrary size
```python
from PIL import Image
import jax
import jax.numpy as jnp
import flaxmodels as fm

key = jax.random.PRNGKey(0)

# Load image
img = Image.open('example.jpg')
# Image should be in range [0, 1]
x = jnp.array(img, dtype=jnp.float32) / 255.0
# Add batch dimension
x = jnp.expand_dims(x, axis=0)

vgg16 = fm.VGG16(output='activations', include_head=False, pretrained='imagenet')
params = vgg16.init(key, x)
out = vgg16.apply(params, x)

```

Usage is equivalent for VGG19.

<a name="documentation"></a>
## 3. Documentation

<a name="vgg16"></a>
### 3.1 VGG16
flax_models.VGG16(*output='softmax', pretrained='imagenet', include_head=True, kernel_init=flax.linen.initializers.lecun_normal(), bias_init=flax.linen.initializers.zeros, ckpt_dir=None, rng=jax.random.PRNGKey(0)*) -> flax.linen.Module

#### Parameters
* **output (str)** - Output of the network. Options:
   * 'softmax': Output is a softmax tensor of shape [N, 1000].
   * 'logits': Output is a tensor of shape [N, 1000].
   * 'activations': Output is a dictionary containing the VGG activations.
* **pretrained (str)** - Which pretrained weights to load. Options:
  * 'imagenet': Loads the network parameters trained on ImageNet.
  * None: Parameters of the module are initialized randomly.
* **include_head (bool)** - If True, include the three fully-connected layers at the top of the network. This option is useful when you want to obtain activations for images whose size is different than 224x224.
* **kernel_init (callable)** - A function that takes in a shape and returns a tensor for initializing the weights.
* **kernel_init (callable)** - A function that takes in a shape and returns a tensor for initializing the biases.
* **ckpt_dir (str)** - The directory to which the pretrained weights are downloaded. Only relevant if a pretrained model is used. If this argument is None, the weights will be saved to a temp directory.
* **rng (jax.numpy.ndarray)** - Random seed.


<a name="vgg19"></a>
### 3.2 VGG19
flax_models.VGG19(*output='softmax', pretrained='imagenet', include_head=True, kernel_init=flax.linen.initializers.lecun_normal(), bias_init=flax.linen.initializers.zeros, ckpt_dir=None, rng=jax.random.PRNGKey(0)*) -> flax.linen.Module

#### Parameters
* **output (str)** - Output of the network. Options:
   * 'softmax': Output is a softmax tensor of shape [N, 1000].
   * 'logits': Output is a tensor of shape [N, 1000].
   * 'activations': Output is a dictionary containing the VGG activations.
* **pretrained (str)** - Which pretrained weights to load. Options:
  * 'imagenet': Loads the network parameters trained on ImageNet.
  * None: Parameters of the module are initialized randomly.
* **include_head (bool)** - If True, include the three fully-connected layers at the top of the network. This option is useful when you want to obtain activations for images whose size is different than 224x224.
* **kernel_init (callable)** - A function that takes in a shape and returns a tensor for initializing the weights.
* **kernel_init (callable)** - A function that takes in a shape and returns a tensor for initializing the biases.
* **ckpt_dir (str)** - The directory to which the pretrained weights are downloaded. Only relevant if a pretrained model is used. If this argument is None, the weights will be saved to a temp directory.
* **rng (jax.numpy.ndarray)** - Random seed.

<a name="license"></a>
## 4. License
<a href="https://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution License</a>
