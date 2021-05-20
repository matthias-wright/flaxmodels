# Deep Residual Learning for Image Recognition

<b>Paper:</b> <a href="https://arxiv.org/abs/1512.03385">https://arxiv.org/abs/1512.03385</a>  
<b>Repository:</b> <a href="https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py">https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py</a>

##### Table of Contents 
* [1. Important Note](#note)
* [2. Basic Usage](#usage)
* [3. Documentation](#documentation)
  * [3.1 ResNet18](#resnet18)
  * [3.2 ResNet34](#resnet34)
  * [3.3 ResNet50](#resnet50)
  * [3.4 ResNet101](#resnet101)
  * [3.5 ResNet152](#resnet152)
* [4. License](#license)

<a name="note"></a>
## 1. Important Note
Images must be in range [0, 1]. If the pretrained ImageNet weights are selected, the images are internally normalized with the ImageNet mean and standard deviation.

<a name="usage"></a>
## 2. Basic Usage
For more usage examples check out this [Colab](https://colab.research.google.com/drive/1hjOV3_3OT5xz0iaj4fdCJurL7XWBJUWc?usp=sharing).

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

resnet18 = fm.ResNet18(output='logits', pretrained='imagenet')
params = resnet18.init(key, x)
# Shape [1, 1000]
out = resnet18.apply(params, x)

```
Usage is equivalent for ResNet34, ResNet50, ResNet101, and Resnet152.

<a name="documentation"></a>
## 3. Documentation

<a name="resnet18"></a>
### 3.1 ResNet18
flaxmodels.ResNet18(*output='softmax', pretrained='imagenet', kernel_init=flax.linen.initializers.lecun_normal(), bias_init=flax.linen.initializers.zeros, ckpt_dir=None*) -> flax.linen.Module

#### Parameters
* **output (str)** - Output of the network. Options:
   * 'softmax': Output is a softmax tensor of shape [N, 1000].
   * 'logits': Output is a tensor of shape [N, 1000].
   * 'activations': Output is a dictionary containing the ResNet activations.
* **pretrained (str)** - Which pretrained weights to load. Options:
  * 'imagenet': Loads the network parameters trained on ImageNet.
  * None: Parameters of the module are initialized randomly.
* **kernel_init (callable)** - A function that takes in a shape and returns a tensor for initializing the weights.
* **kernel_init (callable)** - A function that takes in a shape and returns a tensor for initializing the biases.
* **ckpt_dir (str)** - The directory to which the pretrained weights are downloaded. Only relevant if a pretrained model is used. If this argument is None, the weights will be saved to a temp directory.

<a name="resnet34"></a>
### 3.2 ResNet34
flaxmodels.ResNet34(*output='softmax', pretrained='imagenet', include_head=True, kernel_init=flax.linen.initializers.lecun_normal(), bias_init=flax.linen.initializers.zeros, ckpt_dir=None*) -> flax.linen.Module

#### Parameters
* **output (str)** - Output of the network. Options:
   * 'softmax': Output is a softmax tensor of shape [N, 1000].
   * 'logits': Output is a tensor of shape [N, 1000].
   * 'activations': Output is a dictionary containing the ResNet activations.
* **pretrained (str)** - Which pretrained weights to load. Options:
  * 'imagenet': Loads the network parameters trained on ImageNet.
  * None: Parameters of the module are initialized randomly.
* **kernel_init (callable)** - A function that takes in a shape and returns a tensor for initializing the weights.
* **kernel_init (callable)** - A function that takes in a shape and returns a tensor for initializing the biases.
* **ckpt_dir (str)** - The directory to which the pretrained weights are downloaded. Only relevant if a pretrained model is used. If this argument is None, the weights will be saved to a temp directory.

<a name="resnet50"></a>
### 3.3 ResNet50
flaxmodels.ResNet50(*output='softmax', pretrained='imagenet', include_head=True, kernel_init=flax.linen.initializers.lecun_normal(), bias_init=flax.linen.initializers.zeros, ckpt_dir=None*) -> flax.linen.Module

#### Parameters
* **output (str)** - Output of the network. Options:
   * 'softmax': Output is a softmax tensor of shape [N, 1000].
   * 'logits': Output is a tensor of shape [N, 1000].
   * 'activations': Output is a dictionary containing the ResNet activations.
* **pretrained (str)** - Which pretrained weights to load. Options:
  * 'imagenet': Loads the network parameters trained on ImageNet.
  * None: Parameters of the module are initialized randomly.
* **kernel_init (callable)** - A function that takes in a shape and returns a tensor for initializing the weights.
* **kernel_init (callable)** - A function that takes in a shape and returns a tensor for initializing the biases.
* **ckpt_dir (str)** - The directory to which the pretrained weights are downloaded. Only relevant if a pretrained model is used. If this argument is None, the weights will be saved to a temp directory.


<a name="resnet101"></a>
### 3.4 ResNet101
flaxmodels.ResNet101(*output='softmax', pretrained='imagenet', include_head=True, kernel_init=flax.linen.initializers.lecun_normal(), bias_init=flax.linen.initializers.zeros, ckpt_dir=None*) -> flax.linen.Module


#### Parameters
* **output (str)** - Output of the network. Options:
   * 'softmax': Output is a softmax tensor of shape [N, 1000].
   * 'logits': Output is a tensor of shape [N, 1000].
   * 'activations': Output is a dictionary containing the ResNet activations.
* **pretrained (str)** - Which pretrained weights to load. Options:
  * 'imagenet': Loads the network parameters trained on ImageNet.
  * None: Parameters of the module are initialized randomly.
* **kernel_init (callable)** - A function that takes in a shape and returns a tensor for initializing the weights.
* **kernel_init (callable)** - A function that takes in a shape and returns a tensor for initializing the biases.
* **ckpt_dir (str)** - The directory to which the pretrained weights are downloaded. Only relevant if a pretrained model is used. If this argument is None, the weights will be saved to a temp directory.


<a name="resnet152"></a>
### 3.5 ResNet152
flaxmodels.ResNet152(*output='softmax', pretrained='imagenet', include_head=True, kernel_init=flax.linen.initializers.lecun_normal(), bias_init=flax.linen.initializers.zeros, ckpt_dir=None>*) -> flax.linen.Module

#### Parameters
* **output (str)** - Output of the network. Options:
   * 'softmax': Output is a softmax tensor of shape [N, 1000].
   * 'logits': Output is a tensor of shape [N, 1000].
   * 'activations': Output is a dictionary containing the ResNet activations.
* **pretrained (str)** - Which pretrained weights to load. Options:
  * 'imagenet': Loads the network parameters trained on ImageNet.
  * None: Parameters of the module are initialized randomly.
* **kernel_init (callable)** - A function that takes in a shape and returns a tensor for initializing the weights.
* **kernel_init (callable)** - A function that takes in a shape and returns a tensor for initializing the biases.
* **ckpt_dir (str)** - The directory to which the pretrained weights are downloaded. Only relevant if a pretrained model is used. If this argument is None, the weights will be saved to a temp directory.


<a name="license"></a>
## 4. License
<a href="https://opensource.org/licenses/MIT">MIT License</a>


