# Deep Residual Learning for Image Recognition

<b>Paper:</b> <a href="https://arxiv.org/abs/1512.03385">https://arxiv.org/abs/1512.03385</a>  
<b>Repository:</b> <a href="https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py">https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py</a>

##### Table of Contents 
* [1. Important Note](#note)
* [2. Basic Usage](#usage)
* [3. Documentation](#documentation)
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
The documentation can be found [here](../../docs/Documentation.md#resnet).

<a name="license"></a>
## 4. License
<a href="https://opensource.org/licenses/MIT">MIT License</a>


