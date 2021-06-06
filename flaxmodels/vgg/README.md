# Very Deep Convolutional Networks for Large-Scale Image Recognition

<b>Paper:</b> <a href="https://arxiv.org/abs/1409.1556">https://arxiv.org/abs/1409.1556</a>  
<b>Project Page:</b> <a href="https://www.robots.ox.ac.uk/~vgg/research/very_deep/">https://www.robots.ox.ac.uk/~vgg/research/very_deep/</a>  
<b>Repository:</b> <a href="https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py">https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py</a>

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
For more usage examples check out this [Colab](vgg_demo.ipynb).

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
out = vgg16.apply(params, x, train=False)

```
Usage is equivalent for VGG19.

<a name="documentation"></a>
## 3. Documentation
The documentation can be found [here](../../docs/Documentation.md#vgg).

<a name="license"></a>
## 4. License
<a href="https://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution License</a>
