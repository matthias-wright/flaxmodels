# Analyzing and Improving the Image Quality of StyleGAN

<div align="center"><img src="images/title.jpg" alt="img" width="1050"></div>
  
<b>Paper:</b> <a href="https://arxiv.org/abs/1912.04958">https://arxiv.org/abs/1912.04958</a>  
<b>Repository:</b> <a href="https://github.com/NVlabs/stylegan2">https://github.com/NVlabs/stylegan2</a> and <a href="https://github.com/NVlabs/stylegan2-ada">https://github.com/NVlabs/stylegan2-ada</a>

##### Table of Contents  
* [1. Basic Usage](#usage)
* [2. Documentation](#documentation)
* [3. Training](#training)
* [4. Pretrained Models](#models)
* [5. License](#license)


<a name="usage"></a>
## 1. Basic Usage
For more usage examples check out this [Colab](stylegan2_demo.ipynb).

```python
import numpy as np
from PIL import Image
import jax
import jax.numpy as jnp
import flaxmodels as fm

# Seed
key = jax.random.PRNGKey(0)

# Input noise
z = jax.random.normal(key, shape=(4, 512))

generator = fm.stylegan2.Generator(pretrained='metfaces')
params = generator.init(key, z)
images = generator.apply(params, z, train=False)

# Normalize images to be in range [0, 1]
images = (images - jnp.min(images)) / (jnp.max(images) - jnp.min(images))

# Save images
for i in range(images.shape[0]):
    Image.fromarray(np.uint8(images[i] * 255)).save(f'image_{i}.jpg')

```

<div align="center"><img src="images/gen_images_wo_trunc.jpg" alt="img" width="800"></div>

<a name="documentation"></a>
## 2. Documentation
The documentation can be found [here](../../docs/Documentation.md#stylegan2).

<a name="training"></a>
## 3. Training
If you want to train StyleGAN2 in Jax/Flax, go [here](https://github.com/matthias-wright/flaxmodels/tree/main/training/stylegan2).

<a name="models"></a>
## 4. Pretrained Models

### Metfaces
<div><img src="images/metfaces.jpg" alt="img" width="700"></div>

### FFHQ
<div><img src="images/ffhq.jpg" alt="img" width="700"></div>

### AFHQ Wild
<div><img src="images/afhqwild.jpg" alt="img" width="700"></div>

### AFHQ Dog
<div><img src="images/afhqdog.jpg" alt="img" width="700"></div>

### AFHQ Cat
<div><img src="images/afhqcat.jpg" alt="img" width="700"></div>

### LSUN Cat
<div><img src="images/cat.jpg" alt="img" width="700"></div>

### LSUN Horse
<div><img src="images/horse.jpg" alt="img" width="700"></div>

### LSUN Car
<div><img src="images/car.jpg" alt="img" width="700"></div>

### BreCaHAD
<div><img src="images/brecahad.jpg" alt="img" width="700"></div>

### CIFAR-10
<div><img src="images/cifar10.jpg" alt="img" width="700"></div>

### LSUN Church
<div><img src="images/church.jpg" alt="img" width="700"></div>


<a name="license"></a>
## 5. License
<a href="https://nvlabs.github.io/stylegan2/license.html">Nvidia Source Code License-NC</a>


