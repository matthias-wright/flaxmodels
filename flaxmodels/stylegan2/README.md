# Analyzing and Improving the Image Quality of StyleGAN

<div align="center"><img src="images/title.jpg" alt="img" width="1050"></div>
  
<b>Paper:</b> <a href="https://arxiv.org/abs/1912.04958">https://arxiv.org/abs/1912.04958</a>  
<b>Repository:</b> <a href="https://github.com/NVlabs/stylegan2">https://github.com/NVlabs/stylegan2</a> and <a href="https://github.com/NVlabs/stylegan2-ada">https://github.com/NVlabs/stylegan2-ada</a>

##### Table of Contents  
* [1. Basic Usage](#usage)
* [2. Documentation](#documentation)
  * [2.1 Generator](#doc_generator)
  * [2.2 SynthesisNetwork](#doc_syn)
  * [2.3 MappingNetwork](#doc_map)
  * [2.4 Discriminator](#doc_discriminator)
* [3. Models](#models)
  * [3.1 Metfaces](#metfaces)
  * [3.2 FFHQ](#ffhq)
  * [3.3 AFHQ Wild](#afhqwild)
  * [3.4 AFHQ Dog](#afhqdog)
  * [3.5 AFHQ Cat](#afhqcat)
  * [3.6 LSUN Cat](#cat)
  * [3.7 LSUN Horse](#horse)
  * [3.8 LSUN Car](#car)
  * [3.9 BreCaHAD](#brecahad)
  * [3.10 CIFAR-10](#cifar10)
  * [3.11 LSUN Church](#church)
* [4. License](#license)


<a name="usage"></a>
## 1. Basic Usage
For more usage examples check out this [Colab](https://colab.research.google.com/drive/1klNP4LbrXK5P3KwFM9_PqCVx5MwwilCI?usp=sharing).

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
images = generator.apply(params, z)

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

<a name="models"></a>
## 3. Models

<a name="metfaces"></a>
### 3.1 Metfaces
<div><img src="images/metfaces.jpg" alt="img" width="700"></div>

<a name="ffhq"></a>
### 3.2 FFHQ
<div><img src="images/ffhq.jpg" alt="img" width="700"></div>

<a name="afhqwild"></a>
### 3.3 AFHQ Wild
<div><img src="images/afhqwild.jpg" alt="img" width="700"></div>

<a name="afhqdog"></a>
### 3.4 AFHQ Dog
<div><img src="images/afhqdog.jpg" alt="img" width="700"></div>

<a name="afhqcat"></a>
### 3.5 AFHQ Cat
<div><img src="images/afhqcat.jpg" alt="img" width="700"></div>

<a name="cat"></a>
### 3.6 LSUN Cat
<div><img src="images/cat.jpg" alt="img" width="700"></div>

<a name="horse"></a>
### 3.7 LSUN Horse
<div><img src="images/horse.jpg" alt="img" width="700"></div>

<a name="car"></a>
### 3.8 LSUN Car
<div><img src="images/car.jpg" alt="img" width="700"></div>

<a name="brecahad"></a>
### 3.9 BreCaHAD
<div><img src="images/brecahad.jpg" alt="img" width="700"></div>

<a name="cifar10"></a>
### 3.10 CIFAR-10
<div><img src="images/cifar10.jpg" alt="img" width="700"></div>

<a name="church"></a>
### 3.11 LSUN Church
<div><img src="images/church.jpg" alt="img" width="700"></div>



<a name="license"></a>
## 4. License
<a href="https://nvlabs.github.io/stylegan2/license.html">Nvidia Source Code License-NC</a>


