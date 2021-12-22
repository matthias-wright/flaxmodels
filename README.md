<div align="center"><img src="https://raw.githubusercontent.com/matthias-wright/flaxmodels/main/docs/img/flax.png" alt="flax" width="200" height="200"></div>
<div align="center"><h3>Flax Models</h3></div>
<div align="center">A collection of pretrained models in <a href="https://github.com/google/flax">Flax</a>.</div>

</br>

<!-- ABOUT -->
### About
The goal of this project is to make current deep learning models more easily available for the awesome <a href="https://github.com/google/jax">Jax</a>/<a href="https://github.com/google/flax">Flax</a> ecosystem.

### Models
* GPT2 [[model](flaxmodels/gpt2)]  
* StyleGAN2 [[model](flaxmodels/stylegan2)] [[training](training/stylegan2)]  
* ResNet{18, 34, 50, 101, 152} [[model](flaxmodels/resnet)] [[training](training/resnet)]  
* VGG{16, 19} [[model](flaxmodels/vgg)] [[training](training/vgg)]  
* FewShotGanAdaption [[model](flaxmodels/few_shot_gan_adaption)] [[training](training/few_shot_gan_adaption)]  


### Installation
You will need Python 3.7 or later.
 
1. For GPU usage, follow the <a href="https://github.com/google/jax#installation">Jax</a> installation with CUDA.
2. Then install:
   ```sh
   > pip install --upgrade git+https://github.com/matthias-wright/flaxmodels.git
   ```
For CPU-only you can skip step 1.

### Documentation
The documentation for the models can be found [here](docs/Documentation.md#models).

### Checkpoints
The checkpoints are taken from the repositories that are referenced on the model pages. The processing steps and the format of the checkpoints are documented [here](docs/Documentation.md#1-checkpoints).

### Testing
To run the tests, pytest needs to be installed. 
```sh
> git clone https://github.com/matthias-wright/flaxmodels.git
> cd flaxmodels
> python -m pytest tests/
```
See [here](docs/Documentation.md#2-testing) for an explanation of the testing strategy.


### Acknowledgments
Thank you to the developers of Jax and Flax. The title image is a photograph of a flax flower, kindly made available by <a href="https://unsplash.com/@matyszczyk">Marta Matyszczyk</a>. 

### License
Each model has an individual license.
