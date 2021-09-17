# VGG Training in Jax/Flax
This is the training code for the [Jax/Flax implementation](https://github.com/matthias-wright/flaxmodels/tree/main/flaxmodels/vgg) of [VGG](https://arxiv.org/abs/1409.1556).

##### Table of Contents
* [Getting Started](#getting_started)
* [Training](#training)
* [Options](#options)
* [References](#references)
* [License](#license)


<a name="getting_started"></a>
## Getting Started
You will need Python 3.7 or later.
 
1. Clone the repository:
   ```sh
   > git clone https://github.com/matthias-wright/flaxmodels.git
   ```
2. Go into the directory:
   ```sh
   > cd flaxmodels/training/vgg
   ```
3. Install <a href="https://github.com/google/jax#installation">Jax</a> with CUDA.
4. Install requirements: 
   ```sh
   > pip install -r requirements.txt
   ```

<a name="training"></a>
## Training

### Basic Training
```python
CUDA_VISIBLE_DEVICES=0 python main.py
```

### Multi GPU Training
The script will automatically use all the visible GPUs for distributed training.
```python
CUDA_VISIBLE_DEVICES=0,1 python main.py
```

### Mixed-Precision Training
```python
CUDA_VISIBLE_DEVICES=0,1 python main.py --mixed_precision
```

<a name="options"></a>
## Options
* `--work_dir` - Path to directory for logging and checkpoints (str).
* `--data_dir` - Path for storing the dataset (str).
* `--name` - Name of the training run (str).
* `--group` - Group name of the training run (str).
* `--arch` - Architecture (str). Options: vgg16, vgg19.
* `--resume` - Resume training from best checkpoint (bool).
* `--num_epochs` - Number of epochs (int).
* `--learning_rate` - Learning rate (float).
* `--warmup_epochs` - Number of warmup epochs with lower learning rate (int).
* `--batch_size` - Batch size (int).
* `--num_classes` - Number of classes (int).
* `--img_size` - Image size (int).
* `--img_channels` - Number of image channels (int).
* `--mixed_precision` - Use mixed precision training (bool).
* `--random_seed` - Random seed (int).
* `--wandb` - Use Weights&Biases for logging (bool).
* `--log_every` - Log every log_every steps (int).

<a name="references"></a>
## References
* [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
* [pytorch/vision/torchvision/models/vgg.py](https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py)
* [google/flax/examples](https://github.com/google/flax/tree/main/examples)


<a name="license"></a>
## License
[MIT License](https://opensource.org/licenses/MIT)

