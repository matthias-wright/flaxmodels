## Documentation

##### Table of Contents  
* [1. Checkpoints](#ckpts)
  * [1.1 Preprocessing](#prepro)
  * [1.2 Structure](#structure)
* [2. Testing](#tests)


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



