# Better Language Models and Their Implications (GPT2)

  
<b>Paper:</b> <a href="https://openai.com/blog/better-language-models/">https://openai.com/blog/better-language-models/</a>  
<b>Repository:</b> <a href="https://github.com/huggingface/transformers/tree/master/src/transformers/models/gpt2">https://github.com/huggingface/transformers/tree/master/src/transformers/models/gpt2</a>


##### Table of Contents
* [1. Models](#models)
* [2. Example usages](#usages)
  * [2.1 Generate text](#gen_img_wo_trunc)
  * [2.2 Get language model head output from text input](#output_lm_head_text)
  * [2.3 Get language model head output from embeddings](#output_lm_head_embd)
  * [2.4 Get model output from text input](#output_model_text)
  * [2.5 Get model output from embeddings](#output_model_embd)
* [3. Documentation](#documentation)
  * [3.1 GPT2LMHeadModel](#doc_lmhead)
  * [3.2 GPT2Model](#doc_model)
* [4. Acknowledgments](#ack)
* [5. License](#license)


<a name="models"></a>
## 1. Models

| Model  | Parameters | Size | URL |
| ------------- | ------------- | ------------- | ------------- |
| gpt2  | ~ 120 Million  | ~ 500 MB | <a href="https://huggingface.co/gpt2">https://huggingface.co/gpt2</a> |
| gpt2-medium  | ~ 350 Million  | ~ 1.5 GB | <a href="https://huggingface.co/gpt2-medium">https://huggingface.co/gpt2-medium</a> |
| gpt2-large  | ~ 800 Million  | ~ 3 GB | <a href="https://huggingface.co/gpt2-large">https://huggingface.co/gpt2-large</a> |
| gpt2-xl  | ~ 1.5 Billion | ~ 6 GB | <a href="https://huggingface.co/gpt2-xl">https://huggingface.co/gpt2-xl</a> |


<a name="usages"></a>
## 2. Example usages

<a name="gen_img_wo_trunc"></a>
### 2.1 Generate text
This is very simple greedy text generation. There are more sophisticated <a href="https://huggingface.co/blog/how-to-generate">methods</a> out there.
```python
import jax
import jax.numpy as jnp
import flaxmodels as fm

key = jax.random.PRNGKey(0)

# Initialize tokenizer
tokenizer = fm.gpt2.get_tokenizer()

# Encode start sequence
generated = tokenizer.encode('The Manhattan bridge')

context = jnp.array([generated])
past = None

# Initialize model
# Models to choose from ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
model = fm.gpt2.GPT2LMHeadModel(pretrained='gpt2')
params = model.init(key, input_ids=context, past_key_values=past)

for i in range(20):
    # Predict next token in sequence
    output = model.apply(params, input_ids=context, past_key_values=past, use_cache=True)
    token = jnp.argmax(output['logits'][..., -1, :])
    #context = jnp.expand_dims(token, axis=(0, 1))
    context = jnp.expand_dims(token, axis=0)
    # Add token to sequence
    generated += [token]
    # Update past keys and values
    past = output['past_key_values']

# Decode sequence of tokens
sequence = tokenizer.decode(generated)
print(sequence)
```


<a name="output_lm_head_text"></a>
### 2.2 Get language model head output from text input

```python
import jax
import jax.numpy as jnp
import flaxmodels as fm

key = jax.random.PRNGKey(0)

# Initialize tokenizer
tokenizer = fm.gpt2.get_tokenizer()

# Encode start sequence
input_ids = tokenizer.encode('The Manhattan bridge')
input_ids = jnp.array([input_ids])

# Initialize model
model = fm.gpt2.GPT2LMHeadModel(pretrained='gpt2')
params = model.init(key, input_ids=input_ids)

# Compute output
output = model.apply(params, input_ids=input_ids, use_cache=True)
# output: {'last_hidden_state': ..., 'past_key_values': ..., 'loss': ..., 'logits': ...}
```


<a name="output_lm_head_embd"></a>
### 2.3 Get language model head output from embeddings

```python
import jax
import jax.numpy as jnp
import flaxmodels as fm
                                                                    
key = jax.random.PRNGKey(0)

# Dummy input                                        
input_embds = jax.random.normal(key, shape=(2, 10, 768))

# Initialize model
model = fm.gpt2.GPT2LMHeadModel(pretrained='gpt2')
params = model.init(key, input_embds=input_embds)
# Compute output
output = model.apply(params, input_embds=input_embds, use_cache=True)
# output: {'last_hidden_state': ..., 'past_key_values': ..., 'loss': ..., 'logits': ...}
```


<a name="output_model_text"></a>
### 2.4 Get model output from text input

```python
import jax
import jax.numpy as jnp
import flaxmodels as fm

key = jax.random.PRNGKey(0)

# Initialize tokenizer
tokenizer = fm.gpt2.get_tokenizer()

# Encode start sequence
input_ids = tokenizer.encode('The Manhattan bridge')
input_ids = jnp.array([input_ids])

# Initialize model
model = fm.gpt2.GPT2Model(pretrained='gpt2')
params = model.init(key, input_ids=input_ids)

# Compute output
output = model.apply(params, input_ids=input_ids, use_cache=True)
# output: {'last_hidden_state': ..., 'past_key_values': ...}
```


<a name="output_model_embd"></a>
### 2.5 Get model output from embeddings

```python
import jax
import jax.numpy as jnp
import flaxmodels as fm
                                                                    
key = jax.random.PRNGKey(0)

# Dummy input
input_embds = jax.random.normal(key, shape=(2, 10, 768))
                                                                                                      
# Initialize model
model = fm.gpt2.GPT2Model(pretrained='gpt2')
params = model.init(key, input_embds=input_embds)

# Compute output
output = model.apply(params, input_embds=input_embds, use_cache=True)
# output: {'last_hidden_state': ..., 'past_key_values': ...}
```


<a name="documentation"></a>
## 3. Documentation

<a name="doc_lmhead"></a>
### 3.1 GPT2LMHeadModel
flaxmodels.gpt2.GPT2LMHeadModel(*config=None, pretrained=None, ckpt_dir=None, rng=jax.random.PRNGKey(0)*)


#### Parameters
* **config (types.SimpleNamespace)** - Configuration file.
* **pretrained (str)** - Which pretrained model to use, None for random initialization. Options:
  * 'gpt2'
  * 'gpt2-medium'
  * 'gpt2-large'
  * 'gpt2-xl'
* **ckpt_dir (str)** - The directory to which the pretrained weights are downloaded. Only relevant if a pretrained model is used. If this argument is None, the weights will be saved to a temp directory.
* **rng (jax.numpy.ndarray)** - Random seed.

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

<a name="doc_model"></a>
### 3.2 GPT2Model

flaxmodels.gpt2.GPT2Model(*config=None, pretrained=None, ckpt_dir=None, param_dict=None, rng=jax.random.PRNGKey(0)*)

#### Parameters
* **config (types.SimpleNamespace)** - Configuration file.
* **pretrained (str)** - Which pretrained model to use, None for random initialization. Options:
  * 'gpt2'
  * 'gpt2-medium'
  * 'gpt2-large'
  * 'gpt2-xl'
* **ckpt_dir (str)** - The directory to which the pretrained weights are downloaded. Only relevant if a pretrained model is used. If this argument is None, the weights will be saved to a temp directory.
* **param_dict (dict)** - Parameter dict with pretrained parameters. If not None, 'pretrained' will be ignored.
* **rng (jax.numpy.ndarray)** - Random seed.

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

<a name="ack"></a>
### 4. Acknowledgments
The tokenizer is taken from <a href="https://huggingface.co/transformers/model_doc/gpt2.html#gpt2tokenizer">Huggingface</a>.

<a name="license"></a>
## 5. License
<a href="https://www.apache.org/licenses/LICENSE-2.0">Apache-2.0 License</a>


