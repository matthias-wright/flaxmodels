# Better Language Models and Their Implications (GPT2)

  
<b>Paper:</b> <a href="https://openai.com/blog/better-language-models/">https://openai.com/blog/better-language-models/</a>  
<b>Repository:</b> <a href="https://github.com/huggingface/transformers/tree/master/src/transformers/models/gpt2">https://github.com/huggingface/transformers/tree/master/src/transformers/models/gpt2</a>


##### Table of Contents
* [1. Models](#models)
* [2. Basic Usage](#usage)
* [3. Documentation](#documentation)
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


<a name="usage"></a>
## 2. Basic Usage
For more usage examples check out this [Colab](gpt2_demo.ipynb).

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
    context = jnp.expand_dims(token, axis=0)
    # Add token to sequence
    generated += [token]
    # Update past keys and values
    past = output['past_key_values']

# Decode sequence of tokens
sequence = tokenizer.decode(generated)
print(sequence)
```

<a name="documentation"></a>
## 3. Documentation
The documentation can be found [here](../../docs/Documentation.md#gpt2).

<a name="ack"></a>
## 4. Acknowledgments
The tokenizer is taken from <a href="https://huggingface.co/transformers/model_doc/gpt2.html#gpt2tokenizer">Huggingface</a>.

<a name="license"></a>
## 5. License
<a href="https://www.apache.org/licenses/LICENSE-2.0">Apache-2.0 License</a>


