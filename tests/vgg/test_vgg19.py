import jax
import jax.numpy as jnp
import numpy as np
import flaxmodels as fm
from PIL import Image


def test_output_softmax():
    # If output='softmax', the output should be in range [0, 1]
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(1, 224, 224, 3), minval=0, maxval=1)

    vgg19 = fm.VGG19(output='softmax', pretrained=None)
    init_rngs = {'params': jax.random.PRNGKey(1), 'dropout': jax.random.PRNGKey(2)}
    params = vgg19.init(init_rngs, x)
    out = vgg19.apply(params, x, train=False)

    assert jnp.min(out) >= 0.0 and jnp.max(out) <= 1.0


def test_output_activations():
    # If output='activations', the output should be a dict.
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(1, 224, 224, 3), minval=0, maxval=1)

    vgg19 = fm.VGG19(output='activations', pretrained=None)
    init_rngs = {'params': jax.random.PRNGKey(1), 'dropout': jax.random.PRNGKey(2)}
    params = vgg19.init(init_rngs, x)
    out = vgg19.apply(params, x, train=False)

    assert isinstance(out, dict) 


def test_include_head_true():
    # If include_head=True, the output should be a tensor of shape [B, 1000].
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(1, 224, 224, 3), minval=0, maxval=1)

    vgg19 = fm.VGG19(include_head=True, pretrained=None)
    init_rngs = {'params': jax.random.PRNGKey(1), 'dropout': jax.random.PRNGKey(2)}
    params = vgg19.init(init_rngs, x)
    out = vgg19.apply(params, x, train=False)

    assert hasattr(out, 'shape') and len(out.shape) == 2 and out.shape[0] == 1 and out.shape[1] == 1000


def test_include_head_false():
    # If include_head=False, the output should be a tensor of shape [B, *, *, 512].
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(1, 224, 224, 3), minval=0, maxval=1)

    vgg19 = fm.VGG19(include_head=False, pretrained=None)
    init_rngs = {'params': jax.random.PRNGKey(1), 'dropout': jax.random.PRNGKey(2)}
    params = vgg19.init(init_rngs, x)
    out = vgg19.apply(params, x, train=False)

    assert hasattr(out, 'shape') and len(out.shape) == 4 and out.shape[0] == 1 and out.shape[-1] == 512


def test_reference_output():
    key = jax.random.PRNGKey(0)
    img = Image.open('tests/aux_files/elefant.jpg')
    img = img.resize((224, 224))
    x = jnp.array(img, dtype=jnp.float32) / 255.0
    x = jnp.expand_dims(x, axis=0)

    vgg19 = fm.VGG19(output='logits', pretrained='imagenet')
    init_rngs = {'params': jax.random.PRNGKey(1), 'dropout': jax.random.PRNGKey(2)}
    params = vgg19.init(init_rngs, x)
    out = vgg19.apply(params, x, train=False)
    
    out_ref = jnp.load('tests/vgg/aux_files/vgg19_elefant_output_ref.npy')
    diff = jnp.mean(jnp.abs(out - out_ref))

    assert diff < 1e-5

