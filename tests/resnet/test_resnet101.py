import jax
import jax.numpy as jnp
import numpy as np
import flaxmodels as fm
from PIL import Image


def test_output_softmax():
    # If output='softmax', the output should be in range [0, 1]
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(1, 224, 224, 3), minval=0, maxval=1)

    resnet101 = fm.ResNet101(output='softmax', pretrained=None)
    params = resnet101.init(key, x)
    out = resnet101.apply(params, x, train=False)

    assert jnp.min(out) >= 0.0 and jnp.max(out) <= 1.0


def test_output_activations():
    # If output='activations', the output should be a dict.
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(1, 224, 224, 3), minval=0, maxval=1)

    resnet101 = fm.ResNet101(output='activations', pretrained=None)
    params = resnet101.init(key, x)
    out = resnet101.apply(params, x, train=False)

    assert isinstance(out, dict)


def test_reference_output():
    key = jax.random.PRNGKey(0)
    img = Image.open('tests/aux_files/elefant.jpg')
    img = img.resize((224, 224))
    x = jnp.array(img, dtype=jnp.float32) / 255.0
    x = jnp.expand_dims(x, axis=0)

    resnet101 = fm.ResNet101(output='logits', pretrained='imagenet')
    params = resnet101.init(key, x)
    out = resnet101.apply(params, x, train=False)
    
    out_ref = jnp.load('tests/resnet/aux_files/resnet101_elefant_output_ref.npy')
    diff = jnp.mean(jnp.abs(out - out_ref))

    assert diff < 1e-5
