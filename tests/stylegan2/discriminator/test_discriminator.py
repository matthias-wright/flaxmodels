import jax
import jax.numpy as jnp
import numpy as np
import flaxmodels as fm
from PIL import Image


def test_reference_output_afhqcat():
    img = jnp.load('tests/stylegan2/discriminator/aux_files/afhqcat_input_img.npy')
    out_ref = jnp.load('tests/stylegan2/discriminator/aux_files/afhqcat_output_ref.npy')

    key = jax.random.PRNGKey(0)
    discriminator = fm.stylegan2.Discriminator(pretrained='afhqcat')
    params = discriminator.init(key, img)
    out = discriminator.apply(params, img)

    diff = jnp.mean(jnp.abs(out - out_ref))
    
    assert diff < 5e-4


def test_reference_output_afhqdog():
    img = jnp.load('tests/stylegan2/discriminator/aux_files/afhqdog_input_img.npy')
    out_ref = jnp.load('tests/stylegan2/discriminator/aux_files/afhqdog_output_ref.npy')

    key = jax.random.PRNGKey(0)
    discriminator = fm.stylegan2.Discriminator(pretrained='afhqdog')
    params = discriminator.init(key, img)
    out = discriminator.apply(params, img)

    diff = jnp.mean(jnp.abs(out - out_ref))
    
    assert diff < 4e-3


def test_reference_output_afhqwild():
    img = jnp.load('tests/stylegan2/discriminator/aux_files/afhqwild_input_img.npy')
    out_ref = jnp.load('tests/stylegan2/discriminator/aux_files/afhqwild_output_ref.npy')

    key = jax.random.PRNGKey(0)
    discriminator = fm.stylegan2.Discriminator(pretrained='afhqwild')
    params = discriminator.init(key, img)
    out = discriminator.apply(params, img)

    diff = jnp.mean(jnp.abs(out - out_ref))
    
    assert diff < 4e-4


def test_reference_output_brecahad():
    img = jnp.load('tests/stylegan2/discriminator/aux_files/brecahad_input_img.npy')
    out_ref = jnp.load('tests/stylegan2/discriminator/aux_files/brecahad_output_ref.npy')

    key = jax.random.PRNGKey(0)
    discriminator = fm.stylegan2.Discriminator(pretrained='brecahad')
    params = discriminator.init(key, img)
    out = discriminator.apply(params, img)

    diff = jnp.mean(jnp.abs(out - out_ref))
    
    assert diff < 2e-4


def test_reference_output_car():
    img = jnp.load('tests/stylegan2/discriminator/aux_files/car_input_img.npy')
    out_ref = jnp.load('tests/stylegan2/discriminator/aux_files/car_output_ref.npy')

    key = jax.random.PRNGKey(0)
    discriminator = fm.stylegan2.Discriminator(pretrained='car')
    params = discriminator.init(key, img)
    out = discriminator.apply(params, img)

    diff = jnp.mean(jnp.abs(out - out_ref))
    
    assert diff < 3e-4


def test_reference_output_cat():
    img = jnp.load('tests/stylegan2/discriminator/aux_files/cat_input_img.npy')
    out_ref = jnp.load('tests/stylegan2/discriminator/aux_files/cat_output_ref.npy')

    key = jax.random.PRNGKey(0)
    discriminator = fm.stylegan2.Discriminator(pretrained='cat')
    params = discriminator.init(key, img)
    out = discriminator.apply(params, img)

    diff = jnp.mean(jnp.abs(out - out_ref))
    
    assert diff < 1e-4


def test_reference_output_church():
    img = jnp.load('tests/stylegan2/discriminator/aux_files/church_input_img.npy')
    out_ref = jnp.load('tests/stylegan2/discriminator/aux_files/church_output_ref.npy')

    key = jax.random.PRNGKey(0)
    discriminator = fm.stylegan2.Discriminator(pretrained='church')
    params = discriminator.init(key, img)
    out = discriminator.apply(params, img)

    diff = jnp.mean(jnp.abs(out - out_ref))
    
    assert diff < 1e-4


def test_reference_output_cifar10():
    img = jnp.load('tests/stylegan2/discriminator/aux_files/cifar10_input_img.npy')
    label = jnp.load('tests/stylegan2/discriminator/aux_files/cifar10_input_label.npy')
    out_ref = jnp.load('tests/stylegan2/discriminator/aux_files/cifar10_output_ref.npy')

    key = jax.random.PRNGKey(0)
    discriminator = fm.stylegan2.Discriminator(pretrained='cifar10')
    params = discriminator.init(key, img, label)
    out = discriminator.apply(params, img, label)

    diff = jnp.mean(jnp.abs(out - out_ref))
    
    assert diff < 1e-2


def test_reference_output_ffhq():
    img = jnp.load('tests/stylegan2/discriminator/aux_files/ffhq_input_img.npy')
    out_ref = jnp.load('tests/stylegan2/discriminator/aux_files/ffhq_output_ref.npy')

    key = jax.random.PRNGKey(0)
    discriminator = fm.stylegan2.Discriminator(pretrained='ffhq')
    params = discriminator.init(key, img)
    out = discriminator.apply(params, img)

    diff = jnp.mean(jnp.abs(out - out_ref))
    
    assert diff < 1e-4


def test_reference_output_horse():
    img = jnp.load('tests/stylegan2/discriminator/aux_files/horse_input_img.npy')
    out_ref = jnp.load('tests/stylegan2/discriminator/aux_files/horse_output_ref.npy')

    key = jax.random.PRNGKey(0)
    discriminator = fm.stylegan2.Discriminator(pretrained='horse')
    params = discriminator.init(key, img)
    out = discriminator.apply(params, img)

    diff = jnp.mean(jnp.abs(out - out_ref))
    
    assert diff < 1e-4


def test_reference_output_metfaces():
    img = jnp.load('tests/stylegan2/discriminator/aux_files/metfaces_input_img.npy')
    out_ref = jnp.load('tests/stylegan2/discriminator/aux_files/metfaces_output_ref.npy')

    key = jax.random.PRNGKey(0)
    discriminator = fm.stylegan2.Discriminator(pretrained='metfaces')
    params = discriminator.init(key, img)
    out = discriminator.apply(params, img)

    diff = jnp.mean(jnp.abs(out - out_ref))
    
    assert diff < 2e-3


