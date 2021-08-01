import jax
import jax.numpy as jnp
import numpy as np
import flaxmodels as fm
from PIL import Image


def test_reference_output_afhqcat():
    z = jnp.load('tests/stylegan2/generator/aux_files/afhqcat_input_z.npy')
    out_ref = jnp.load('tests/stylegan2/generator/aux_files/afhqcat_output_ref.npy')

    key = jax.random.PRNGKey(0)
    generator = fm.stylegan2.Generator(pretrained='afhqcat')
    params = generator.init(key, z)
    out = generator.apply(params, z, noise_mode='const', train=False)

    diff = jnp.mean(jnp.abs(out - out_ref))
    
    assert diff < 2e-2


def test_reference_output_afhqdog():
    z = jnp.load('tests/stylegan2/generator/aux_files/afhqdog_input_z.npy')
    out_ref = jnp.load('tests/stylegan2/generator/aux_files/afhqdog_output_ref.npy')

    key = jax.random.PRNGKey(0)
    generator = fm.stylegan2.Generator(pretrained='afhqdog')
    params = generator.init(key, z)
    out = generator.apply(params, z, noise_mode='const', train=False)

    diff = jnp.mean(jnp.abs(out - out_ref))
    
    assert diff < 5e-3


def test_reference_output_afhqwild():
    z = jnp.load('tests/stylegan2/generator/aux_files/afhqwild_input_z.npy')
    out_ref = jnp.load('tests/stylegan2/generator/aux_files/afhqwild_output_ref.npy')

    key = jax.random.PRNGKey(0)
    generator = fm.stylegan2.Generator(pretrained='afhqwild')
    params = generator.init(key, z)
    out = generator.apply(params, z, noise_mode='const', train=False)

    diff = jnp.mean(jnp.abs(out - out_ref))
    
    assert diff < 3e-3


def test_reference_output_brecahad():
    z = jnp.load('tests/stylegan2/generator/aux_files/brecahad_input_z.npy')
    out_ref = jnp.load('tests/stylegan2/generator/aux_files/brecahad_output_ref.npy')

    key = jax.random.PRNGKey(0)
    generator = fm.stylegan2.Generator(pretrained='brecahad')
    params = generator.init(key, z)
    out = generator.apply(params, z, noise_mode='const', train=False)

    diff = jnp.mean(jnp.abs(out - out_ref))
    
    assert diff < 3e-3


def test_reference_output_car():
    z = jnp.load('tests/stylegan2/generator/aux_files/car_input_z.npy')
    out_ref = jnp.load('tests/stylegan2/generator/aux_files/car_output_ref.npy')

    key = jax.random.PRNGKey(0)
    generator = fm.stylegan2.Generator(pretrained='car')
    params = generator.init(key, z)
    out = generator.apply(params, z, noise_mode='const', train=False)

    diff = jnp.mean(jnp.abs(out - out_ref))
    
    assert diff < 1e-5


def test_reference_output_cat():
    z = jnp.load('tests/stylegan2/generator/aux_files/cat_input_z.npy')
    out_ref = jnp.load('tests/stylegan2/generator/aux_files/cat_output_ref.npy')

    key = jax.random.PRNGKey(0)
    generator = fm.stylegan2.Generator(pretrained='cat')
    params = generator.init(key, z)
    out = generator.apply(params, z, noise_mode='const', train=False)

    diff = jnp.mean(jnp.abs(out - out_ref))
    
    assert diff < 1e-5


def test_reference_output_church():
    z = jnp.load('tests/stylegan2/generator/aux_files/church_input_z.npy')
    out_ref = jnp.load('tests/stylegan2/generator/aux_files/church_output_ref.npy')

    key = jax.random.PRNGKey(0)
    generator = fm.stylegan2.Generator(pretrained='church')
    params = generator.init(key, z)
    out = generator.apply(params, z, noise_mode='const', train=False)

    diff = jnp.mean(jnp.abs(out - out_ref))
    
    assert diff < 1e-5


def test_reference_output_cifar10():
    z = jnp.load('tests/stylegan2/generator/aux_files/cifar10_input_z.npy')
    label = jnp.load('tests/stylegan2/generator/aux_files/cifar10_input_label.npy')
    out_ref = jnp.load('tests/stylegan2/generator/aux_files/cifar10_output_ref.npy')

    key = jax.random.PRNGKey(0)
    generator = fm.stylegan2.Generator(pretrained='cifar10')
    params = generator.init(key, z, label)
    out = generator.apply(params, z, label, noise_mode='const', train=False)

    diff = jnp.mean(jnp.abs(out - out_ref))
    
    assert diff < 3e-3


def test_reference_output_ffhq():
    z = jnp.load('tests/stylegan2/generator/aux_files/ffhq_input_z.npy')
    out_ref = jnp.load('tests/stylegan2/generator/aux_files/ffhq_output_ref.npy')

    key = jax.random.PRNGKey(0)
    generator = fm.stylegan2.Generator(pretrained='ffhq')
    params = generator.init(key, z)
    out = generator.apply(params, z, noise_mode='const', train=False)

    diff = jnp.mean(jnp.abs(out - out_ref))
    
    assert diff < 1e-5


def test_reference_output_horse():
    z = jnp.load('tests/stylegan2/generator/aux_files/horse_input_z.npy')
    out_ref = jnp.load('tests/stylegan2/generator/aux_files/horse_output_ref.npy')

    key = jax.random.PRNGKey(0)
    generator = fm.stylegan2.Generator(pretrained='horse')
    params = generator.init(key, z)
    out = generator.apply(params, z, noise_mode='const', train=False)

    diff = jnp.mean(jnp.abs(out - out_ref))
    
    assert diff < 1e-5


def test_reference_output_metfaces():
    z = jnp.load('tests/stylegan2/generator/aux_files/metfaces_input_z.npy')
    out_ref = jnp.load('tests/stylegan2/generator/aux_files/metfaces_output_ref.npy')

    key = jax.random.PRNGKey(0)
    generator = fm.stylegan2.Generator(pretrained='metfaces')
    params = generator.init(key, z)
    out = generator.apply(params, z, noise_mode='const', train=False)

    diff = jnp.mean(jnp.abs(out - out_ref))
    
    assert diff < 3e-4


