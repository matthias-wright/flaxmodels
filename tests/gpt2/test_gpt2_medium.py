import jax
import jax.numpy as jnp
import flaxmodels as fm


def test_reference_output_lm_head_input_ids():
    input_ids = jnp.load(f'tests/gpt2/aux_files/gpt2-medium/gpt2-medium_lmhead_input_ids_input.npy')
    labels = jnp.load(f'tests/gpt2/aux_files/gpt2-medium/gpt2-medium_lmhead_input_ids_labels.npy')
    
    ref_logits = jnp.load(f'tests/gpt2/aux_files/gpt2-medium/gpt2-medium_lmhead_input_ids_logits_ref.npy') 
    ref_loss = jnp.load(f'tests/gpt2/aux_files/gpt2-medium/gpt2-medium_lmhead_input_ids_loss_ref.npy')

    key = jax.random.PRNGKey(0)
    model = fm.gpt2.GPT2LMHeadModel(pretrained='gpt2-medium')

    params = model.init(key, input_ids=input_ids, labels=labels)
    output = model.apply(params, input_ids=input_ids, labels=labels)
    logits, loss = output['logits'], output['loss']

    diff_logits = jnp.mean(jnp.abs(ref_logits - logits))
    diff_loss = jnp.mean(jnp.abs(ref_loss - loss))

    assert diff_logits < 1e-3 and diff_loss < 1e-3


def test_reference_output_lm_head_input_embds():
    input_embds = jnp.load(f'tests/gpt2/aux_files/gpt2-medium/gpt2-medium_lmhead_input_embds_input.npy')
    labels = jnp.load(f'tests/gpt2/aux_files/gpt2-medium/gpt2-medium_lmhead_input_embds_labels.npy')

    ref_logits = jnp.load(f'tests/gpt2/aux_files/gpt2-medium/gpt2-medium_lmhead_input_embds_logits_ref.npy') 
    ref_loss = jnp.load(f'tests/gpt2/aux_files/gpt2-medium/gpt2-medium_lmhead_input_embds_loss_ref.npy')

    key = jax.random.PRNGKey(0)
    model = fm.gpt2.GPT2LMHeadModel(pretrained='gpt2-medium')

    params = model.init(key, input_embds=input_embds, labels=labels)
    output = model.apply(params, input_embds=input_embds, labels=labels)
    logits, loss = output['logits'], output['loss']

    diff_logits = jnp.mean(jnp.abs(ref_logits - logits))
    diff_loss = jnp.mean(jnp.abs(ref_loss - loss))

    assert diff_logits < 1e-3 and diff_loss < 1e-3


def test_reference_output_model_input_ids():
    input_ids = jnp.load(f'tests/gpt2/aux_files/gpt2-medium/gpt2-medium_model_input_ids_input.npy')

    ref_output = jnp.load(f'tests/gpt2/aux_files/gpt2-medium/gpt2-medium_model_input_ids_output_ref.npy') 

    key = jax.random.PRNGKey(0)
    model = fm.gpt2.GPT2Model(pretrained='gpt2-medium')

    params = model.init(key, input_ids=input_ids)
    output = model.apply(params, input_ids=input_ids)

    diff = jnp.mean(jnp.abs(ref_output - output['last_hidden_state']))

    assert diff < 1e-4


def test_reference_output_model_input_embds():
    input_embds = jnp.load(f'tests/gpt2/aux_files/gpt2-medium/gpt2-medium_model_input_embds_input.npy')

    ref_output = jnp.load(f'tests/gpt2/aux_files/gpt2-medium/gpt2-medium_model_input_embds_output_ref.npy') 

    key = jax.random.PRNGKey(0)
    model = fm.gpt2.GPT2Model(pretrained='gpt2-medium')

    params = model.init(key, input_embds=input_embds)
    output = model.apply(params, input_embds=input_embds)

    diff = jnp.mean(jnp.abs(ref_output - output['last_hidden_state']))

    assert diff < 1e-4

