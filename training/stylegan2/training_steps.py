import jax
import jax.numpy as jnp
import functools


def main_step_G(state_G, state_D, batch, z_latent1, z_latent2, metrics, cutoff, rng):

    def loss_fn(params):
        w_latent1, new_state_G = state_G.apply_mapping({'params': params['mapping'], 'moving_stats': state_G.moving_stats},
                                                       z_latent1,
                                                       batch['label'],
                                                       mutable=['moving_stats'])
        w_latent2 = state_G.apply_mapping({'params': params['mapping'], 'moving_stats': state_G.moving_stats},
                                          z_latent2,
                                          batch['label'],
                                          skip_w_avg_update=True)
        
        w_latent = jax.ops.index_update(w_latent1, jax.ops.index[:, cutoff:], w_latent2[:, cutoff:])

        image_gen = state_G.apply_synthesis({'params': params['synthesis'], 'noise_consts': state_G.noise_consts},
                                            w_latent,
                                            rng=rng)

        fake_logits = state_D.apply_fn(state_D.params, image_gen, batch['label'])
        loss = jnp.mean(jax.nn.softplus(-fake_logits)) 
        return loss, (fake_logits, image_gen, new_state_G)

    dynamic_scale = state_G.dynamic_scale_main

    if dynamic_scale:
        grad_fn = dynamic_scale.value_and_grad(loss_fn, has_aux=True, axis_name='batch')
        dynamic_scale, is_fin, aux, grads = grad_fn(state_G.params)
    else:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        aux, grads = grad_fn(state_G.params)
        grads = jax.lax.pmean(grads, axis_name='batch')

    loss = aux[0]
    _, image_gen, new_state = aux[1]
    metrics['G_loss'] = loss
    metrics['image_gen'] = image_gen

    new_state_G = state_G.apply_gradients(grads=grads, moving_stats=new_state['moving_stats'])
    
    if dynamic_scale:
        new_state_G = new_state_G.replace(opt_state=jax.tree_multimap(functools.partial(jnp.where, is_fin),
                                                                      new_state_G.opt_state,
                                                                      state_G.opt_state),
                                          params=jax.tree_multimap(functools.partial(jnp.where, is_fin),
                                                                   new_state_G.params,
                                                                   state_G.params))
        metrics['G_scale'] = dynamic_scale.scale

    return new_state_G, metrics


def regul_step_G(state_G, batch, z_latent, pl_noise, pl_mean, metrics, config, rng):

    def loss_fn(params):
        w_latent, new_state_G = state_G.apply_mapping({'params': params['mapping'], 'moving_stats': state_G.moving_stats},
                                                      z_latent,
                                                      batch['label'],
                                                      mutable=['moving_stats'])
        
        pl_grads = jax.grad(lambda *args: jnp.sum(state_G.apply_synthesis(*args) * pl_noise), argnums=1)({'params': params['synthesis'],
                                                                                                          'noise_consts': state_G.noise_consts},
                                                                                                          w_latent,
                                                                                                          'random',
                                                                                                          rng)
        pl_lengths = jnp.sqrt(jnp.mean(jnp.sum(jnp.square(pl_grads), axis=2), axis=1))
        pl_mean_new = pl_mean + config.pl_decay * (jnp.mean(pl_lengths) - pl_mean)
        pl_penalty = jnp.square(pl_lengths - pl_mean_new) * config.pl_weight
        loss = jnp.mean(pl_penalty) * config.G_reg_interval

        return loss, pl_mean_new

    dynamic_scale = state_G.dynamic_scale_reg

    if dynamic_scale:
        grad_fn = dynamic_scale.value_and_grad(loss_fn, has_aux=True)
        dynamic_scale, is_fin, aux, grads = grad_fn(state_G.params)
    else:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        aux, grads = grad_fn(state_G.params)
        grads = jax.lax.pmean(grads, axis_name='batch')

    loss = aux[0]
    pl_mean_new = aux[1]

    metrics['G_regul_loss'] = loss
    new_state_G = state_G.apply_gradients(grads=grads)
    
    if dynamic_scale:
        new_state_G = new_state_G.replace(opt_state=jax.tree_multimap(functools.partial(jnp.where, is_fin),
                                                                      new_state_G.opt_state,
                                                                      state_G.opt_state),
                                          params=jax.tree_multimap(functools.partial(jnp.where, is_fin),
                                                                   new_state_G.params,
                                                                   state_G.params))
        metrics['G_regul_scale'] = dynamic_scale.scale

    return new_state_G, metrics, pl_mean_new


def main_step_D(state_G, state_D, batch, z_latent1, z_latent2, metrics, cutoff, rng):

    def loss_fn(params):
        w_latent1 = state_G.apply_mapping({'params': state_G.params['mapping'], 'moving_stats': state_G.moving_stats},
                                         z_latent1,
                                         batch['label'],
                                         train=False)

        w_latent2 = state_G.apply_mapping({'params': state_G.params['mapping'], 'moving_stats': state_G.moving_stats},
                                          z_latent2,
                                          batch['label'],
                                          train=False)
        
        w_latent = jax.ops.index_update(w_latent1, jax.ops.index[:, cutoff:], w_latent2[:, cutoff:])

        image_gen = state_G.apply_synthesis({'params': state_G.params['synthesis'], 'noise_consts': state_G.noise_consts},
                                            w_latent,
                                            rng=rng)

        fake_logits = state_D.apply_fn(params, image_gen, batch['label'])
        real_logits = state_D.apply_fn(params, batch['image'], batch['label'])

        loss_fake = jax.nn.softplus(fake_logits)
        loss_real = jax.nn.softplus(-real_logits)
        loss = jnp.mean(loss_fake + loss_real)
        
        return loss, (fake_logits, real_logits)

    dynamic_scale = state_D.dynamic_scale_main

    if dynamic_scale:
        grad_fn = dynamic_scale.value_and_grad(loss_fn, has_aux=True)
        dynamic_scale, is_fin, aux, grads = grad_fn(state_D.params)
    else:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        aux, grads = grad_fn(state_D.params)
        grads = jax.lax.pmean(grads, axis_name='batch')

    loss = aux[0]
    fake_logits, real_logits = aux[1]
    metrics['D_loss'] = loss
    metrics['fake_logits'] = jnp.mean(fake_logits)
    metrics['real_logits'] = jnp.mean(real_logits)

    new_state_D = state_D.apply_gradients(grads=grads)
    
    if dynamic_scale:
        new_state_D = new_state_D.replace(opt_state=jax.tree_multimap(functools.partial(jnp.where, is_fin),
                                                                      new_state_D.opt_state,
                                                                      state_D.opt_state),
                                          params=jax.tree_multimap(functools.partial(jnp.where, is_fin),
                                                                   new_state_D.params,
                                                                   state_D.params))
        metrics['D_scale'] = dynamic_scale.scale

    return new_state_D, metrics


def regul_step_D(state_D, batch, metrics, config):

    def loss_fn(params):
        r1_grads = jax.grad(lambda *args: jnp.sum(state_D.apply_fn(*args)), argnums=1)(params, batch['image'], batch['label'])
        r1_penalty = jnp.sum(jnp.square(r1_grads), axis=(1, 2, 3)) * (config.r1_gamma / 2) * config.D_reg_interval
        loss = jnp.mean(r1_penalty)
        return loss, None

    dynamic_scale = state_D.dynamic_scale_reg

    if dynamic_scale:
        grad_fn = dynamic_scale.value_and_grad(loss_fn, has_aux=True)
        dynamic_scale, is_fin, aux, grads = grad_fn(state_D.params)
    else:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        aux, grads = grad_fn(state_D.params)
        grads = jax.lax.pmean(grads, axis_name='batch')

    loss = aux[0]
    metrics['D_regul_loss'] = loss

    new_state_D = state_D.apply_gradients(grads=grads)
    
    if dynamic_scale:
        new_state_D = new_state_D.replace(opt_state=jax.tree_multimap(functools.partial(jnp.where, is_fin),
                                                                      new_state_D.opt_state,
                                                                      state_D.opt_state),
                                          params=jax.tree_multimap(functools.partial(jnp.where, is_fin),
                                                                   new_state_D.params,
                                                                   state_D.params))
        metrics['D_regul_scale'] = dynamic_scale.scale

    return new_state_D, metrics


def eval_step_G(generator, params, z_latent, labels, truncation):
    image_gen = generator.apply(params, z_latent, labels, truncation_psi=truncation, train=False, noise_mode='const')
    return image_gen

