import jax
import jax.numpy as jnp
import flax.linen as nn
import math
import json
from types import SimpleNamespace


#----------------------------------------------------------
# Linear
#----------------------------------------------------------
def linear(features, param_dict, bias=True):
    if param_dict is None:
        return nn.Dense(features=features, use_bias=bias)
    else:
        if bias:
            assert 'bias' in param_dict
            assert 'weight' in param_dict
            return nn.Dense(features=features,
                            kernel_init=lambda *_ : jnp.array(param_dict['weight']),
                            bias_init=lambda *_ : jnp.array(param_dict['bias']))
        else:
            assert 'weight' in param_dict
            return nn.Dense(features=features,
                            kernel_init=lambda *_ : jnp.array(param_dict['weight']))


def embedding(num_embeddings, features, param_dict, dtype='float32'):
    if param_dict is None:
        return nn.Embed(num_embeddings=num_embeddings, features=features, dtype=dtype)
    else:
        assert 'weight' in param_dict
        embedding_init = lambda *_ : jnp.array(param_dict['weight'])
        return nn.Embed(num_embeddings=num_embeddings, features=features, embedding_init=embedding_init, dtype=dtype)


#----------------------------------------------------------
# Activation
#----------------------------------------------------------
def apply_activation(x, activation='linear'):
    if activation == 'linear':
        return x
    elif activation == 'gelu_new':
        return 0.5 * x * (1.0 + nn.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * jnp.power(x, 3.0))))
    elif activation == 'gelu_fast':
        return 0.5 * x * (1.0 + nn.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))
    elif activation == 'gelu':
        return jax.nn.gelu(x)
    elif activation == 'relu':
        return jax.nn.relu(x)
    elif activation == 'leaky_relu':
        return jax.nn.leaky_relu(x)
    elif activation == 'sigmoid':
        return jax.nn.sigmoid(x)
    else:
        raise ValueError(f'Unknown activation function: {activation}.')


#----------------------------------------------------------
# Normalization
#----------------------------------------------------------
def layer_norm(param_dict, use_bias=True, use_scale=True, eps=1e-06, dtype='float32'):
    if param_dict is None:
        return nn.LayerNorm(use_bias=use_bias, use_scale=use_scale, epsilon=eps, dtype=dtype)
    else:
        kwargs = {'use_bias': use_bias, 'use_scale': use_scale, 'epsilon': eps, 'dtype': dtype}
        if use_bias:
            assert 'bias' in param_dict, 'use_bias is set True but bias parameter does not exist in param_dict.'
            kwargs['bias_init'] = lambda *_ : jnp.array(param_dict['bias'])
        if use_scale:
            assert 'scale' in param_dict, 'use_scale is set True but scale parameter does not exist in param_dict.'
            kwargs['scale_init'] = lambda *_ : jnp.array(param_dict['scale'])
        return nn.LayerNorm(**kwargs)



#----------------------------------------------------------
# Attention
#----------------------------------------------------------
def split_heads(x, num_heads, head_dim):
    """
    Splits embeddings for different heads.

    Args:
        x (tensor): Input tensor, shape [B, seq_len, embd_dim] or [B, blocks, block_len, embd_dim].
        num_heads (int): Number of heads.
        head_dim (int): Dimension of embedding for each head.

    Returns:
        (tensor): Output tensor, shape [B, num_head, seq_len, head_dim] or [B, blocks, num_head, block_len, head_dim].
    """
    newshape = x.shape[:-1] + (num_heads, head_dim)
    x = jnp.reshape(x, newshape)
    if x.ndim == 5:
        # [batch, blocks, head, block_len, head_dim]
        return jnp.transpose(x, axes=(0, 1, 3, 2, 4))
    elif x.ndim == 4:
        # [batch, head, seq_len, head_dim]
        return jnp.transpose(x, axes=(0, 2, 1, 3))
    else:
        raise ValueError(f'Input tensor should have rank 4 or 5, but has rank {x.ndim}.')


def merge_heads(x, num_heads, head_dim):
    """
    Merge embeddings for different heads.

    Args:
        x (tensor): Input tensor, shape [B, num_head, seq_len, head_dim] or [B, blocks, num_head, block_len, head_dim].
        num_heads (int): Number of heads.
        head_dim (int): Dimension of embedding for each head.

    Returns:
        (tensor): Output tensor, shape [B, seq_len, embd_dim] or [B, blocks, block_len, embd_dim].
    """
    if x.ndim == 5:
        x = jnp.transpose(x, axes=(0, 1, 3, 2, 4))
    elif x.ndim == 4:
        x = jnp.transpose(x, axes=(0, 2, 1, 3))
    else:
        raise ValueError(f'Input tensor should have rank 4 or 5, but has rank {x.ndim}.')

    newshape = x.shape[:-2] + (num_heads * head_dim,)
    x = jnp.reshape(x, newshape)
    return x


def attention(query, key, value, casual_mask, masked_bias, dropout, scale_attn_weights, rng, training, attn_mask=None, head_mask=None):
    """
    Computes Dot-Product Attention for the given query, key and value.
    
    Args:
        query (tensor): Query, shape [B, num_heads, seq_len, embd_dim].
        key (tensor): Key, shape [B, num_heads, seq_len, embd_dim].
        value (tensor): Value, shape [B, num_heads, seq_len, embd_dim].
        casual_mask (tensor): Mask to ensure that attention is only applied to the left of the input sequence, 
                              shape [1, 1, key_len - query_len :key_len, :key_len].
        masked_bias (float): Value to insert for masked part of the sequence.
        dropout (nn.Dropout): Dropout module that is applied to the attention output.
        scale_attn_weights (bool): If True, scale the attention weights.
        rng (jax.random.PRNGKey) Random seed for dropout.
        training (bool): Training mode.
        attn_mask (tensor): Mask to avoid performing attention on padded tokens indices, shape [B, seq_len].
        head_mask (tensor): Mask to nullify selected heads of the self-attention modules, shape [num_heads,] or [num_layers, num_heads].

    Returns:
        (tensor): Attention output, shape [B, num_heads, seq_len, embd_dim].
        (tensor): Attention weights, shape [B, num_heads, seq_len, seq_len].
    """
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)
    attn_weights = jnp.matmul(query, jnp.swapaxes(key, -1, -2))
    
    if scale_attn_weights:
        attn_weights = attn_weights / (float(value.shape[-1]) ** 0.5)

    attn_weights = jnp.where(casual_mask, attn_weights, masked_bias)

    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask
    
    attn_weights = nn.softmax(attn_weights, axis=-1)
    attn_weights = attn_weights.astype(value.dtype)
    attn_weights = dropout(attn_weights, deterministic=not training, rng=rng)

    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    out = jnp.matmul(attn_weights, value)
    return out, attn_weights


#----------------------------------------------------------
# Losses
#----------------------------------------------------------
def cross_entropy(logits, labels, ignore_index=-100):
    """
    Computes the cross entroy loss (on logits).

    Args:
        logits (tensor): Logits, shape [B, num_classes].
        labels (tensor): Labels, shape [B,].
        ignore_index (int): Value of label to ignore for loss computation.

    Returns:
        (tensor): Cross entroy loss.
    """
    batch_size, num_classes = logits.shape
    logits = nn.log_softmax(logits)
    # Get indices where label is equal to ignore_index
    idx = jnp.nonzero(labels == ignore_index)[0]
    one_hot_labels = jax.nn.one_hot(labels, num_classes=num_classes)
    mult = one_hot_labels * logits
    # Insert zeros, where the labels are equal to ignore_index
    jax.ops.index_update(mult, idx=idx, y=jnp.zeros((idx.shape[0], num_classes)))
    return -jnp.sum(jnp.sum(mult, axis=-1)) / (batch_size - idx.shape[0])


#----------------------------------------------------------
# Misc
#----------------------------------------------------------
def get(dictionary, key):
    if dictionary is None or key not in dictionary:
        return None
    return dictionary[key]


def get_attention_mask(attn_mask, batch_size):
    assert batch_size > 0, 'batch_size should be > 0.'
    attn_mask = jnp.reshape(attn_mask, newshape=(batch_size, -1))
    attn_mask = jnp.expand_dims(attn_mask, axis=(1, 2))
    attn_mask = (1.0 - attn_mask) * -10000.0
    return attn_mask


def get_head_mask(head_mask, num_layers):
    if head_mask.ndim == 1:
        head_mask = jnp.expand_dims(head_mask, newshape=(0, 1, -2, -1))
        head_mask = jnp.repeat(head_mask, repeats=num_layers, axis=0)
    elif head_mask.ndim == 2:
        head_mask = jnp.expand_dims(head_mask, newshape=(1, -2, -1))
    else:
        raise ValueError(f'head_mask must have rank 5, but has rank {head_mask.ndim}.')
    return head_mask


def load_config(path):
    return json.loads(open(path, 'r', encoding='utf-8').read(), object_hook=lambda d : SimpleNamespace(**d))


