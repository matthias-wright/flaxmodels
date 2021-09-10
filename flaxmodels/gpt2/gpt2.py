import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any
import h5py

from .. import utils
from . import ops


URLS = {'gpt2': 'https://www.dropbox.com/s/0wdgj0gazwt9nm7/gpt2.h5?dl=1',
        'gpt2-medium': 'https://www.dropbox.com/s/nam11kbd83wsm7d/gpt2-medium.h5?dl=1',
        'gpt2-large': 'https://www.dropbox.com/s/oy8623qwkkjm8gt/gpt2-large.h5?dl=1',
        'gpt2-xl': 'https://www.dropbox.com/s/6c6qt0bzz4v2afx/gpt2-xl.h5?dl=1'}

CONFIGS = {'gpt2': 'https://www.dropbox.com/s/s5xl32dgwc8322p/gpt2.json?dl=1',
           'gpt2-medium': 'https://www.dropbox.com/s/7mwkijxoh1earm5/gpt2-medium.json?dl=1',
           'gpt2-large': 'https://www.dropbox.com/s/nhslkxwxtpn7auz/gpt2-large.json?dl=1',
           'gpt2-xl': 'https://www.dropbox.com/s/1iv0nq1xigsfdvb/gpt2-xl.json?dl=1'}


class GPT2SelfAttention(nn.Module):
    """
    GPT2 Self Attention.

    Attributes:
        config (Any): Configuration object. If 'pretrained' is not None, this parameter will be ignored.
        param_dict (dict): Parameter dict with pretrained parameters. If not None, 'pretrained' will be ignored.
    """
    config: dict=None
    param_dict: dict=None
    
    def setup(self):
        self.max_pos = self.config.n_positions
        self.embd_dim = self.config.n_embd
        self.num_heads = self.config.n_head
        self.head_dim = self.embd_dim // self.num_heads
        self.attn_dropout = self.config.attn_pdrop
        self.resid_dropout = self.config.resid_pdrop
        self.scale_attn_weights = self.config.scale_attn_weights

    @nn.compact
    def __call__(self, x, layer_past=None, attn_mask=None, head_mask=None, use_cache=False, training=False, rng=jax.random.PRNGKey(0)):
        """
        Run attention.

        Args:
            x (tensor): Input tensor.
            layer_past (Tuple): Tuple of past keys and values.
            attn_mask (tensor): Mask to avoid performing attention on padding token indices.
            head_mask (tensor): Mask to nullify selected heads of the self-attention modules.
            use_cache (bool): If True, keys and values are returned (past_key_values).
            training (bool): Training mode.
            rng (jax.random.PRNGKey): Random seed for dropout.

        Returns:
            (tensor, Tuple): Output tensor, tuple of keys and values.
        """
        x = ops.linear(3 * self.embd_dim, ops.get(self.param_dict, 'c_proj'))(x)
        query, key, value = jnp.split(x, 3, axis=2)

        query = ops.split_heads(query, self.num_heads, self.head_dim)
        value = ops.split_heads(value, self.num_heads, self.head_dim)
        key = ops.split_heads(key, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = jnp.concatenate((past_key, key), axis=-2)
            value = jnp.concatenate((past_value, value), axis=-2)

        present = (key, value) if use_cache else None

        query_len, key_len = query.shape[-2], key.shape[-2]
        casual_mask = jnp.tril(jnp.ones((1, 1, self.max_pos, self.max_pos)))[:, :, key_len - query_len :key_len, :key_len]
        casual_mask = casual_mask.astype(bool)

        attn_dropout = nn.Dropout(rate=self.attn_dropout)
        out, _ = ops.attention(query, key, value, casual_mask, -1e4, attn_dropout, self.scale_attn_weights, rng, training, attn_mask, head_mask)
        out = ops.merge_heads(out, self.num_heads, self.head_dim)
        out = ops.linear(self.embd_dim, ops.get(self.param_dict, 'out_proj'))(out)
        _, rng = jax.random.split(rng)
        out = nn.Dropout(rate=self.resid_dropout)(out, deterministic=not training, rng=rng)
        return out, present


class GPT2MLP(nn.Module):
    """
    GPT2 MLP.

    Attributes:
        intermediate_dim (int): Dimension of the intermediate layer.
        config (Any): Configuration object. If 'pretrained' is not None, this parameter will be ignored.
        param_dict (dict): Parameter dict with pretrained parameters. If not None, 'pretrained' will be ignored.
    """
    intermediate_dim: int
    config: dict=None
    param_dict: dict=None
    
    def setup(self):
        self.embd_dim = self.config.n_embd
        self.resid_dropout = self.config.resid_pdrop
        self.activation = self.config.activation_function

    @nn.compact
    def __call__(self, x, training=False, rng=jax.random.PRNGKey(0)):
        """
        Run the MLP.

        Args:
            x (tensor): Input tensor.
            training (bool): Training mode.
            rng (jax.random.PRNGKey): Random seed for dropout.
        """
        x = ops.linear(self.intermediate_dim, ops.get(self.param_dict, 'c_fc'))(x)
        x = ops.apply_activation(x, activation=self.activation)
        x = ops.linear(self.embd_dim, ops.get(self.param_dict, 'c_proj'))(x)
        x = nn.Dropout(rate=self.resid_dropout)(x, deterministic=not training, rng=rng)
        return x


class GPT2Block(nn.Module):
    """
    GPT2 Block.

    Attributes:
        config (Any): Configuration object. If 'pretrained' is not None, this parameter will be ignored.
        param_dict (dict): Parameter dict with pretrained parameters. If not None, 'pretrained' will be ignored.
    """
    config: dict=None
    param_dict: dict=None
    
    def setup(self):
        self.embd_dim = self.config.n_embd
        self.eps = self.config.layer_norm_epsilon
        self.inner_dim = self.config.n_inner if self.config.n_inner is not None else 4 * self.embd_dim

    @nn.compact
    def __call__(self, x, layer_past=None, attn_mask=None, head_mask=None, use_cache=False, training=False, rng=jax.random.PRNGKey(0)):
        """
        Run the block.

        Args:
            x (tensor): Input tensor.
            layer_past (Tuple): Tuple of past keys and values.
            attn_mask (tensor): Mask to avoid performing attention on padding token indices.
            head_mask (tensor): Mask to nullify selected heads of the self-attention modules.
            use_cache (bool): If True, keys and values are returned (past_key_values).
            training (bool): Training mode.
            rng (jax.random.PRNGKey): Random seed for dropout.

        Returns:
            (tensor, Tuple): Output tensor, tuple of keys and values.
        """
        residual = x
        x = ops.layer_norm(ops.get(self.param_dict, 'ln_1'), eps=self.eps)(x)
        kwargs = {'layer_past': layer_past, 'attn_mask': attn_mask, 'head_mask': head_mask,
                'use_cache': use_cache, 'training': training, 'rng': rng}
        x, present = GPT2SelfAttention(self.config, ops.get(self.param_dict, 'attn'))(x, **kwargs)
        x += residual

        residual = x
        x = ops.layer_norm(ops.get(self.param_dict, 'ln_2'), eps=self.eps)(x)
        _, rng = jax.random.split(rng)
        x = GPT2MLP(self.inner_dim, self.config, ops.get(self.param_dict, 'mlp'))(x, training, rng)
        x += residual
        return x, present


class GPT2Model(nn.Module):
    """
    The GPT2 Model.

    Attributes:
        config (Any): Configuration object. If 'pretrained' is not None, this parameter will be ignored.
        pretrained (str): Which pretrained model to use, None for random initialization.
        ckpt_dir (str): Directory to which the pretrained weights are downloaded. If None, a temp directory will be used.
        param_dict (dict): Parameter dict with pretrained parameters. If not None, 'pretrained' will be ignored.
    """
    config: dict=None
    pretrained: str=None
    ckpt_dir: str=None
    param_dict: dict=None
    
    def setup(self):
        if self.pretrained is not None:
            assert self.pretrained in URLS.keys(), f'Pretrained model not available {self.pretrained}.'
            ckpt_file = utils.download(self.ckpt_dir, URLS[self.pretrained])
            self.param_dict_ = h5py.File(ckpt_file, 'r')['transformer']
            config_file = utils.download(self.ckpt_dir, CONFIGS[self.pretrained])
            self.config_ = ops.load_config(config_file)
        else:
            self.config_ = self.config
            self.param_dict_ = self.param_dict
        self.vocab_size = self.config_.vocab_size
        self.max_pos = self.config_.n_positions
        self.embd_dim = self.config_.n_embd
        self.embd_dropout = self.config_.embd_pdrop
        self.num_layers = self.config_.n_layer
        self.eps = self.config_.layer_norm_epsilon

    @nn.compact
    def __call__(self,
                 input_ids=None,
                 past_key_values=None,
                 input_embds=None,
                 position_ids=None,
                 attn_mask=None,
                 head_mask=None,
                 use_cache=False,
                 training=False,
                 rng=jax.random.PRNGKey(0)):
        """
        Run the model.

        Args:
            input_ids (tensor): Input token ids, shape [B, seq_len].
            past_key_values (Tuple): Precomputed hidden keys and values, tuple of tuples.
                                     If past_key_values is used, only input_ids that do not have their
                                     past calculated should be passed as input_ids.
            input_embds (tensor): Input embeddings, shape [B, seq_len, embd_dim].
            labels (tensor): Labels for language modeling, shape [B, seq_len]. Will be shifted inside the model. Ignore label = -100.
            position_ids (tensor): Indices of positions of each input sequence tokens in the position embeddings, shape [B, seq_len].
            attn_mask (tensor): Mask to avoid performing attention on padding token indices, shape [B, seq_len].
            head_mask (tensor): Mask to nullify selected heads of the self-attention modules, shape [num_heads] or [num_layers, num_heads].
            use_cache (bool): If True, keys and values are returned (past_key_values).
            training (bool): Training mode.
            rng (jax.random.PRNGKey): Random seed for dropout.

        Returns:
            (dict): Dictionary containing 'last_hidden_state', 'past_key_values'.            
        """
        if input_ids is not None and input_embds is not None:
            raise ValueError('You cannot specify both input_ids and input_embd at the same time.')
        elif input_ids is not None:
            input_shape = input_ids.shape
            input_ids = jnp.reshape(input_ids, newshape=(-1, input_shape[-1]))
            batch_size = input_ids.shape[0]
        elif input_embds is not None:
            input_shape = input_embds.shape[:-1]
            batch_size = input_embds.shape[0]
        else:
            raise ValueError('You have to specify either input_ids or input_embd.')

        if position_ids is not None:
            position_ids = jnp.reshape(position_ids, newshape=(-1, input_shape[-1]))

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.num_layers)
        else:
            past_length = past_key_values[0][0].shape[-2]
        
        if position_ids is None:
            position_ids = jnp.arange(start=past_length, stop=input_shape[-1] + past_length)
            position_ids = jnp.reshape(jnp.expand_dims(position_ids, axis=0), newshape=(-1, input_shape[-1])) 

        if input_embds is None:
            input_embds = ops.embedding(self.vocab_size, self.embd_dim, ops.get(self.param_dict_, 'token_embd'))(input_ids)

        if attn_mask is not None:
            attn_mask = ops.get_attention_mask(attn_mask, batch_size)

        if head_mask is not None:
            head_mask = ops.get_head_mask(head_mask, self.num_layers)
        else:
            head_mask = [None] * self.num_layers
        
        position_embds = ops.embedding(self.max_pos, self.embd_dim, ops.get(self.param_dict_, 'pos_embd'))(position_ids)
        x = input_embds + position_embds
        
        x = nn.Dropout(rate=self.embd_dropout)(x, deterministic=not training, rng=rng)
        output_shape = input_shape + (x.shape[-1],)

        presents = () if use_cache else None
        for i in range(self.num_layers):
            kwargs = {'layer_past': past_key_values[i], 'attn_mask': attn_mask, 'head_mask': head_mask[i],
                      'use_cache': use_cache, 'training': training, 'rng': rng}
            _, rng = jax.random.split(rng)
            x, present = GPT2Block(self.config_, ops.get(self.param_dict_, f'block{i}'))(x, **kwargs)
            if use_cache:
                presents = presents + (present,)

        x = ops.layer_norm(ops.get(self.param_dict_, 'ln_final'), eps=self.eps)(x)
        return {'last_hidden_state': x, 'past_key_values': presents}


class GPT2LMHeadModel(nn.Module):
    """
    The GPT2 Model transformer with a language model head on top.

    Attributes:
        config (Any): Configuration object. If 'pretrained' is not None, this parameter will be ignored.
        pretrained (str): Which pretrained model to use, None for random initialization.
        ckpt_dir (str): Directory to which the pretrained weights are downloaded. If None, a temp directory will be used.
    """
    config: Any=None
    pretrained: str=None
    ckpt_dir: str=None
    
    def setup(self):
        if self.pretrained is not None:
            assert self.pretrained in URLS.keys(), f'Pretrained model not available {self.pretrained}.'
            ckpt_file = utils.download(self.ckpt_dir, URLS[self.pretrained])
            self.param_dict = h5py.File(ckpt_file, 'r')
            config_file = utils.download(self.ckpt_dir, CONFIGS[self.pretrained])
            self.config_ = ops.load_config(config_file)
        else:
            self.config_ = self.config
        self.vocab_size = self.config_.vocab_size
        self.max_pos = self.config_.n_positions
        self.embd_dim = self.config_.n_embd
        self.embd_dropout = self.config_.embd_pdrop
        self.num_layers = self.config_.n_layer
        self.eps = self.config_.layer_norm_epsilon

    @nn.compact
    def __call__(self,
                 input_ids=None,
                 past_key_values=None,
                 input_embds=None,
                 labels=None,
                 position_ids=None,
                 attn_mask=None,
                 head_mask=None,
                 use_cache=False,
                 training=False,
                 rng=jax.random.PRNGKey(0)):
        """
        Run the model.

        Args:
            input_ids (tensor): Input token ids, shape [B, seq_len].
            past_key_values (Tuple): Precomputed hidden keys and values, tuple of tuples.
                                     If past_key_values is used, only input_ids that do not have their
                                     past calculated should be passed as input_ids.
            input_embds (tensor): Input embeddings, shape [B, seq_len, embd_dim].
            labels (tensor): Labels for language modeling, shape [B, seq_len]. Will be shifted inside the model. Ignore label = -100.
            position_ids (tensor): Indices of positions of each input sequence tokens in the position embeddings, shape [B, seq_len].
            attn_mask (tensor): Mask to avoid performing attention on padding token indices, shape [B, seq_len].
            head_mask (tensor): Mask to nullify selected heads of the self-attention modules, shape [num_heads] or [num_layers, num_heads].
            use_cache (bool): If True, keys and values are returned (past_key_values).
            training (bool): Training mode.
            rng (jax.random.PRNGKey): Random seed for dropout.

        Returns:
            (dict): Dictionary containing 'last_hidden_state', 'past_key_values', 'loss', and 'logits'.            
        """
        kwargs = {'input_ids': input_ids,
                  'past_key_values': past_key_values,
                  'input_embds': input_embds, 
                  'position_ids': position_ids, 
                  'attn_mask': attn_mask, 
                  'head_mask': head_mask,
                  'use_cache': use_cache,
                  'training': training,
                  'rng': rng}
        output = GPT2Model(self.config_, param_dict=ops.get(self.param_dict, 'transformer'))(**kwargs)
        lm_logits = ops.linear(self.vocab_size, ops.get(self.param_dict, 'lm_head'), bias=False)(output['last_hidden_state'])

        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # flatten the tokens
            loss = ops.cross_entropy(jnp.reshape(shift_logits, (-1, shift_logits.shape[-1])), jnp.reshape(shift_labels, (-1)))
        
        output['loss'] = loss
        output['logits'] = lm_logits
        return output


