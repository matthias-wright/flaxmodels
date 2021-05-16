from .third_party.huggingface_transformers.configuration_gpt2 import GPT2Tokenizer
from .. import utils


def get_tokenizer(errors='replace',
                  unk_token='<|endoftext|>',
                  bos_token='<|endoftext|>',
                  eos_token='<|endoftext|>',
                  add_prefix_space=False,
                  ckpt_dir=None):

    merges_file = utils.download(ckpt_dir, 'https://www.dropbox.com/s/7f5n1gf348sy1mt/merges.txt?dl=1')
    vocab_file = utils.download(ckpt_dir, 'https://www.dropbox.com/s/s93xkhgcac5nbmn/vocab.json?dl=1')

    return GPT2Tokenizer(vocab_file=vocab_file,
                         merges_file=merges_file,
                         errors=errors,
                         unk_token=unk_token,
                         bos_token=bos_token,
                         eos_token=eos_token,
                         add_prefix_space=add_prefix_space)



