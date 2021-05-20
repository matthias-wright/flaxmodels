from .third_party.huggingface_transformers.configuration_gpt2 import GPT2Tokenizer
from .. import utils


def get_tokenizer(errors='replace',
                  unk_token='<|endoftext|>',
                  bos_token='<|endoftext|>',
                  eos_token='<|endoftext|>',
                  add_prefix_space=False,
                  ckpt_dir=None):
    """
    Returns the GPT2Tokenizer from Huggingface with loaded merges and vocab files.
    See: https://huggingface.co/transformers/model_doc/gpt2.html#gpt2tokenizer
    
    Args:
        errors (str): Paradigm to follow when decoding bytes to UTF-8.
        unk_token (str): The unknown token. A token that is not in the 
                         vocabulary cannot be converted to an ID and is set to be this token instead.
        bos_token (str): The beginning of sequence token.
        eos_token (str): The end of sequence token.
        add_prefix_space (bool): Whether or not to add an initial space to the input.
                                 This allows to treat the leading word just as any other word.
        ckpt_dir (str): Path to directory, where merges and vocab files are downloaded to.
                        If None, the files will be downloaded to a temp directory.

    Returns:
        (GPT2Tokenizer): GPT2 Tokenizer.

    """
    merges_file = utils.download(ckpt_dir, 'https://www.dropbox.com/s/7f5n1gf348sy1mt/merges.txt?dl=1')
    vocab_file = utils.download(ckpt_dir, 'https://www.dropbox.com/s/s93xkhgcac5nbmn/vocab.json?dl=1')

    return GPT2Tokenizer(vocab_file=vocab_file,
                         merges_file=merges_file,
                         errors=errors,
                         unk_token=unk_token,
                         bos_token=bos_token,
                         eos_token=eos_token,
                         add_prefix_space=add_prefix_space)



