from pathlib import Path

import screed
from screed import ScreedDB

from tokenizers import CharBPETokenizer

def train_tokenizer(tokenizer, files: list, **kwargs):
    '''
    https://github.com/huggingface/tokenizers/tree/master/bindings/python

    Usage:

    # Encode proteins in Dayhoff encoding, one sequence per line
    # !pip install screed tqdm tokenizers==0.7.0
    files = ['./uniprot_sprot.dayhoff.txt']
    tokenizer = train_tokenizer(
        CharBPETokenizer(bert_normalizer=False),
        files,
        vocab_size=10000, min_frequency=2)
    tokenizer.save('.', 'uniprot_sprot.dayhoff')
    '''
    tokenizer.train(files, **kwargs)
    return tokenizer

v = 30000
f = 5

#files = ['2020-05-29_uniref50.0.03125.dayhoff.txt']
#tokenizer = train_tokenizer(
#    CharBPETokenizer(bert_normalizer=False),
#    files,
#    vocab_size=v, min_frequency=f, special_tokens=['<unk>', '<pad>'])
#tokenizer.save('.', f"uniref50.0.03125.dayhoff.vocab{v}.freq{f}")

#files = ['2020-05-29_uniref50.0.0625.dayhoff.txt']
#tokenizer = train_tokenizer(
#    CharBPETokenizer(bert_normalizer=False),
#    files,
#    vocab_size=v, min_frequency=f, special_tokens=['<unk>', '<pad>'])
#tokenizer.save('.', f"uniref50.0.0625.dayhoff.vocab{v}.freq{f}")

#files = ['2020-05-29_uniref50.0.125.dayhoff.txt']
#tokenizer = train_tokenizer(
#    CharBPETokenizer(bert_normalizer=False),
#    files,
#    vocab_size=v, min_frequency=f, special_tokens=['<unk>', '<pad>'])
#tokenizer.save('.', f"uniref50.0.125.dayhoff.vocab{v}.freq{f}")

#files = ['2020-05-29_uniref50.0.25.dayhoff.txt']
#tokenizer = train_tokenizer(
#    CharBPETokenizer(bert_normalizer=False),
#    files,
#    vocab_size=v, min_frequency=f, special_tokens=['<unk>', '<pad>'])
#tokenizer.save('.', f"uniref50.0.25.dayhoff.vocab{v}.freq{f}")

#files = ['2020-05-29_uniref50.0.5.dayhoff.txt']
#tokenizer = train_tokenizer(
#    CharBPETokenizer(bert_normalizer=False),
#    files,
#    vocab_size=v, min_frequency=f, special_tokens=['<unk>', '<pad>'])
#tokenizer.save('.', f"uniref50.0.5.dayhoff.vocab{v}.freq{f}")

files = ['2020-05-29_uniref50.1.dayhoff.txt']
tokenizer = train_tokenizer(
    CharBPETokenizer(bert_normalizer=False),
    files,
    vocab_size=v, min_frequency=f, special_tokens=['<unk>', '<pad>'])
tokenizer.save('.', f"uniref50.1.dayhoff.vocab{v}.freq{f}")
