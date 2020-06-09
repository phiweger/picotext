from pathlib import Path
import json
from types import SimpleNamespace

import screed
from screed import ScreedDB
from tokenizers import CharBPETokenizer
import torch


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


def load_pretrained_tokenizer(tokenizer_type, path_prefix):
    '''
    Load a pretrained tokenizer.

    Usage:

    from tokenizers import CharBPETokenizer
    seq = 'eeeefebeeb'
    tokenizer = load_pretrained_tokenizer(
        CharBPETokenizer, './uniprot_sprot.dayhoff')
    encoded = tokenizer.encode(seq)
    '''
    vocab = f'{path_prefix}-vocab.json'
    merges = f'{path_prefix}-merges.txt'
    return tokenizer_type(vocab, merges)


def load_sequences(path):
    '''
    Load the sequences and index them if no index has been computed before.
    '''
    if not Path(path + '_screed').is_file():
        print('Indexing sequences ...')
        screed.make_db(path)
    return ScreedDB(path)


def load_config(path):
    '''
    Load params from json file

    https://stackoverflow.com/questions/43921240

    conf = load_config("config.json")
    conf.foo
    '''
    with open(path, 'r') as fh: 
        d = json.load(fh)
    return SimpleNamespace(**d)


def encode_dayhoff(seq):
    '''
    Turn a protein sequence into its corresponding Dayhoff encoding. Return
    None if the encoding is not unique (for details on this see comment in
    scripts/encoding.py)

    https://en.wikipedia.org/wiki/Margaret_Oakley_Dayhoff

    a: sulfur polymerization
    b: small
    c: acid and amide
    d: acid and amide
    e: hydrophobic
    f: aromatic

    Usage:

    seq = 'MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVEC'
    encode_dayhoff(seq)
    # 'ebfbbcceedcfcddddecbeeebeffbccddeecfdcfbbbdececa'
    '''
    dayhoff = {
        'C' + 'U': 'a', 
        'GSTAP': 'b',
        'DENQ' + 'Z': 'c',
        'RHK' + 'O': 'd',
        'LVMI' + 'J': 'e',
        'YFW': 'f'}

    encoding = ''

    try:
        for letter in seq:
            k_ = [k for k in dayhoff if letter in k][0]
            encoding += dayhoff[k_]
        return encoding
    
    except IndexError:  # amino-acid not in our dict
        return None


def batchify(data, bsz, device):
    '''Splits text into chunks of words of length bsz, "kmerize".'''
    # TODO: replace w/ BPTTIterator?
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def repackage_hidden(h):
    '''Detach hidden states from their history.'''
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

