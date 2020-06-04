'''
TODO:

- add subsample arg to create reduced test set

Aim:

Read in 2 fasta files, one holding positive examples of some sequence, and another holding negative examples.

Create a single fasta w/ a sensical header, e.g.

>id::label
sequence

Then split them into train/ dev/ test sets.

Tokenize.

Reshape so it can enter the NN.
'''
from collections import Counter
from pathlib import Path

import numpy as np
import screed
from tokenizers import CharBPETokenizer
import torch
from tqdm import tqdm


def encode_dayhoff(seq):
    '''
    Turn a protein sequence into its corresponding Dayhoff encoding. Return
    None if the encoding is not unique (for details on this see comment in
    scripts/encoding.py)

    https://en.wikipedia.org/wiki/Margaret_Oakley_Dayhoff

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
    
    except IndexError:
        return None


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


'''
mkdir -p tmp/data
mkdir tmp/processed
cp ToxClassifier/datasets/trainingsets/* tmp/data
'''


tokenizer = load_pretrained_tokenizer(
    CharBPETokenizer, 'tmp/data/uniprot_sprot.dayhoff')


splits = [0.8, 0.1, 0.1]
reformat = lambda x: x.split()[0]
outdir = 'tmp/processed'
labels = {
    'toxin': 'tmp/data/dataset_pos.fa',
    'negative': 'tmp/data/dataset_easy.fa'}


with open(Path(outdir) / 'train.csv', 'w+') as train_out, \
     open(Path(outdir) / 'dev.csv', 'w+') as dev_out, \
     open(Path(outdir) / 'test.csv', 'w+') as test_out:

    for label, path in labels.items():
        cnt = []
    
        with screed.open(path) as file:
            for read in file:
                if reformat:
                    name = reformat(read.name)
                    # sp|Q15WI5|QUEA_PSEA6 S-adeno ...
                else:
                    name = read.name
    
                throw = np.random.multinomial(1, splits)
                # [0, 0, 1] -- now in which position is 1?
                group = ['train', 'dev', 'test'][list(throw).index(1)]
                # groups[name] = group
                cnt.append(group)
    
                dayhoff = encode_dayhoff(read.sequence)
                if dayhoff:
                    sequence = tokenizer.encode(dayhoff).tokens
                else:
                    continue

                str_ = f'{name},{label},{" ".join(sequence)}\n'

                if group == 'train':
                    train_out.write(str_)
                elif group == 'dev':
                    dev_out.write(str_)
                elif group == 'test':
                    test_out.write(str_)
                else:
                    raise ValueError('A very specific bad thing happened.')
    
        print(label, Counter(cnt))

# from collections import Counter
# print(Counter(cnt))
