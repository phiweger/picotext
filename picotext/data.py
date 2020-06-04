from collections import namedtuple
import json
from types import SimpleNamespace


def load_config(path):
    '''
    Load params from json file

    https://stackoverflow.com/questions/43921240

    conf = load_config("config.json")
    conf.foo
    '''
    with open(fp, 'r') as fh: 
        d = json.load(fh)
    return SimpleNamespace(**d)


'''
Code adapted from:

https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data.py

TODO:

- we assume that the data can be fit into memory -- add streaming
'''


import os
from io import open
import torch


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.lm.txt'))
        self.dev = self.tokenize(os.path.join(path, 'dev.lm.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.lm.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split()# + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split()# + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids







import numpy as np
from tokenizers import CharBPETokenizer
import torch
from tqdm import tqdm

from picotext.utils import encode_dayhoff, load_sequences, load_pretrained_tokenizer










# class Corpus(object):
#     '''
#     corpus = Corpus(
#         sequences='uniprot_sprot.fasta.gz',
#         tokenizer='uniprot_sprot.dayhoff',
#         split=[0.8, 0.1, 0.1], seed=42)
#     '''
#     def __init__(self, sequences, tokenizer, split=[0.8, 0.1, 0.1], seed=None):
#         self.db = load_sequences(sequences)
#         self.tokenizer = load_pretrained_tokenizer(
#             CharBPETokenizer, tokenizer)

#         self.train_ix, self.dev_ix, self.test_ix = self.train_dev_test_split(
#             len(self.db), split, seed)

#         print('Loading and tokenizing data:')
#         print('Training set ...')
#         self.train = self.tokenize(self.db, self.train_ix, self.tokenizer)
#         print('Dev set ...')
#         self.dev = self.tokenize(self.db, self.dev_ix, self.tokenizer)
#         print('Test set ...')
#         self.test = self.tokenize(self.db, self.test_ix, self.tokenizer)


#     def train_dev_test_split(self, n, sizes=[0.8, 0.1, 0.1], seed=None):
#         '''
#         Get indices split into train, dev and test sets so we can later triage
#         sequences into each category.
    
#         n .. number of sequences in a file, get it e.g. w/
    
#         zcat < uniprot_sprot.fasta.gz | grep ">" | wc -l
#         '''
#         break1 = np.floor(sizes[0] * n).astype(int)
#         break2 = np.floor((sizes[0] + sizes[1]) * n).astype(int)

#         ix = np.arange(n)
#         np.random.seed(seed)
#         np.random.shuffle(ix)

#         train = ix[:break1]
#         dev = ix[break1:break2]
#         test = ix[break2:]

#         if all([len(i) > 0 for i in [train, dev, test]]):
#             return train, dev, test
#         else:
#             raise ValueError('At least one split does not hold any samples')


#     def tokenize(self, db, index, tokenizer):
#         '''
#         Select train/ dev/ test samples from the sequences, encode them as
#         Dayhoff, encode that using BPE and finally return a corresponding vector
#         of numbers.
#         '''
#         ids = []
#         for i in tqdm(index):
#             record = db.loadRecordByIndex(i)
#             seq = db.loadRecordByIndex(i).sequence.__str__()
    
#             dayhoff = encode_dayhoff(seq)
#             # Some sequences contain letters we cannot encode in Dayhoff, skip
#             if dayhoff:
#                 encoded = tokenizer.encode(dayhoff)
#                 ids.append(torch.tensor(encoded.ids).type(torch.int64))
    
#         return torch.cat(ids)


