'''
This script takes a fasta file and splits it into three datasets: train, dev
and test. Our use case is language modelling, and we assume that the data
has to be streamed during e.g. training bc/ it would be too big to load into
memory.
'''

from pathlib import Path

import screed
from screed import ScreedDB

from tokenizers import CharBPETokenizer

from utils import encode_dayhoff, load_db, load_pretrained_tokenizer



path = 'uniprot_sprot.fasta.gz'
db = load_db(path)
train, dev, test = train_dev_test_split(len(db), [0.8, 0.1, 0.1], 42)

tokenizer = load_pretrained_tokenizer(
    CharBPETokenizer, './uniprot_sprot.dayhoff')

ids_train = tokenize(db, dev, tokenizer)
# TODO: don't tokenize at this step, but write Dayhoff to file as train.txt etc.



class Corpus(object):
    '''
    corpus = Corpus(
        db='uniprot_sprot.fasta.gz', tokenizer='uniprot_sprot.dayhoff')
    '''
    def __init__(self, db, tokenizer):
        self.db = load_db(db)
        self.tokenizer = load_pretrained_tokenizer(
            CharBPETokenizer, tokenizer)

        self.train_ix, self.dev_ix, self.test_ix = self.train_dev_test_split(
            len(self.db), [0.8, 0.1, 0.1], 42)

        self.train = self.tokenize(self.db, self.train_ix, self.tokenizer)
        self.dev = self.tokenize(self.db, self.dev_ix, self.tokenizer)
        self.test = self.tokenize(self.db, self.test_ix, self.tokenizer)


    def train_dev_test_split(self, n, sizes=[0.8, 0.1, 0.1], seed=None):
        '''
        Get indices split into train, dev and test sets so we can later triage
        sequences into each category.
    
        n .. number of sequences in a file, get it e.g. w/
    
        zcat < uniprot_sprot.fasta.gz | grep ">" | wc -l
        '''
        break1 = np.floor(sizes[0] * n).astype(int)
        break2 = np.floor((sizes[0] + sizes[1]) * n).astype(int)

        ix = np.arange(n)
        np.random.seed(seed)
        np.random.shuffle(ix)

        train = ix[:break1]
        dev = ix[break1:break2]
        test = ix[break2:]

        if all([len(i) > 0 for i in [train, dev, test]]):
            return train, dev, test
        else:
            raise ValueError('At least one split does not hold any samples')


    def tokenize(self, db, index, tokenizer):
        '''
        Select train/ dev/ test samples from the sequences, encode them as
        Dayhoff, encode that using BPE and finally return a corresponding vector
        of numbers.
        '''
        ids = []
        for i in tqdm(index):
            record = db.loadRecordByIndex(i)
            seq = db.loadRecordByIndex(i).sequence.__str__()
    
            dayhoff = encode_dayhoff(seq)
            # Some sequences contain letters we cannot encode in Dayhoff, skip
            if dayhoff:
                encoded = tokenizer.encode(dayhoff)
                ids.append(torch.tensor(encoded.ids).type(torch.int64))
    
        return torch.cat(ids)


