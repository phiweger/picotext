'''
Code adapted from:

https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data.py

TODO:

- we assume that the data can be fit into memory -- add streaming
'''
import os
from io import open
import torch
from tqdm import tqdm

'''
mkdir -p tmp/data
mkdir tmp/processed
cp ToxClassifier/datasets/trainingsets/* tmp/data
'''


# tokenizer = load_pretrained_tokenizer(
#     CharBPETokenizer, 'tmp/data/uniprot_sprot.dayhoff')



# # TODO:
# # CharBPETokenizer.train?
# # special_tokens: List[Union[str, AddedToken]] = ['<unk>'],
# files = ['2020-05-29_uniref50.dayhoff.txt']

# tokenizer = train_tokenizer(
#     CharBPETokenizer(bert_normalizer=False),
#     files,
#     vocab_size=10000, min_frequency=2,
#     special_tokens=['<unk>', '<pad>'])
# tokenizer.save('.', '2020-05-29_uniref50.dayhoff.vocab10k.freq2')



'''
class Dictionary(object):
    def __init__(self, tokenizer):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)
'''



class Corpus(object):
    def __init__(self, path, tokenizer):
        self.vocab = tokenizer.get_vocab()
        self.train = self.tokenize(
            os.path.join(path, 'train.lm.txt'), tokenizer)
        self.dev = self.tokenize(
            os.path.join(path, 'dev.lm.txt'), tokenizer)
        self.test = self.tokenize(
            os.path.join(path, 'test.lm.txt'), tokenizer)

    def tokenize(self, path, tokenizer):
        """Tokenizes a text file."""
        assert os.path.exists(path)

        print(path)
        with open(path, 'r', encoding="utf8") as file:
            idss = []

            for line in tqdm(file):
                ids = tokenizer.encode(line.strip()).ids
                ids = torch.tensor(ids).type(torch.int64)
                idss.append(ids)

            ids = torch.cat(idss)
        return ids


'''
class Corpus(object):
    def __init__(self, path, tokenizer):
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
'''