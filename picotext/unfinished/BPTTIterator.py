


'''
> Provides contiguous streams of examples together with targets that are one timestep further forward, for language modeling training with backpropagation through time (BPTT). Expects a Dataset with a single example and a single field called ‘text’ and produces Batches with text and target attributes.
'''

import torch
from torchtext import data

TEXT = data.Field(sequential=True)


TEXT = data.Field(sequential=True, include_lengths=False)
'''
> TorchText Fields have a preprocessing argument. A function passed here will be applied to a sentence after it has been tokenized (transformed from a string into a list of tokens), but before it has been numericalized (transformed from a list of tokens to a list of indexes). This is where we'll pass our generate_bigrams function.
'''
LABELS = data.LabelField(dtype=torch.float)
NAMES = data.RawField(is_target=False)
# TODO: preprocessing=... useful?


# Fields are added by column left to write in the underlying table
fields=[('name', NAMES), ('label', LABELS), ('text', TEXT)]

train, dev, test = data.TabularDataset.splits(
    path='.', format='CSV', fields=fields,
    train='train.csv', validation='dev.csv', test='test.csv')

TEXT.build_vocab(train)
# TEXT.vocab.itos[1] ... '<pad>'
# TEXT.vocab.itos[0] ... '<unk>'
LABELS.build_vocab(train)


a = next(iter(data.BPTTIterator(train, 20, 20)))


train_iter, dev_iter, test_iter = data.BPTTIterator.splits(
    ([i.text for i in train], dev, test),
    bptt_len=13,
    batch_size=7,
    sort_key=lambda x: len(x.text),
    device='cpu')




# https://mlexplained.com/2018/02/15/language-modeling-tutorial-in-torchtext-practical-torchtext-part-2/
from torchtext.datasets import WikiText2
train, valid, test = WikiText2.splits(TEXT) # loading custom datas
len(train)


data.Example?







