#! /usr/bin/env python

'''
- https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb
- https://github.com/pytorch/examples/blob/master/word_language_model/main.py

TODO:

- get line by line seq from uniprot (cleaned) then shuffle and split into train, test, dev using shuf and head | tail
- tensorboard ... write avg loss not llost of last batch

python .../picotext/scripts/preprocess.py --seq uniref50.fasta.gz --out uniref50.clean.dayhoff.txt -p 1 --skip-header --maxlen 2000 --excluded-aa XBZJ


'''
import argparse
from collections import Counter

import math
from tokenizers import CharBPETokenizer
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from picotext.data import Corpus
from picotext.model import RNN_lm, RNN_tr
from picotext.utils import batchify, get_batch, repackage_hidden
from picotext.utils import load_config, load_pretrained_tokenizer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.

    ntokens = len(corpus.vocab)
    hidden = model.init_hidden(batch_size)
    
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        # To overfit one batch do
        # ... in [next(enumerate(range(0, train_data.size(0) - 1, bptt)))]
        # Then revert
        # ... in enumerate(range(0, train_data.size(0) - 1, bptt))
        data, targets = get_batch(train_data, i, bptt)
        '''
        Starting each batch, we detach the hidden state from how it was previously produced. If we didn't, the model would try backpropagating all the way to start of the dataset.
        '''
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()

        # train_loss = round(loss.detach().item(), 4)
        if batch % log_interval == 0:
            print(round(total_loss / log_interval, 4))
            total_loss = 0.

    return train_loss


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.

    ntokens = len(corpus.vocab)
    hidden = model.init_hidden(eval_batch_size)
    
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            
            data, targets = get_batch(data_source, i, bptt)
            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            
            loss = criterion(output, targets)
            total_loss += len(data) * criterion(output, targets).item()

    loss_avg = round(total_loss / (len(data_source) - 1), 4)

    sm = nn.Softmax(dim=1)
    p_max = torch.argsort(sm(output), dim=1)[:, -10:]

    # TODO: how many words are correct? make this run on test time only, bc/
    # computationally intensive
    # nextword = []
    # for ix, i in enumerate(p_max):
    #     if ix < 5:
    #         print(ix, i, targets[ix])
    #     nextword.append(any([x == targets[ix] for x in i]))

    # topk = round(sum(nextword)/len(nextword), 4)
    # print(f'Top-k correct words: {topk}')

    return loss_avg


c = load_config('config.json')

batch_size = 100
eval_batch_size = 10
bptt = 30
clip = 0.5
log_interval = 100
lr = 0.001# 3e-4  #20
best_val_loss = None
epochs = 1
save = 'foo'
emsize = 100
nhid = 100#1024
nlayers = 2
dropout = 0.5
tied = False
save = 'foo.model'

writer = SummaryWriter(log_dir='log')
# tensorboard --logdir .
# in floydhub https://docs.floydhub.com/guides/jobs/tensorboard/


'''
cd .../picotext/journal/2020-05-23T1315/tmp/processed
for i in train dev test
do
    cut -d, -f3 ${i}.csv | head -n 10000 > ${i}.lm.txt
done

!wc train.lm.txt
    43785 2983318 17233692 train.lm.txt
len(corpus.train)
    2983318

TODO: Later make sure to turn the same word into the same number btw/
models -- maybe save a json for the tokens and map <unk> and <pad> to
0 and 1 respectively

TEXT.vocab.itos[1] ... '<pad>'
TEXT.vocab.itos[0] ... '<unk>'
'''
tokenizer = load_pretrained_tokenizer(CharBPETokenizer, c.tokenizer)

print('Loading corpus ...')
corpus = Corpus('.', tokenizer)
ntokens = len(corpus.vocab)

print('Moving along')
train_data = batchify(corpus.train, batch_size, device)
dev_data = batchify(corpus.dev, eval_batch_size, device)
test_data = batchify(corpus.test, eval_batch_size, device)



# Instead of ntokens we pass in nclass here
# init_args =['GRU', nclass, emsize, nhid, nlayers, dropout, tied]

init_args = {
    'rnn_type': 'GRU',
    'ntoken': ntokens,
    'ninp': emsize,
    'nhid': nhid,
    'nlayers': nlayers,
    'dropout': dropout,
    'tie_weights': tied,
    }


model = RNN_lm(init_args).to(device)
criterion = nn.NLLLoss().to(device)
'''
nn.NLLLoss?

> The `input` given through a forward call is expected to contain
log-probabilities of each class. Obtaining log-probabilities in a neural 
network is easily achieved by adding a  `LogSoftmax`  layer in the last layer
of your network. You may use `CrossEntropyLoss` instead, if you prefer not to
add an extra layer.

nn.CrossEntropyLoss?

> This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single
class. -- https://pytorch.org/docs/stable/nn.html#crossentropyloss
'''
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# TODO: scheduler


best_dev_loss = None
# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, epochs+1):
        train_loss = train()
        dev_loss = evaluate(dev_data)
        dev_ppl = round(math.exp(dev_loss), 4)  # perplexity
        
        print('-' * 80)
        print(f'| Epoch {epoch} | Dev loss {dev_loss} | Dev ppl {dev_ppl}')
        print('-' * 80)
        
        writer.add_scalar('Train loss', round(train_loss, 4), epoch)
        writer.add_scalar('Dev loss', round(dev_loss, 4), epoch)

        # Save the model if the validation loss is the best we've seen so far.
        if not best_dev_loss or dev_loss < best_dev_loss:
            with open(save, 'wb') as out:
                torch.save(model, out)
            best_dev_loss = dev_loss

except KeyboardInterrupt:
    print('-' * 80)
    print('Exiting from training early')


test_loss = evaluate(test_data)
test_ppl = round(math.exp(test_loss), 4)
print('=' * 80)
print(f'| End of training | Test loss {test_loss} | Test ppl {test_ppl}')
print('=' * 80)

