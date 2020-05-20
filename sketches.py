import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

from picotext.data import Corpus
from picotext.model import RNNModel


'''
> When we do language modeling, we will infer the labels from the text during training, so there's no need to label. The training loop expects labels however, so we need to add dummy ones.
-- https://github.com/fastai/course-v3/blob/master/nbs/dl2/12_text.ipynb
'''


# https://github.com/pytorch/examples/blob/master/word_language_model/main.py
# https://github.com/Smerity/sha-rnn


'''
!head -n 1000 uniprot_sprot.dayhoff.txt > train.txt
!tail -n 1000 uniprot_sprot.dayhoff.txt > test.txt
!head -n 2000 uniprot_sprot.dayhoff.txt | tail -n 1000 > valid.txt
'''


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = corpus.tokenizer.get_vocab_size()
    # if args.model != 'Transformer':
    
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            # if args.model == 'Transformer':
            #     output = model(data)
            #     output = output.view(-1, ntokens)
            # else:
            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = corpus.tokenizer.get_vocab_size()  # len(corpus.dictionary)
    # if args.model != 'Transformer':
    hidden = model.init_hidden(batch_size)
    
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        # if args.model == 'Transformer':
        #     output = model(data)
        #     output = output.view(-1, ntokens)
        # else:
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // bptt, lr,
                elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


device = 'cpu'
# preprocessing
batch_size = 20
eval_batch_size = 10
# model
model_type = 'GRU'  # LSTM
ntokens = corpus.tokenizer.get_vocab_size()
emsize = 200
nhid = 200
nlayers = 2
dropout = 0.2
tied = False
# training
lr = 20
best_val_loss = None
epochs = 10
bptt = 35
clip = 0.25
log_interval = 200
save = 'final'


corpus = Corpus(
    sequences='uniprot_sprot.fasta.gz',
    tokenizer='uniprot_sprot.dayhoff',
    split=[0.8, 0.1, 0.1], seed=42)

train_data = batchify(corpus.train, batch_size)
val_data = batchify(corpus.dev, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

model = RNNModel(
    model_type, ntokens, emsize, nhid, nlayers, dropout, tied).to(device)

criterion = nn.NLLLoss()

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

