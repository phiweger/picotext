'''
https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb

load pretrained

https://stackoverflow.com/questions/49710537/pytorch-gensim-how-to-load-pre-trained-word-embeddings

'''
import time

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from picotext.data import Corpus


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def batchify(data, bsz):
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


def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")

            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            # hidden and cells
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            # only hidden
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


def train():
    # Turn on training mode which enables dropout.
    model.train()  # defaults to train anyway, here to make this explicit

    start_time = time.time()
    ntokens = len(corpus.dictionary)
    
    hidden = model.init_hidden(batch_size)
    
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        # To overfit one batch do
        # ... in [next(enumerate(range(0, train_data.size(0) - 1, bptt)))]
        data, targets = get_batch(train_data, i)
        
        optimizer.zero_grad()
        # model.zero_grad()
        
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        
        output, hidden = model(data, hidden)
        # TODO: Why is this the other way around than in the eval fn and does
        # it matter?
        
        loss = criterion(output, targets)



        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()

        # for p in model.parameters():
        #     p.data.add_(-lr, p.grad)

        # total_loss += loss.item()

        if batch % log_interval == 0:
            print(loss)
            # cur_loss = total_loss / log_interval
            # elapsed = time.time() - start_time
            # print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
            #         'loss {:5.2f} | ppl {:8.2f}'.format(
            #     epoch, batch, len(train_data) // bptt, lr,
            #     elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            # total_loss = 0
            # start_time = time.time()

    return loss.detach().item()


from collections import Counter

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.

    ntokens = len(corpus.dictionary)
    
    hidden = model.init_hidden(eval_batch_size)
    
    # bar = []

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            loss = criterion(output, targets)

            # How many words are actually accurately predicted?
            # https://thegradient.pub/understanding-evaluation-metrics-for-language-models/
            # https://forums.fast.ai/t/language-model-accuracy/19574
            # https://medium.com/@davidmasse8/using-perplexity-to-evaluate-a-word-prediction-model-8820cf3fd3aa#:~:text=Accuracy%20is%20quite%20good%20(44,despite%20a%20mode%20of%201.
            
            # topk = 10
            # for ix, j in enumerate(output):
            #     foo = j.argsort()[-topk:]
            #     bar.append(targets[ix] in foo)
            total_loss += len(data) * criterion(output, targets).item()

    # print(Counter(bar))
    loss_avg = total_loss / (len(data_source) - 1)
    # return output and loss of the last batch
    return loss_avg


'''
    >>> m = nn.LogSoftmax(dim=1)
    >>> loss = nn.NLLLoss()
    >>> # input is of size N x C = 3 x 5
    >>> input = torch.randn(3, 5, requires_grad=True)
    >>> # each element in target has to have 0 <= value < C
    >>> target = torch.tensor([1, 0, 4])
    >>> output = loss(m(input), target)
    >>> output.backward()

TODO: why .view(-1)


data, targets = get_batch(dev_data, 0)
output, hidden = model(data, hidden)

topk = 5
output[0].argsort()[-topk:]
tensor([126, 131, 245,  53,  87])
'''



# https://discuss.pytorch.org/t/how-to-properly-setup-pytorch-text-bpttiterator/36784

# https://github.com/pytorch/text/blob/master/test/language_modeling.py
# print batch information

# https://github.com/pytorch/text/issues/137

'''
cd .../picotext/journal/2020-05-23T1315/tmp/processed
for i in train dev test
do
    cut -d, -f3 ${i}.csv | head -n 1000 > ${i}.lm.txt
done
# | head -n 10000
'''
corpus = Corpus('.')
'''
!wc train.lm.txt
    43785 2983318 17233692 train.lm.txt
len(corpus.train)
    2983318

TODO: Later make sure to turn the same word into the same number btw/
models -- maybe save a json for the tokens and map <unk> and <pad> to
0 and 1 respectively

TEXT.vocab.itos[1] ... '<pad>'
TEXT.vocab.itos[0] ... '<unk>'

https://discuss.pytorch.org/t/aligning-torchtext-vocab-index-to-loaded-embedding-pre-trained-weights/20878/2
'''


batch_size = 50
eval_batch_size = 10
train_data = batchify(corpus.train, batch_size)
dev_data = batchify(corpus.dev, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

ntokens = len(corpus.dictionary)

bptt = 30
clip = 0.5
log_interval = 10
lr = 3e-4  #20
best_val_loss = None
epochs = 20
save = 'foo'

emsize = 100
nhid = 1024
nlayers = 2
dropout = 0.5
tied = False

model = RNNModel('GRU', ntokens, emsize, nhid, nlayers, dropout, tied).to(device)
criterion = nn.CrossEntropyLoss() #nn.NLLLoss()
'''
nn.NLLLoss?

The `input` given through a forward call is expected to contain
log-probabilities of each class.

Obtaining log-probabilities in a neural network is easily achieved by
adding a  `LogSoftmax`  layer in the last layer of your network.
You may use `CrossEntropyLoss` instead, if you prefer not to add an extra
layer.

nn.CrossEntropyLoss()

> This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class. -- https://pytorch.org/docs/stable/nn.html#crossentropyloss
'''
optimizer = torch.optim.Adam(model.parameters(), lr=lr) 


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='log')

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        train_loss = train()
        val_loss = evaluate(dev_data)
        
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        writer.add_scalar('Loss/ train', round(train_loss, 4), epoch)
        writer.add_scalar('Loss/ dev', round(val_loss, 4), epoch)
        # Save the model if the validation loss is the best we've seen so far.
        # if not best_val_loss or val_loss < best_val_loss:
        #     with open(save, 'wb') as f:
        #         torch.save(model, f)
        #     best_val_loss = val_loss
        # else:
        #     # Anneal the learning rate if no improvement has been seen in the validation dataset.
        #     lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')




test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)


# TODO:
'''
            # topk = 10
            # for ix, j in enumerate(output):
            #     foo = j.argsort()[-topk:]
            #     bar.append(targets[ix] in foo)
'''


# tensorboard --logdir .
# in floydhub https://docs.floydhub.com/guides/jobs/tensorboard/
