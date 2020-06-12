'''
TODO:

- avg training loss
- baseline? ie what is the prediction 
'''
from tokenizers import CharBPETokenizer
import torch
import torch.nn as nn
from torchtext import data

from picotext.model import RNN_lm, RNN_tr
from picotext.utils import batchify, get_batch, repackage_hidden
from picotext.utils import load_pretrained_tokenizer, load_config


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TODO: should work w/o pretraining


'''
We trained the LM w/o padding, but for classification we will. Note that the RNN is length independent, i.e. it is a (stack of) hidden layers unrolled along the sequence. So for classification, we should be able to use padding.
'''


def train_fn():
    # Turn on training mode which enables dropout.
    model.train()  # defaults to train anyway, here to make this explicit
    total_loss, n = 0., 0
    hidden = model.init_hidden(batch_size)
    
    for i, batch in enumerate(train_iter):
        # print(batch.text[0].shape)
        # print(hidden.shape)
        if len(batch) != batch_size:
            print('Damn')
            continue

        # The overflow examples' batch is smaller, ignore. Otherwise creates
        # a RuntimeError. Known problem w/ data.BucketIterator():
        # https://github.com/pytorch/text/issues/640
        # https://github.com/pytorch/text/issues/438
        # stackoverflow.com/questions/54307824

        # print(batch.text[0].shape)
        # if batch.text[0].shape[-1] != 100:
        #     break
        # To overfit one batch do
        # ... in [next(enumerate(range(0, train_data.size(0) - 1, bptt)))]
        # Then revert
        # ... in enumerate(range(0, train_data.size(0) - 1, bptt))
        data_, targets = batch.text[0], batch.label
        # print(data, targets)
        # print(len(data[:, 0]))
        optimizer.zero_grad()
        # model.zero_grad()
        
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)

        '''
        TODO: the vocab is built from the training set but a minimal one and so
        we can encounter words that are not present in the embedding lookup table
        
        IndexError: index out of range in self
        
        https://discuss.pytorch.org/t/embeddings-index-out-of-range-error/12582/4
        https://stackoverflow.com/questions/50747947/embedding-in-pytorch
        
        once we use the full data this should not be an issue
        '''        
        #try:
        output, hidden = model(data_, hidden)
        #except IndexError:
        #    continue


        loss = criterion(output.squeeze(1), targets)

        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        total_loss += loss.item()
        # for p in model.parameters():
        #     p.data.add_(-lr, p.grad)

        # total_loss += loss.item()

        if (i % log_interval == 0) and (i != 0):
            print(epoch, i, round(total_loss / log_interval, 4))
            total_loss = 0.
            # cur_loss = total_loss / log_interval
            # elapsed = time.time() - start_time
            # print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
            #         'loss {:5.2f} | ppl {:8.2f}'.format(
            #     epoch, batch, len(train_data) // bptt, lr,
            #     elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            # total_loss = 0
            # start_time = time.time()


def evaluate():
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss, total_acc, n = 0., 0., 0

    hidden = model.init_hidden(batch_size)
    
    with torch.no_grad():
        for i, batch in enumerate(dev_iter):
            if len(batch) != batch_size:
                print('Damn')
                continue
            
            data_, targets = batch.text[0], batch.label
            output, hidden = model(data_, hidden)
            hidden = repackage_hidden(hidden)
            
            # loss = criterion(output, targets)
            total_loss += len(data_) * criterion(output.squeeze(1), targets).item()
            n += len(batch)
            total_acc += binary_accuracy(output.squeeze(1), targets).item()

    loss_avg = round(total_loss / n, 4)
    acc_avg = round(total_acc / n, 4)
    print('Dev loss:', loss_avg)
    print('Dev acc: ', acc_avg)
    print(torch.round(output).T, targets)

def binary_accuracy(preds, y):
    """
    slight modification from original

    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(preds)
    correct = (rounded_preds.T == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc


tokenizer = load_pretrained_tokenizer(CharBPETokenizer, '/Users/phi/Dropbox/repos_git/picotext/picotext/tokenizers/uniref50.full.dayhoff.vocab30k.freq5')
ntokens = len(tokenizer.get_vocab())
print(f'Found {ntokens} tokens')


nclass = 1

# Instead of ntokens we pass in nclass here
# init_args =['GRU', nclass, emsize, nhid, nlayers, dropout, tied]

batch_size = 250
# Can be a list w/ sizes for train, dev, test -- but we'd need to rewrite
# train and evaluate fn
bptt = 30
clip = 0.5
log_interval = 25
lr = 3e-4  #20
best_val_loss = None
epochs = 10
save = 'foo'
emsize = 100
nhid = 500
nlayers = 2
dropout = 0.5
tied = False
save = 'foo.model'

init_args = {
    'rnn_type': 'GRU',
    'ntoken': ntokens,
    'ninp': emsize,
    'nhid': nhid,
    'nlayers': nlayers,
    'dropout': dropout,
    'tie_weights': tied,
    }


# We have to use the same numericalization as in the example before.
TEXT = data.Field(
    sequential=True,
    include_lengths=True,
    use_vocab=True,
    tokenize=lambda x : tokenizer.encode(x).tokens)

LABELS = data.LabelField(dtype=torch.float, is_target=True)  # , is_target=True
NAMES = data.RawField(is_target=False)

# Fields are added by column left to write in the underlying table
fields=[('name', NAMES), ('label', LABELS), ('text', TEXT)]

train, dev, test = data.TabularDataset.splits(
    path='/Users/phi/Dropbox/projects/picotext/journal/2020-05-23T1315/tmp/processed', format='CSV', fields=fields,
    train='train.csv', validation='dev.csv', test='test.csv')

TEXT.build_vocab()  # We'll fill this w/ the tokenizer
# https://github.com/pytorch/text/issues/358
# TEXT.vocab.itos[1] ... '<pad>'
# TEXT.vocab.itos[0] ... '<unk>'
# TEXT.vocab.itos[:10]
# ['<unk>', '<pad>', 33, 1, 35, 43, 28, 32, 48, 45]
# Make sure the numericalisation is the same used in the tokenizer AND
# thus across models the numericalisation is the same.
TEXT.vocab.stoi = tokenizer.get_vocab()
# d = {k: v for k, v in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])}
'''
TODO: missing is the <pad> token
{'<unk>': 0,
 'a': 1,
  ...
 'f': 6,
 'e</w>': 7,
  ...
 'a</w>': 12,
'''
# Sort tokenizer dict by value, take keys, transfer to TEXT field.
TEXT.vocab.itos = [k for k, v in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])]
# What's in the bag? -- TEXT.vocab.itos[:10]
# ['<unk>', 'a', 'b', 'c', 'd', 'e', 'f', 'e</w>', 'c</w>', 'd</w>']
LABELS.build_vocab(train)


# Make sure the numericalisation is from the tokenizer, not random
a = [k for k, v in sorted(TEXT.vocab.stoi.items(), key=lambda item: item[1])]
b = [k for k, v in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])]
assert a == b


# https://github.com/pytorch/text/issues/641
train_iter, dev_iter, test_iter = data.BucketIterator.splits(
    (train, dev, test),
    batch_size=batch_size,
    # batch_sizes=(100, 100, 100),
    sort_key=lambda x: len(x.text),
    sort_within_batch=True,  # this really allows length bucketing
    device=device)
# BucketIterator will reorder the samples on each iteration, i.e. calling the
# following line twice will result in two reorderings of the samples.
for i in train_iter: pass


# Load model
# https://pytorch.org/tutorials/beginner/saving_loading_models.html
pretrained_model = torch.load('/Users/phi/data_local/picotext/models/language.45.model', map_location=torch.device('cpu'))
model = RNN_tr(init_args, nclass, pretrained_model).to(device)

# OR load a new model w/ random weights
# model_ft = RNN_tr(init_args)


# Freeze all layers
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#initialize-and-reshape-the-networks

'''
for name, param in model.named_parameters():
    if not name in ['decoder.weight', 'decoder.bias']:
        param.requires_grad = False
    print(f'{param.requires_grad}\t{name}')
'''

# TODO: thaw layers iteratively
# optimizer_ft.add_param_group?
criterion = nn.BCELoss().to(device)
# WithLogits?

# https://discuss.pytorch.org/t/understanding-nllloss-function/23702
# https://discuss.pytorch.org/t/cross-entropy-with-one-hot-targets/13580/4

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# TODO: scheduler

# https://github.com/davidtvs/pytorch-lr-finder/issues/49
# from torch_lr_finder import LRFinder
# 
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-2)
# lr_finder = LRFinder(model, optimizer, criterion, device="cpu")
# lr_finder.range_test(train, end_lr=100, num_iter=100)
# lr_finder.plot() # to inspect the loss-learning rate graph
# lr_finder.reset() # to reset the model and optimizer to their initial state



import pdb, traceback, sys

for epoch in range(1, epochs+1):
    # try:
    train_fn()
    evaluate()
    # except RuntimeError:
    #     extype, value, tb = sys.exc_info()
    #     traceback.print_exc()
    #     pdb.post_mortem(tb)

'''
~/miniconda3/envs/picotext/lib/python3.7/site-packages/torch/nn/modules/rnn.py in check_hidden_size(self, hx, expected_hidden_size, msg)
    185         # type: (Tensor, Tuple[int, int, int], str) -> None
    186         if hx.size() != expected_hidden_size:
--> 187             raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))
    188
    189     def check_forward_args(self, input, hidden, batch_sizes):

RuntimeError: Expected hidden size (2, 85, 100), got (2, 100, 100)
'''



# Now transfer these weights and train classifier

# https://discuss.pytorch.org/t/does-deepcopying-optimizer-of-one-model-works-across-the-model-or-should-i-create-new-optimizer-every-time/14359
# https://discuss.pytorch.org/t/transfer-learning-of-weights-to-one-model-to-another/23962
# https://discuss.pytorch.org/t/copy-weights-only-from-a-networks-parameters/5841/2?u=ptrblck

'''
so really we just copy weights, freeze stuff, get a new optimizer and go

http://seba1511.net/tutorials/beginner/transfer_learning_tutorial.html#finetuning-the-convnet
'''


'''
# Load pretrained weights
model2 = RNN_lm('GRU', ntokens, emsize, nhid, nlayers, dropout, tied).to(device)
model2.load_state_dict(model.state_dict())  # <All keys matched successfully>

# Change output layer
insize = model.decoder.in_features
model2.decoder = nn.Linear(insize, 1)
model2.forward = lambda x: print(x)

def foo(x):
    return x

model2.forward = foo


# https://discuss.pytorch.org/t/are-there-any-recommended-methods-to-clone-a-model/483/11
import copy
m2 = copy.deepcopy(model)
n_classes = 2
m2.decoder = nn.Linear(model.decoder.in_features, n_classes)




def forward(self, input, hidden):
    emb = self.drop(self.encoder(input))
    output, hidden = self.rnn(emb, hidden)
    output = self.drop(output)
    decoded = self.decoder(output)
    return 'whoop'
    # decoded = decoded.view(-1, self.ntoken)
    # return F.log_softmax(decoded, dim=1), hidden


m2.forward = forward
m2(m2, batch, hidden)
'''

'''
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
'''

# batch, targets = get_batch(train_data, 0)
# emb = model2.encoder(batch)
# hidden = model.init_hidden(batch_size)
# output, hidden = model2.rnn(emb, hidden)

# model2.decoder(output)[-1].shape

# https://discuss.pytorch.org/t/how-to-modify-a-pretrained-model/60509
# https://discuss.pytorch.org/t/modify-forward-of-pretrained-model/52530
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
# https://heartbeat.fritz.ai/transfer-learning-with-pytorch-cfcb69016c72
# https://brsoff.github.io/tutorials/beginner/transfer_learning_tutorial.html
# https://github.com/pytorch/tutorials/blob/master/beginner_source/transfer_learning_tutorial.py

# https://github.com/pytorch/tutorials/blob/master/beginner_source/transfer_learning_tutorial.py
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

# https://www.youtube.com/watch?v=K0j9AqcFsiw
# Python Pytorch Tutorials # 1 Transfer Learning : DataLoaders Pytorch

'''
See "Freezing the convolutional layers & replacing the fully connected layers with a custom classifier" -- https://heartbeat.fritz.ai/transfer-learning-with-pytorch-cfcb69016c72

They call it "Reshaping" the model:

> Now to the most interesting part. Here is where we handle the reshaping of each network. Note, this is not an automatic procedure and is unique to each model. -- https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#initialize-and-reshape-the-networks

Terminology:

- Language model == feature extraction
- CLassifying == finetuning

> Then the reinitialized layer’s parameters have .requires_grad=True by default. -- https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#initialize-and-reshape-the-networks
'''




# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
'''
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
'''


# TODO
def get_batch_with_labels(fp):
    '''
    Use tokenize method from Corpus? Advantage is we would be sure to
    use the same numericalization indices.
    '''
    pass


def train_ft():
    # Turn on training mode which enables dropout.
    model_ft.train()  # defaults to train anyway, here to make this explicit

    start_time = time.time()
    ntokens = len(corpus.dictionary)
    
    hidden = model_ft.init_hidden(batch_size)
    
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        # To overfit one batch do
        # ... in [next(enumerate(range(0, train_data.size(0) - 1, bptt)))]
        # Then revert
        # ... in enumerate(range(0, train_data.size(0) - 1, bptt))
        data_, targets = get_batch(train_data, i, bptt)
        
        optimizer_ft.zero_grad()
        # model.zero_grad()
        
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        
        output, hidden = model_ft(data_, hidden)
        '''
        model(batch, hidden)[0].shape
        torch.Size([3000, 9978])
        len(targets)
        3000
        '''
        
        # TODO: load targets
        targets = torch.randint(0, 2, [200]).view([100, 2]).float()
        loss = criterion_ft(output, targets)

        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer_ft.step()

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



# Freeze/ unfreeze
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#initialize-and-reshape-the-networks
# Send the model to GPU
model_ft = model_ft.to(device)

for epoch in range(1, epochs+1):
    train_loss = train_ft()
    print(train_loss)


# Manual walkthrough to check dimensions etc.
data, targets = get_batch(train_data, 0)
targets = torch.randint(0, 2, [200]).view([100, 2]).float()

emb = model_ft.encoder(data)

hidden = model_ft.init_hidden(batch_size)
hidden = repackage_hidden(hidden)
o, h = model_ft.rnn(emb, hidden)
decoded = model_ft.decoder(o)  # .shape is torch.Size([30, 100, 2])

torch.log_softmax(decoded.view(-1, nclass), dim=1).shape
# torch.Size([3000, 2])
torch.sigmoid(decoded.view(-1, nclass)).shape
# torch.Size([3000, 2])



'''
TODO: pad, allow 0s and unequal lengths

- https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html
- https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
- on the collate fn https://discuss.pytorch.org/t/batching-with-padded-sequences-and-pack-padded-sequence/65501/3

yes! how to get padded x and y:

- https://github.com/florijanstamenkovic/PytorchRnnLM/blob/master/main.py#L72

TODO: interesting

- https://github.com/PyTorchLightning/pytorch-lightning
- https://github.com/pytorch/ignite#why-ignite
- https://medium.com/pytorch/pytorch-lightning-0-7-1-release-and-venture-funding-dd12b2e75fb3
- https://github.com/williamFalcon/test-tube
- https://github.com/lanpa/tensorboardX
- https://github.com/williamFalcon/deep-learning-gpu-box-build-instructions

Secondary structure:

- https://www.uniprot.org/uniprot/Q9UXC2
- https://www.rcsb.org/pdb/protein/Q9UXC2
- https://www.rcsb.org/structure/2C38

'''





'''
We trained the LM w/o padding, but for classification we will. Note that the RNN is length independent, i.e. it is a (stack of) hidden layers unrolled along the sequence. So for classification, we should be able to use padding.
'''

from torchtext import data

batch_size = 1000  # can be a list w/ sizes for train, dev, test
TEXT = data.Field(sequential=True, include_lengths=True)
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

# https://github.com/pytorch/text/issues/641
train_iter, dev_iter, test_iter = data.BucketIterator.splits(
    (train, dev, test),
    batch_size=batch_size,
    # batch_sizes=(100, 100, 100),
    sort_key=lambda x: len(x.text),
    sort_within_batch=True,  # this really allows length bucketing
    device='cpu')

TEXT.build_vocab(train)
# TEXT.vocab.itos[1] ... '<pad>'
# TEXT.vocab.itos[0] ... '<unk>'
LABELS.build_vocab(train)


def train_ft():
    # Turn on training mode which enables dropout.
    model_ft.train()  # defaults to train anyway, here to make this explicit

    start_time = time.time()
    ntokens = len(corpus.dictionary)
    
    hidden = model_ft.init_hidden(batch_size)
    
    for i, batch in enumerate(train_iter):
        # To overfit one batch do
        # ... in [next(enumerate(range(0, train_data.size(0) - 1, bptt)))]
        # Then revert
        # ... in enumerate(range(0, train_data.size(0) - 1, bptt))
        data, targets = batch.text[0], batch.label
        # print(data, targets)
        # print(len(data[:, 0]))
        optimizer_ft.zero_grad()
        # model.zero_grad()
        
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        
        try:
            output, hidden = model_ft(data, hidden)
        except IndexError:
            continue

        '''
        TODO: the vocab is built from the training set but a minimal one and so
        we can encounter words that are not present in the embedding lookup table
        
        IndexError: index out of range in self
        
        https://discuss.pytorch.org/t/embeddings-index-out-of-range-error/12582/4
        https://stackoverflow.com/questions/50747947/embedding-in-pytorch
        
        once we use the full data this should not be an issue
        '''

        output = output.squeeze(1)
        '''
        model(batch, hidden)[0].shape
        torch.Size([3000, 9978])
        len(targets)
        3000
        '''
        
        # TODO: load targets
        # targets = torch.randint(0, 2, [200]).view([100, 2]).float()
        loss = criterion_ft(output, targets)

        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer_ft.step()

        # for p in model.parameters():
        #     p.data.add_(-lr, p.grad)

        # total_loss += loss.item()

        if i % log_interval == 0:
            print(epoch, i, loss)
            # cur_loss = total_loss / log_interval
            # elapsed = time.time() - start_time
            # print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
            #         'loss {:5.2f} | ppl {:8.2f}'.format(
            #     epoch, batch, len(train_data) // bptt, lr,
            #     elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            # total_loss = 0
            # start_time = time.time()

    return loss.detach().item()


init_args =['GRU', ntokens, emsize, nhid, nlayers, dropout, tied]
nclass = 1
model_ft = RNN_tr(model, nclass, init_args).to(device)

# model_ft = model_ft.to(device)
criterion_ft = nn.BCELoss()  # WithLogits
optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

log_interval = 1

for epoch in range(1, epochs+1):
    train_loss = train_ft()
    print(train_loss)



for batch in train_iter: pass
data, targets = batch.text[0], batch.label
output, hidden = model_ft(data, hidden)
output = output.squeeze(1)


