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
    '''
    https://github.com/pytorch/examples/blob/master/word_language_model/main.py#L68
    '''
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def repackage_hidden(h):
    '''
    Wraps hidden states in new Tensors, to detach them from their history.
    '''
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i):
    '''
    https://github.com/pytorch/examples/blob/master/word_language_model/main.py#L119
    '''
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    # .view(-1) concatenates the target vectors bc/ downstream this is what the
    # criterion expects (the output of the model is of equal dimensions, albeit
    # its input is still batched up when it leaves the get_batch() fn.
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


# def accuracy_score(y_true, y_pred):
#     '''
#     stackoverflow.com/questions/43962599
#     '''
#     y_pred = np.concatenate(tuple(y_pred))
#     y_true = np.concatenate(tuple([[t for t in y] for y in y_true])).reshape(y_pred.shape)
#     return (y_true == y_pred).sum() / float(len(y_true))


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
        
        loss = criterion(output, targets)  # y_pred, y_true
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # TODO: Is this the optimizer.step() part? We subtract the lr from the
        # gradients.
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


# Let's go
# corpus = Corpus(
#     sequences='uniprot_sprot.fasta.gz',
#     tokenizer='uniprot_sprot.dayhoff',
#     split=[0.8, 0.1, 0.1], seed=42)

# !zcat < uniprot_sprot.fasta.gz | head -n50000 > redux.fasta
corpus = Corpus(
    sequences='redux',
    tokenizer='uniprot_sprot.dayhoff',
    split=[0.8, 0.1, 0.1], seed=42)



device = 'cpu'
# preprocessing
batch_size = 20  # TODO: todal seq len bptt * batch_size?
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
epochs = 5
bptt = 50  # 35
clip = 0.25
log_interval = 50
save = 'final'


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


'''
Eval redux:

nlayers = 2

bptt 50 5.49
bptt 35 5.53 x 5.54 (repeat)
bptt 20 5.64

nlayers = 4

bptt 50
bptt 35 
bptt 20 
'''


# Load the best saved model.
with open(save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
    if model_type in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        model.rnn.flatten_parameters()


test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)


'''
https://discuss.pytorch.org/t/got-warning-couldnt-retrieve-source-code-for-container/7689/5

| epoch   1 | 48400/48492 batches | lr 20.00 | ms/batch 205.63 | loss  7.93 | ppl  2766.47
-----------------------------------------------------------------------------------------
| end of epoch   1 | time: 10171.76s | valid loss  7.84 | valid ppl  2545.27
-----------------------------------------------------------------------------------------
/Users/phi/miniconda3/envs/redux/lib/python3.7/site-packages/torch/serialization.py:360: UserWarning: Couldn't retrieve source code for container of type RNNModel. It won't be checked for correctness upon loading.
  "type " + obj.__name__ + ". It won't be checked "
| epoch   2 |   200/48492 batches | lr 20.00 | ms/batch 210.18 | loss  7.92 | ppl  2756.23
'''


# TODO: accuracy
# https://discuss.pytorch.org/t/training-accuracy-do-not-increase-when-train-lstm-with-batchsize-1/7578
# --> https://www.analyticsvidhya.com/blog/2020/01/first-text-classification-in-pytorch/

# TODO: optimizer, where?
# https://machinetalk.org/2019/02/08/text-generation-with-pytorch/
# https://www.analyticsvidhya.com/blog/2020/01/first-text-classification-in-pytorch/

# TODO: padded sequence
# https://gist.github.com/bentrevett/0580ff2cb8c14773e475d7f1e065b1a4
# https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec
# https://gist.github.com/Tushar-N/dfca335e370a2bc3bc79876e6270099e
# https://discuss.pytorch.org/t/understanding-pack-padded-sequence-and-pad-packed-sequence/4099
# https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch

# TODO: modify
# https://github.com/fastai/course-v3/blob/master/nbs/dl2/12_text.ipynb

# TODO; LSTM explained
# http://colah.github.io/posts/2015-08-Understanding-LSTMs/

# TODO: deploy on Floydhub

# TODO: bptt lower

# TODO: tokenizers -- how save state?

# TODO: use transformer? https://huggingface.co/blog/how-to-train

# TODO: provision floydhub nicely
# https://docs.floydhub.com/guides/jobs/installing_dependencies/
# https://docs.floydhub.com/guides/jobs/installing_dependencies/#installing-non-python-dependencies

'''
> Why do I need to reinstall the package for every time I run the workspace?
Unfortunately, the packages you install in a Job  or Workspaces is not preserved when the run finishes. We are currently planning to support preserving of environment changes in the near future.
-- https://help.floydhub.com/en/articles/2437328-how-can-i-install-python-packages-for-my-project


# TODO: eval task -- generate proteins and then mmseqs2 try to find them in training set at 50% or so identity -- how eval mofrad? maybe take 2-grams/ 3-grams etc and see if we can find them in the original sequences

# TODO: write tests and run them

# TODO: better loading and DataLoader as in here
# https://huggingface.co/blog/how-to-train
# Also try their approach to masked language modelling (BERT-like)
# 31:20 "chunks" in pandas https://www.youtube.com/watch?v=h5Tz7gZT9Fo&t=4191s%5D

# TODO: learn how to generate text
# https://huggingface.co/blog/how-to-generate

# TODO: create a table to store hyperparams used in the experiments as well as loss an time trained

Let's try a reduced set, can we overfit?
head -n 20000 tmp | grep -v ">" > redux

# TODO: vis encoding
model.encoder.state_dict()

# TODO: generate text
https://machinetalk.org/2019/02/08/text-generation-with-pytorch/
https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
https://www.kaggle.com/ab971631/beginners-guide-to-text-generation-pytorch

# TODO:
pytorch vector shapes

# TODO: test fn esp generate on words first where we can actually say whether
# the learned thing is sensical

# TODO: use parallelism
https://pytorch.org/docs/master/notes/faq.html#pack-rnn-unpack-with-data-parallelism


# misc
# https://www.curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/

# learn about learning rate
https://www.jeremyjordan.me/nn-learning-rate/

'''


def generate(prime_str='eeddcbee', predict_len=100, temperature=0.8):
    '''
    generate('ef')
    '''
    prime_len = 2
    hidden = model.init_hidden(prime_len)#.cuda()

    for p in range(predict_len):
        
        prime_input = torch.tensor(
            tokenizer.encode(prime_str).ids, dtype=torch.long)#.cuda()
        
        # STUCK HERE
        inp = prime_input[-prime_len:].unsqueeze(0) #last two words as input
        # output, hidden = model(inp, hidden)
        # output, hidden = model(train_data[0][:2].unsqueeze(0), hidden)
        output, hidden = model(inp, hidden)

        # Sample from the network as a multinomial distribution
        output_dist = output[0].view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        predicted_word = tokenizer.decode([top_i])
        prime_str += " " + predicted_word
    
    return prime_str

        # Add predicted word to string and use as next input

        # predicted_word = list(word_to_ix.keys())[list(word_to_ix.values()).index(top_i)]
        # prime_str += " " + predicted_word

#         inp = torch.tensor(word_to_ix[predicted_word], dtype=torch.long)



