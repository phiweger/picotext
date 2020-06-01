'''
https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb

test_loss, test_acc = evaluate(model, test_iter, loss_fn)
print(round(test_loss, 4), round(test_acc, 4))
0.0805 0.9684

tensorboard --logdir .
'''


import torch
import torch.nn as nn
from torch.autograd import Variable
from torchtext import data

# > The first difference is that we do not need to set the dtype in 
# the LABEL field. --
# https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/5%20-%20Multi-class%20Sentiment%20Analysis.ipynb
# Two classes: dtype=torch.float


batch_size = 1000


TEXT = data.Field(sequential=True, include_lengths=True)
'''
> TorchText Fields have a preprocessing argument. A function passed here will be applied to a sentence after it has been tokenized (transformed from a string into a list of tokens), but before it has been numericalized (transformed from a list of tokens to a list of indexes). This is where we'll pass our generate_bigrams function.
'''
LABELS = data.LabelField(dtype=torch.float)
NAMES = data.RawField(is_target=False)

# Fields are added by column left to write in the underlying table
fields=[('name', NAMES), ('label', LABELS), ('text', TEXT)]

train, dev, test = data.TabularDataset.splits(
    path='tmp/processed', format='CSV', fields=fields,
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
# LABELS.vocab.freqs
# Counter({'toxin': 6271, 'negative': 37514})
'''
a = train.examples[0]
a.name ... 'sp_tox|F8QN53|PA2A2_VIPRE'
a.label ... 'toxin'
a.text[:4] ... ['edee', 'feebe', 'aeebe', 'cbcef']
'''


cnt = dict(LABELS.vocab.freqs)
baseline = round(1 - cnt['toxin'] / sum(cnt.values()), 4)
print(f'Baseline accuracy is {baseline}')

# iterators etc explained
# https://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/
# http://mlexplained.com/2018/02/15/language-modeling-tutorial-in-torchtext-practical-torchtext-part-2/


# for batch in train_iter:
#     break
#     print(batch.text[1]) # ... works, lengths grouped


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            text, text_lengths = batch.text
            
            predictions = model(text, text_lengths).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def learn(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        text, text_lengths = batch.text
        
        predictions = model(text, text_lengths).squeeze(1)
        
        loss = criterion(predictions, batch.label)
        
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)



'''
For multiclass we map to n_classes neurons in the last layer and then pass this
to the CrossEntropyLoss, basically taking argmax to get the most likely class.

With two classes we simply map to one neuron.
'''
class Model(nn.Module):
    '''
    https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb

    The following comments explain why we have ...

    self.linear = nn.Linear(hidden_size, classes_no)

    ... in our model definition:

    https://discuss.pytorch.org/t/example-of-many-to-one-lstm/1728/4

    > If you need a fixed number of output features you either set hidden_size to that value, or you add an output layer that maps from hidden_size to your output space

    https://discuss.pytorch.org/t/lstm-for-many-to-one-multiclass-classification-problem/14268/3

    > You may need more hidden units in the LSTM layers. In which case you would need to add a Linear layer to squeeze the last_output down to the right size for the number of classes.

    > Are you suggesting that I take the last output from the lstm layer, and add a linear layer to output neuron size will be the size of the class (here 3) and squeeze (using softmax) to predict the label?

    > Yep, exactly that ... Something like this should work:
    '''
    def __init__(self, vocab_size, emb_size, hidden_size, output_size, n_layers, bidirectional, dropout, pad_ix):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_ix)
        # https://stackoverflow.com/questions/50340016/pytorch-lstm-using-word-embeddings-instead-of-nn-embedding
        # https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
        self.rnn = nn.LSTM(emb_size, hidden_size, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        # fc .. fully connected
        '''
        > As the final hidden state of our LSTM has both a forward and a backward component, which will be concatenated together, the size of the input to the nn.Linear layer is twice that of the hidden dimension size.
        '''
        # https://discuss.pytorch.org/t/loss-function-for-binary-classification-with-pytorch/26460/2
        # self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)


    def forward(self, text, text_lengths):
        # text = [sent len, batch size]
        
        embedded = self.dropout(self.embedding(text))
        # embedded = [sent len, batch size, emb dim]

        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths)

        # nn.LSTM? needs **input** of shape `(seq_len, batch, input_size)`
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        # output = [sent len, batch size, hid dim]
        # hidden = [1, batch size, hid dim]

        output, output_lengths = nn.utils.rnn.pad_packed_sequence(
            packed_output)
        #output = [sent len, batch size, hid dim * num directions]
        #output over padding tokens are zero tensors
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]

        # hidden is simply the final hidden state
        # assert torch.equal(output[-1, :, :], hidden.squeeze(0))
        # last_output = output[-1]  # "many-to-one"
        # class_predictions = self.linear(last_output)
        # No sigmoid here, but in the loss fn
        # https://pytorch.org/docs/stable/nn.html#bcewithlogitsloss
        # x = self.sigmoid(class_predictions)
        # return class_predictions

        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout
        hidden = self.dropout(
            torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        #hidden = [batch size, hid dim * num directions]

        return self.fc(hidden)


log_every = 1

# model = Model(
#     vocab_size=len(TEXT.vocab), 
#     emb_size=100,
#     hidden_size=100,
#     output_size=1)
# classes_n = len(LABELS.vocab)


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = Model(INPUT_DIM, 
            EMBEDDING_DIM, 
            HIDDEN_DIM, 
            OUTPUT_DIM, 
            N_LAYERS, 
            BIDIRECTIONAL, 
            DROPOUT, 
            PAD_IDX)


print(f'The model has {count_parameters(model):,} trainable parameters')

'''
> As our <unk> and <pad> token aren't in the pre-trained vocabulary they have been initialized using unk_init (an $\mathcal{N}(0,1)$ distribution) when building our vocab. It is preferable to initialize them both to all zeros to explicitly tell our model that, initially, they are irrelevant for determining sentiment. We do this by manually setting their row in the embedding weights matrix to zeros. We get their row by finding the index of the tokens, which we have already done for the padding index. Note: like initializing the embeddings, this should be done on the weight.data and not the weight!
'''
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 3e-4

# loss_fn = nn.CrossEntropyLoss()  # multiclass
# > This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
# https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html

loss_fn = nn.BCEWithLogitsLoss()
# > This loss combines a Sigmoid layer and the BCELoss in one single class.   
# https://pytorch.org/docs/master/generated/torch.nn.BCEWithLogitsLoss.html
# https://discuss.pytorch.org/t/loss-function-for-binary-classification-with-pytorch/26460/2

# max_len = 100
# A reasonable limit of 250-500 time steps is often used in practice with large LSTM models.
# https://machinelearningmastery.com/handle-long-sequences-long-short-term-memory-recurrent-neural-networks/

# https://pytorch.org/docs/stable/tensorboard.html
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='log')


n_iter = 0
for epoch in range(5):

    model.train()

    for batch in train_iter:
        n_iter += 1
        y = batch.label  # the BPE loss needs float
        '''
        Many sequences are 100 tokens long and longer. These long-range
        interactions are hard for RNN to keep pass down the chain. For now,
        we will simply use the first n tokens.
        '''
        # x = batch.text
        text, text_lengths = batch.text
        #x = batch.text[:30,:]
        # batch.text[:50, -2:]
        # a batch w/ two sequences
        # loss_fn(model(batch.text[:50, -2:]).squeeze(1), batch.label[-2:].float())

        # if text_lengths[0].item() > max_len:
        #     predictions = model(
        #         text[:max_len, :], torch.Tensor(batch_size * [max_len]).long()).squeeze(1)
        # else:
        #     predictions = model(text, text_lengths).squeeze(1)
        predictions = model(text, text_lengths).squeeze(1)

        loss = loss_fn(predictions, y)
        acc = binary_accuracy(predictions, y)

        print(
            text_lengths[0].item(),
            round(loss.item(), 4),
            round(acc.item(), 4))
        writer.add_scalar('Loss/ train', round(loss.item(), 4), n_iter)
        writer.add_scalar('Accuracy/ train', round(acc.item(), 4), n_iter)
        
        # Zero the gradients before backprop
        # stackoverflow.com/questions/48001598
        # https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-nn
        optimizer.zero_grad()
        loss.backward()

        # clip gradients
        # https://stackoverflow.com/questions/54716377
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()

    dev_loss, dev_acc = evaluate(model, dev_iter, loss_fn)
    if epoch % log_every == 0:
        print(
            epoch,
            round(loss.item(), 4),
            round(acc.item(), 4),
            round(dev_loss, 4),
            round(dev_acc, 4))  


test_loss, test_acc = evaluate(model, test_iter, loss_fn)
print(round(test_loss, 4), round(test_acc, 4))



N_EPOCHS = 5

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iter, optimizer, loss_fn)
    dev_loss, dev_acc = evaluate(model, dev_iter, loss_fn)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if dev_loss < best_dev_loss:
        best_dev_loss = dev_loss
        torch.save(model.state_dict(), 'picotext.model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {dev_loss:.3f} |  Val. Acc: {dev_acc*100:.2f}%')