
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchtext import data

# > The first difference is that we do not need to set the dtype in 
# the LABEL field. --
# https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/5%20-%20Multi-class%20Sentiment%20Analysis.ipynb
# Two classes: dtype=torch.float

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


TEXT = data.Field(sequential=True)
LABELS = data.LabelField(dtype=torch.float)
NAMES = data.RawField(is_target=False)

# Fields are added by column left to write in the underlying table
fields=[('name', NAMES), ('label', LABELS), ('text', TEXT)]

train, dev, test = data.TabularDataset.splits(
    path='tmp/processed', format='CSV', fields=fields,
    train='train.csv', validation='dev.csv', test='test.csv')

TEXT.build_vocab(train)
# TEXT.vocab.itos[1] ... '<pad>'
# TEXT.vocab.itos[0] ... '<unk>'
LABELS.build_vocab(train)
# LABELS.vocab.freqs
# Counter({'toxin': 6271, 'negative': 37514})

# https://github.com/pytorch/text/issues/641
train_iter, dev_iter, test_iter = data.BucketIterator.splits(
    (train, dev, test),
    batch_sizes=(100, 100, 100),
    sort_key=lambda x: len(x.text),
    device='cpu')


'''
a = train.examples[0]
a.name ... 'sp_tox|F8QN53|PA2A2_VIPRE'
a.label ... 'toxin'
a.text[:4] ... ['edee', 'feebe', 'aeebe', 'cbcef']
'''











TEXT = data.Field(sequential=True)

# Fields are added by column left to write in the underlying table
fields=[('text', TEXT)]






train, dev, test = data.TabularDataset.splits(
    path='tmp/processed', format='CSV', fields=fields,
    train='train.lm.csv', validation='dev.lm.csv', test='test.lm.csv')

TEXT.build_vocab(train)

a, b, c = train_iter, dev_iter, test_iter = data.BPTTIterator.splits(
    (train, dev, test),
    batch_size=7,
    bptt_len=35,
    device=device,
    repeat=False)

batch = next(iter(a))
print(batch.text)
print(batch.target)





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
            x = batch.text
            y = batch.label.float()

            predictions = model(x).squeeze(1)
            
            loss = criterion(predictions, y)
            
            acc = binary_accuracy(predictions, y)

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
    def __init__(self, vocab_size, emb_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        # https://stackoverflow.com/questions/50340016/pytorch-lstm-using-word-embeddings-instead-of-nn-embedding
        # https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
        self.rnn = nn.RNN(emb_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        # fc .. fully connected
        # 2 classes, binary loss
        # https://discuss.pytorch.org/t/loss-function-for-binary-classification-with-pytorch/26460/2
        # self.sigmoid = nn.Sigmoid()


    def forward(self, input_seq):
        embedded = self.embeddings(input_seq)
        # embedded = [sent len, batch size, emb dim]

        # nn.LSTM? needs **input** of shape `(seq_len, batch, input_size)`
        output, hidden = self.rnn(embedded)  # _ .. hidden
        # output = [sent len, batch size, hid dim]
        # hidden = [1, batch size, hid dim]

        # hidden is simply the final hidden state
        assert torch.equal(output[-1, :, :], hidden.squeeze(0))

        # last_output = output[-1]  # "many-to-one"
        # class_predictions = self.linear(last_output)
        # No sigmoid here, but in the loss fn
        # https://pytorch.org/docs/stable/nn.html#bcewithlogitsloss
        # x = self.sigmoid(class_predictions)
        # return class_predictions

        return self.fc(hidden.squeeze(0))
        # remove batch dimension
        # torch.Size([1, 2, 2]) -> torch.Size([2, 2])


log_every = 1

model = Model(
    vocab_size=len(TEXT.vocab), 
    emb_size=100,
    hidden_size=100,
    output_size=1)
# classes_n = len(LABELS.vocab)



print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 3e-4

# loss_fn = nn.CrossEntropyLoss()  # multiclass
# > This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
# https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html

loss_fn = nn.BCEWithLogitsLoss()
# > This loss combines a Sigmoid layer and the BCELoss in one single class.   
# https://pytorch.org/docs/master/generated/torch.nn.BCEWithLogitsLoss.html
# https://discuss.pytorch.org/t/loss-function-for-binary-classification-with-pytorch/26460/2


for epoch in range(50):
    for batch in train_iter:
        y = batch.label  # the BPE loss needs float
        '''
        Many sequences are 100 tokens long and longer. These long-range
        interactions are hard for RNN to keep pass down the chain. For now,
        we will simply use the first n tokens.
        '''
        # x = batch.text
        x = batch.text[:30,:]
        # batch.text[:50, -2:]
        # a batch w/ two sequences
        # loss_fn(model(batch.text[:50, -2:]).squeeze(1), batch.label[-2:].float())
        predictions = model(x).squeeze(1)
        loss = loss_fn(predictions, y)
        acc = binary_accuracy(predictions, y)
        # Zero the gradients before backprop
        # stackoverflow.com/questions/48001598
        # https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-nn
        optimizer.zero_grad()
        loss.backward()
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






torch.argmax(a, dim=1)
loss_fn(, y.type(torch.FloatTensor))
# cast, to float, otherwise 
# RuntimeError: exp_vml_cpu not implemented for 'Long'
# https://discuss.pytorch.org/t/runtimeerror-exp-vml-cpu-not-implemented-for-long/49025
x.shape

vocab_size = len(TEXT.vocab)
emb = nn.Embedding(vocab_size, 5)
emb(x).shape

model = Model(emb_size, hidden_size, classes_n, vocab_size)  # 2
model(x)
loss_fn(model(x), y)


# https://github.com/pytorch/examples/blob/master/word_language_model/main.py
# https://github.com/pytorch/examples/blob/master/word_language_model/data.py
ids = [TEXT.vocab.stoi[token] for token in train.examples[0].text]
t = torch.tensor(ids).type(torch.int64)  # more: torch.cat(...)
device = 'cpu'
# NameError: name 'device' is not defined
batchify(t, 2)

