import torch
import torch.nn as nn


class RNN_base(nn.Module):
    '''
    Container module with an encoder and a recurrent module.

    https://github.com/pytorch/examples/tree/master/word_language_model

    The decoder is added in inheriting classes. The aim is to allow us to 
    "swap heads" of the RNN in the context of transfer learning: First, a
    language model is trained where we predict the next word, then we replace
    decoder and forward fn, and use the pretrained weights to predict labels.
    '''
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers=2, dropout=0.5, tie_weights=False):
        super(RNN_base, self).__init__()
        self.ntoken = ntoken
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.tie_weights = tie_weights

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

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            # hidden and cells
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            # only hidden
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
        


class RNN_lm(RNN_base):
    '''
    Define a 

    - decoder
    - forward fn
    '''
    def __init__(self, init_args):
        super(RNN_lm, self).__init__(**init_args)

        self.decoder = nn.Linear(self.nhid, self.ntoken)
        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if self.tie_weights:
            if self.nhid != self.ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
        self.init_weights()

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return torch.log_softmax(decoded, dim=1), hidden


class RNN_tr(RNN_base):
    '''
    Define a 

    - decoder
    - forward fn
    '''
    def __init__(self, init_args, nclass=2, pretrained_model=None):
        super(RNN_lm, self).__init__(**init_args)

        # We now predict one out of n classes, not one out of n tokens
        # as in the language model RNN
        self.nclass = nclass

        if pretrained_model:
            self.load_state_dict(pretrained_model.state_dict())

        self.decoder = nn.Linear(nhid, nclass)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output[-1, :, :])  # predict w/ last hidden state
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.nclass)
        
        return torch.sigmoid(decoded), hidden
        # multiclass 
        # return torch.log_softmax(decoded, dim=1), hidden

