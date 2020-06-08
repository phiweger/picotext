class RNN_lm(nn.Module):
    '''
    Container module with an encoder, a recurrent module, and a decoder.

    https://github.com/pytorch/examples/tree/master/word_language_model
    '''
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers=2, dropout=0.5, tie_weights=False):
        super(RNN_lm, self).__init__()
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
        return torch.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            # hidden and cells
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            # only hidden
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


class RNN_tr(RNN_lm):
    '''
    # Pretrain some model
    init_args =['GRU', ntokens, emsize, nhid, nlayers, dropout, tied]
    nclass = 2

    m2 = RNN_tr(model, nclass, init_args)
    output, hidden = m2(batch, hidden)

    # Weights sould be the same
    model.state_dict()['encoder.weight']
    m2.state_dict()['encoder.weight']

    # The new decoder is initialized randomly
    m2.state_dict()['decoder.weight']
    '''
    # TODO: do sth like if model load pretrained else init like RNN_lm
    def __init__(self, init_args, nclass=2, pretrained_model=None):
        # We pass the init args to the super class
        super(RNN_tr, self).__init__(**init_args)
        # We now predict one out of n classes, not one out of n tokens
        self.nclass = nclass
        if pretrained_model:
            self.load_state_dict(pretrained_model.state_dict())
        # Override decoder to reduce number of classes
        self.decoder = nn.Linear(self.decoder.in_features, nclass)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        # nn.GRU?
        # seq_len, batch, num_directions * hidden_size
        # E.g. torch.Size([30, 100, 100])

        # Make prediction based on last hidden state
        output = self.drop(output[-1, :, :])
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.nclass)
        # return torch.log_softmax(decoded, dim=1), hidden
        return torch.sigmoid(decoded), hidden
