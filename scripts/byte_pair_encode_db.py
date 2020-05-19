



# https://github.com/huggingface/tokenizers/tree/master/bindings/python#train-a-new-tokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
# Initialize a tokenizer
tokenizer = Tokenizer(models.BPE())
# Customize pre-tokenization and decoding
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# TODO True

tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
trainer = trainers.BpeTrainer(vocab_size=10000, min_frequency=2)
tokenizer.train(trainer, ['bar'])

encoded = tokenizer.encode(seq)
print(encoded.tokens)

# TODO: Use a clustered set of proteins like UniRef50
# -- https://www.uniprot.org/help/uniref

# TODO: Use an LSTM to train on sequences, then freeze early layers and add
# classification backend, retrain.



# https://github.com/huggingface/tokenizers/tree/master/bindings/python
# https://github.com/huggingface/tokenizers/tree/master/bindings/python#provided-tokenizers
from tokenizers import CharBPETokenizer

tokenizer = CharBPETokenizer(bert_normalizer=False)
tokenizer.train(['./bar'], vocab_size=1000, min_frequency=2)
# tokenizer.encode(seq).tokens

encoded = tokenizer.encode(seq)
a0 =encoded.ids
a1 = encoded.tokens


tokenizer.save('.', 'mytoken3')
# ['./mytoken-vocab.json', './mytoken-merges.txt']



from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors

# Load a BPE Model
vocab = './mytoken3-vocab.json'
merges = './mytoken3-merges.txt'
bpe = CharBPETokenizer(vocab, merges)

# Initialize a tokenizer
encoded = tokenizer.encode(seq)
b0 = encoded.ids
b1 = encoded.tokens

assert a0 == b0
assert a1 == b1



def load_pretrained_tokenizer(tokenizer_type, path_prefix):
    '''
    Load a pretrained tokenizer.

    Usage:

    from tokenizers import CharBPETokenizer
    seq = 'eeeefebeeb'
    tokenizer = load_pretrained_tokenizer(CharBPETokenizer, 'path/to/mytoken')
    encoded = tokenizer.encode(seq)
    '''
    vocab = f'{path_prefix}-vocab.json'
    merges = f'{path_prefix}-merges.txt'
    return tokenizer_type(vocab, merges)



