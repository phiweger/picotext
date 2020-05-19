from tokenizers import CharBPETokenizer


def train_tokenizer(tokenizer, files: list, **kwargs):
    '''
    https://github.com/huggingface/tokenizers/tree/master/bindings/python

    Usage:

    # Encode proteins in Dayhoff encoding, one sequence per line
    files = ['./uniprot_sprot.dayhoff.txt']
    tokenizer = train_tokenizer(
        CharBPETokenizer(bert_normalizer=False),
        files,
        vocab_size=10000, min_frequency=2)
    tokenizer.save('.', 'uniprot_sprot.dayhoff')
    '''
    tokenizer.train(files, **kwargs)
    return tokenizer


def load_pretrained_tokenizer(tokenizer_type, path_prefix):
    '''
    Load a pretrained tokenizer.

    Usage:

    from tokenizers import CharBPETokenizer
    seq = 'eeeefebeeb'
    tokenizer = load_pretrained_tokenizer(
        CharBPETokenizer, './uniprot_sprot.dayhoff')
    encoded = tokenizer.encode(seq)
    '''
    vocab = f'{path_prefix}-vocab.json'
    merges = f'{path_prefix}-merges.txt'
    return tokenizer_type(vocab, merges)


def load_sequences(path):
    '''
    Load the sequences and index them if no index has been computed before.
    '''
    if not Path(path + '_screed').is_file():
        print('Indexing sequences ...')
        screed.make_db(path)
    return ScreedDB(path)


def encode_dayhoff(seq):
    '''
    Turn a protein sequence into its corresponding Dayhoff encoding. Return
    None if the encoding is not unique (for details on this see comment in
    scripts/encoding.py)

    https://en.wikipedia.org/wiki/Margaret_Oakley_Dayhoff

    Usage:

    seq = 'MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVEC'
    encode_dayhoff(seq)
    # 'ebfbbcceedcfcddddecbeeebeffbccddeecfdcfbbbdececa'
    '''
    dayhoff = {
        'C' + 'U': 'a', 
        'GSTAP': 'b',
        'DENQ' + 'Z': 'c',
        'RHK' + 'O': 'd',
        'LVMI' + 'J': 'e',
        'YFW': 'f'}

    encoding = ''

    try:
        for letter in seq:
            k_ = [k for k in dayhoff if letter in k][0]
            encoding += dayhoff[k_]
        return encoding
    
    except IndexError:
        return None
