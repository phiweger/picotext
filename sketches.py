import screed
from tqdm import tqdm

from tokenizers import CharBPETokenizer

from utils import load_pretrained_tokenizer


tokenizer = load_pretrained_tokenizer(
    CharBPETokenizer, './uniprot_sprot.dayhoff')

numbers = []
with open('uniprot_sprot.dayhoff.txt', 'r') as file:
    for line in file:
        encoded = tokenizer.encode(line.strip())
        tokens, ids = encoded.tokens, encoded.ids



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

corpus = Corpus('.')  # TODO: tokenization happens here
device = 'cpu'


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


eval_batch_size = 10
train_data = batchify(corpus.train, 5)





