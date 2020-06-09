'''
TODO:

- add subsample arg to create reduced test set

Aim:

Read in 2 fasta files, one holding positive examples of some sequence, and another holding negative examples.

Create a single fasta w/ a sensical header, e.g.

>id::label
sequence

Then split them into train/ dev/ test sets.

Tokenize.

Reshape so it can enter the NN.
'''
from collections import Counter
from pathlib import Path

import numpy as np
import screed
import torch
from tqdm import tqdm

from picotext.utils import encode_dayhoff


'''
mkdir -p tmp/data
mkdir tmp/processed
cp ToxClassifier/datasets/trainingsets/* tmp/data
'''


# TODO: Allow standard (IUPAC) amino acid code as well as Dayhoff
translate = True


splits = [0.8, 0.1, 0.1]
reformat = lambda x: x.split()[0]
outdir = 'tmp/processed'
labels = {
    'toxin': 'tmp/data/dataset_pos.fa',
    'negative': 'tmp/data/dataset_easy.fa'}


with open(Path(outdir) / 'train.csv', 'w+') as train_out, \
     open(Path(outdir) / 'dev.csv', 'w+') as dev_out, \
     open(Path(outdir) / 'test.csv', 'w+') as test_out:

    for label, path in labels.items():
        cnt = []
    
        with screed.open(path) as file:
            for read in file:
                if reformat:
                    name = reformat(read.name)
                    # sp|Q15WI5|QUEA_PSEA6 S-adeno ...
                else:
                    name = read.name
    
                throw = np.random.multinomial(1, splits)
                # [0, 0, 1] -- now in which position is 1?
                group = ['train', 'dev', 'test'][list(throw).index(1)]
                # groups[name] = group
                cnt.append(group)
    
                if translate:
                    sequence = encode_dayhoff(read.sequence)
                    if not sequence:
                        continue
                else:
                    sequence = read.sequence

                str_ = f'{name},{label},{sequence}\n'

                if group == 'train':
                    train_out.write(str_)
                elif group == 'dev':
                    dev_out.write(str_)
                elif group == 'test':
                    test_out.write(str_)
                else:
                    raise ValueError('A very specific bad thing happened.')
    
        print(label, Counter(cnt))

# from collections import Counter
# print(Counter(cnt))
