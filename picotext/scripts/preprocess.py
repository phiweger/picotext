#! /usr/bin/env python

'''
This script takes a file of protein sequences in fasta/q format, subsamples them if needed, filters out sequences that are very long or contain noncanonical amino-acid symbols and encodes them to the Dayhoff alphabet. You're welcome.

> We removed proteins longer than 2,000 amino acids and records containing noncanonical amino-acid symbols (X, B, Z, J), randomly selected test and validation subsets for monitoring training (1% of the overall dataset each) and used the rest of the data (~24 million sequences) in training. -- Alley et al., "Unified rational protein engineering with sequence-based deep representation learning"

> [...] accessed 21 November 2018

> UniRef50 is built by clustering UniRef90 seed sequences that have at least 50% sequence identity to, and 80% overlap with, the longest sequence in the cluster.

> Suzek, B. E., Wang, Y., Huang, H., McGarvey, P. B. & Wu, C. H. UniRef
clusters: a comprehensive and scalable alternative for improving sequence
similarity searches. Bioinformatics 31, 926–932 (2015).

---

The "default" Dayhoff encoding does not cover some of the letters in the 
"Extended IUPAC" protein encoding, see e.g.

https://biopython.org/DIST/docs/api/Bio.Alphabet.IUPAC.ExtendedIUPACProtein-class.html

> Extended uppercase IUPAC protein single letter alphabet including X etc.
In addition to the standard 20 single letter protein codes, this includes:

[ ] B = "Asx"; Aspartic acid (R) or Asparagine (N)
[ ] X = "Xxx"; Unknown or 'other' amino acid
[x] Z = "Glx"; Glutamic acid (E) or Glutamine (Q)
[x] J = "Xle"; Leucine (L) or Isoleucine (I), used in mass-spec (NMR)
[x] U = "Sec"; Seleno__cysteine__
[x] O = "Pyl"; Pyrro__lysine__

We simply add them, where they map to a unique Dayhoff character ([x]):
'''
import argparse

import numpy as np
import screed
from tqdm import tqdm

from picotext.utils import encode_dayhoff


def clean(sequence, maxlen=2000, excluded_aa='XBZJ'):
    ex = [1 for i in sequence if i in excluded_aa]
    if (len(sequence) <= maxlen) and (not any(ex)):
        return True
    else:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Downsample some sequences')
    parser.add_argument('--seq', help='Downsample these sequences [fasta/q]')
    parser.add_argument('--out', help='Write output here')
    parser.add_argument('-p', default=1, type=float, help='Downsample to this fraction approx.')
    parser.add_argument('--maxlen', help='Skip sequences longer than this', default=2000, type=int)
    parser.add_argument('--skip-header', help='Skip fasta header', action='store_true')
    parser.add_argument('--excluded-aa', default='XBZJ', help='Exclude noncanonical amino-acid symbols')
    args = parser.parse_args()

    
    seen, sampled = 0, 0

    with screed.open(args.seq, 'r') as file, open(args.out, 'w+') as out:
        
        for read in tqdm(file):
            seen += 1
            x = np.random.uniform(0, 1, 1).item()
            
            if x < args.p:
                pass_ = clean(read.sequence, args.maxlen, args.excluded_aa)
                
                if pass_:
                    sampled += 1
                    dayhoff = encode_dayhoff(read.sequence)

                    if args.skip_header:
                        out.write(f'{dayhoff}\n')
                    else:
                        out.write(f'>{read.name}\n{dayhoff}\n')

    print(f'Sampled {round(100 * (sampled / seen), 1)}% of sequences')
    




