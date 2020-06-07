import screed
from tokenizers import CharBPETokenizer
from tqdm import tqdm
'''
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


# path = 'uniprot_sprot.fasta.gz'
path = 'uniref50.fasta.gz'
# !zcat < uniref50.fasta.gz | grep ">" | wc -l
# 39,992,926
maxlen = 2000
# aa .. amino acids
excluded_aa = 'BXZJUO'
# excluded_aa = 'BX'
# excluded_aa = 'XBZJ'
take, notake = 0, 0


with screed.open(path) as file:
    for record in tqdm(file):
        # TODO: sample p sequences here, w/ 0 < p < 1

        ex = [1 for i in record.sequence if i in excluded_aa]
        if (len(seq) < maxlen) and (not any(ex)):
            take += 1
        else:
            notake += 1

        if notake + take > 1e6:
            discard = round(notake / (notake + take), 4)
            print(f'Discarded {discard}')


# 2000, BX     -- 0.0317
# 2000, BXZJUO -- 0.0327


'''
In chunks increase size (1 mio seqs?) and train tokenizer then jaccard distance -- upper triangular all-vs-all.
'''
# TODO: write files to this fake file handle
tokenizer = train_tokenizer(
    CharBPETokenizer(bert_normalizer=False),
    files,
    vocab_size=10000, min_frequency=2)


# TODO: Jaccard

