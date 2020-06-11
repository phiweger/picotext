## README

train Mgnify

then finetune on uniprot SPROT (well curated)

then predict small molecules

skin microbiome interesting, in general outer/ inner surfaces

Unirep:

- https://moalquraishi.wordpress.com/about/
- https://openreview.net/forum?id=SygLehCqtm
- https://moalquraishi.wordpress.com/2019/04/01/the-future-of-protein-science-will-not-be-supervised/
- https://moalquraishi.wordpress.com/2018/02/15/protein-linguistics/
- [Surge's Public Defense- Shared screen with speaker view](https://harvard.zoom.us/rec/play/ucIsf7qsrG43T4aduQSDB_UsW9XuffisgyIZ-PQJyU-zBiYHZ1b1YOdEYOA26wooeBB5t5dbwjY7B-5a?startTime=1588098937000)
- [Exploring amino acid functions in a deep mutational landscape](https://www.biorxiv.org/content/10.1101/2020.05.26.116756v1.full.pdf) -- this is like feature engineering, which ideally we want the model to do



### Installation

```bash
# Install dependencies
sh install.sh
# Get data
osf -p pefs7 clone
```


### Train tokenizer


```bash
for i in 1 0.5 0.25 0.125 0.0625 0.03125
do
    python preprocess.py --seq uniref50.fasta.gz --out uniref50.${i}.dayhoff.txt -p ${i} --skip-header --maxlen 2000 --excluded-aa XBZJ
done
# for p=1 -- uniref50.clean.dayhoff.txt
sort -R -o uniref50.clean.dayhoff.shuffle.txt uniref50.clean.dayhoff.txt
# wc -l yields 39,222,662 sequences
head -n 35000000 uniref50.clean.dayhoff.shuffle.txt > uniref50.clean.dayhoff.train.lm.txt
head -n 37000000 uniref50.clean.dayhoff.shuffle.txt | tail -n 2000000 > uniref50.clean.dayhoff.dev.lm.txt
tail -n 2222662 uniref50.clean.dayhoff.shuffle.txt > uniref50.clean.dayhoff.test.lm.txt
# 35000000 + 2000000 + 2222662 - 39222662 = 0

head -n100000 uniref50.clean.dayhoff.shuffle.txt > lm_redux/train.lm.txt
head -n110000 uniref50.clean.dayhoff.shuffle.txt > lm_redux/tmp
tail -n10000 lm_redux/tmp > lm_redux/dev.lm.txt
head -n120000 uniref50.clean.dayhoff.shuffle.txt > lm_redux/tmp
tail -n10000 lm_redux/tmp > lm_redux/test.lm.txt
```


```python
from tokenizers import CharBPETokenizer
from picotext.utils import train_tokenizer

files = ['uniref50.0p0625.dayhoff.txt']

tokenizer = train_tokenizer(
    CharBPETokenizer(bert_normalizer=False),
    files,
    vocab_size=30000, min_frequency=5, special_tokens=['<unk>', '<pad>'])
tokenizer.save('.', 'uniref50.0p0625.dayhoff.vocab30k.freq5')
```


### Train

```bash
DATAVERSION=6
floyd run --gpu --data phiweger/datasets/lm_redux/${DATAVERSION}:data --mode job --env pytorch-1.4 --message "lm redux" --max-runtime 10000 --follow "\
    pip install --upgrade pip && \
    pip install screed tqdm tokenizers==0.7.0 && \
    git clone https://github.com/phiweger/picotext && \
    pip install git+https://github.com/phiweger/picotext && \
    cp picotext/picotext/models/RNN_LM/main_lm.py . && \
    python main_lm.py --config /data/config.json"
```





### TODO


Use other pretrained sequences, e.g. "Unirep" as coded for JAX [here](https://github.com/ElArkk/jax-unirep) -- supposed to be much faster and Tensorflow independent.


```bash
conda create -y -n unirep python=3.7 ipython
conda activate unirep
# https://github.com/google/jax#installation
pip install --upgrade pip
pip install --upgrade jax jaxlib  # CPU-only version
# https://github.com/ElArkk/jax-unirep
pip install tqdm optuna sklearn
pip install git+https://github.com/ElArkk/jax-unirep.git
```


```python
from jax_unirep import get_reps

sequence = "ASDFGHJKL"

# h_avg is the canonical "reps"
h_avg, h_final, c_final = get_reps(sequence)
h_avg.shape
# (1, 1900)
```




load data:

- json
- script: fasta > json


API:

```python
toxenize(seq, fn=[dayhoff, BPE, ...])
# Will tokenize seq in this order
```

Success metrics:

- accuracy in the prediction of the next token
- classification accuracy of helices and other known folds

https://academic.oup.com/peds/article/16/11/799/1508626

We could collect this as secondary structure annotation from the [PDB](https://www.rcsb.org/pdb/protein/Q12809?addPDB=1BYW). Browse [annotation](https://www.rcsb.org/search/browse/membrane). Maybe __mainly alpha vs. mainly beta fold__ [annotation](https://www.rcsb.org/search/browse/cath) as a good start to classification. Or something specific lika a [B-box zinc-binding domain](https://www.rcsb.org/search/browse/scop)


AMPs:

- [APD3: the antimicrobial peptide database](http://aps.unmc.edu/AP/main.php), 3185 proteins, `wget http://aps.unmc.edu/AP/APD_AMPs_fasta_DongChuan.fa`
- [DBAASP](https://dbaasp.org/), > 15k
- [CAMPR3](http://www.camp.bicnirrh.res.in/), > 8k
- [dbAMP](https://academic.oup.com/nar/article/47/D1/D285/5150231)

> Antimicrobial peptides (AMPs), naturally encoded from genes and generally contained 10–100 amino acids, are crucial components of the innate immune system and can protect the host from various pathogenic bacteria, as well as viruses. 

- [DRAMP 2.0, an updated data repository of antimicrobial peptides](https://www.nature.com/articles/s41597-019-0154-y), > 14k

> At present, the identified structural data is far from meeting the needs of researchers. In addition, due to the rapid development of computer technology, the newly identified sequences are exploding every year. Face this situation, structural prediction may be an appropriate way to bridge this gap26,27,28. Currently predicted structures or methods have been added to certain databases, such as: DBAASP29, AVPdb19, ParaPep20, SATPdb23. The 3D structure is generally predicted by homology modeling, which is mainly base on high homology between sequences30. Homology-modeled structures can be optimized by molecular dynamics and used to visualize the interaction between AMP and biomembrane31,32,33. In this paper, we used MOE2016 (https://www.chemcomp.com/) to establish an initial molecular structure by homology modeling, and Amber14 (http://ambermd.org/) to optimize the molecular structure by molecular dynamics, as previously reported34. In the second version of DRAMP, 82 predicted structures were added and these structures can be downloaded in pdb format (Fig. 3).

