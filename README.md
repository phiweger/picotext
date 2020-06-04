## README

train Mgnify

then finetune on uniprot SPROT (well curated)

then predict small molecules

skin microbiome interesting, in general outer/ inner surfaces


### Installation

```bash
# Install dependencies
sh install.sh
# Get data
osf -p pefs7 clone
```



### TODO

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

