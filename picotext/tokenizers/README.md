## README

`vocab10k.freq2`

```python
files = ['2020-05-29_uniref50.dayhoff.txt']

tokenizer = train_tokenizer(
    CharBPETokenizer(bert_normalizer=False),
    files,
    vocab_size=10000, min_frequency=2)
tokenizer.save('.', '2020-05-29_uniref50.dayhoff.vocab10k.freq2')
```

