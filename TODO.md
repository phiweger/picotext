## TODO

Add a padding token to the tokenizer by simply adding 1 to the json

```python
# TODO:
# CharBPETokenizer.train?
# special_tokens: List[Union[str, AddedToken]] = ['<unk>'],
files = ['2020-05-29_uniref50.dayhoff.txt']

tokenizer = train_tokenizer(
    CharBPETokenizer(bert_normalizer=False),
    files,
    vocab_size=10000, min_frequency=2,
    special_tokens=['<unk>', '<pad>'])
tokenizer.save('.', '2020-05-29_uniref50.dayhoff.vocab10k.freq2')
```