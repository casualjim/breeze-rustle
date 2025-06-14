# Results of chunker runs

## Simple

This one uses:

* **Tokenizer**: characters
* **Text Splitter**: text

### Results (Simple)

```text
=== Chunking Summary ===
Directory: /Users/ivan/github/kuzudb/kuzu
Total files processed: 4467
Total chunks: 163461
Time elapsed: 111.33s
Chunks per second: 1468
Average chunks per file: 36.6
```

## Isolate tokenizer

This one uses:

* **Tokenizer**: hugging face model `ibm-granite/granite-embedding-125m-english`
* **Text Splitter**: text

### Results (Tokenizer only)

```text
=== Chunking Summary ===
Directory: /Users/ivan/github/kuzudb/kuzu
Total files processed: 4467
Total chunks: 85678
Time elapsed: 601.47s
Chunks per second: 142
Average chunks per file: 19.2
```

## Isolate tree-sitter parsers

This one uses:

* **Tokenizer**: characters
* **Text Splitter**: code + text

### Results (Parsers only)

Never finishes, interrupted after 40 minutes.

## Full config

This one uses:

* **Tokenizer**: hugging face model `ibm-granite/granite-embedding-125m-english`
* **Text Splitter**: code + text

### Results (Full config)

Never finishes, interrupted after 40 minutes. but definitely slower than any of the above.

## Full config with default ignores added

This one uses:

* **Tokenizer**: hugging face model `ibm-granite/granite-embedding-125m-english`
* **Text Splitter**: code + text

### Results (extra-ignores)

```text
=== Chunking Summary ===
Directory: /Users/ivan/github/kuzudb/kuzu
Total files processed: 4104
Total chunks: 16491
Time elapsed: 152.30s
Chunks per second: 108
Average chunks per file: 4.0
```
