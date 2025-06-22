# Future todo list

Keeping some bullet points so I don't forget

* On reindexing ensure that documents that no longer exist get deleted
* Run table optimization at the end of a full reindex
* Implement reranking models
* Add extra information about the semantic content of a chunk
* Track failed files/batches and broadcast
* Implement streaming writes to lancedb

## building

* needs to install zig
* needs to install nextest
* needs to install tree-sitter-cli (ideall, same version as our lib)
* needs to install protobuf-compiler

### Ubuntu

* build-essential
* protobuf-compiler
* tree-sitter-cli (from repo not packages)
* libssl-dev
* rust
* volta
* uv
