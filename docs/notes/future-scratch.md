# Future todo list

Keeping some bullet points so I don't forget

* support reindexing, probably through labelling the type with the state (Skip, Reindex, Delete)
* On reindexing ensure that documents that no longer exist get deleted
* Add extra information about the semantic content of a chunk
* Track failed files/batches and broadcast
* Implement streaming writes to lancedb
* make batch size configurable or derived from the model
* avoid re-indexing files with the same path and content hash
* add exclusions for pure data files
* add final override file support with .breezeignore

## building

* needs to install zig
* needs to install nextest
* needs to install protobuf-compiler

### Ubuntu

* build-essential
* protobuf-compiler
* libssl-dev
* rust
* volta
* uv
