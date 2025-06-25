// Wrapper for the native module that adds async generator support
const native = require('./index.native.js');

// Helper to convert our iterator to an async generator
async function* makeAsyncGenerator(iterator) {
  while (true) {
    const result = await iterator.next();
    if (result.done) {
      return;
    }
    if (result.value) {
      yield result.value;
    }
  }
}

// Wrap the SemanticChunker class using composition
class SemanticChunker {
  constructor(maxChunkSize, tokenizer, hfModel) {
    this._native = new native.SemanticChunker(maxChunkSize, tokenizer, hfModel);
  }

  // Return async generators directly
  async *chunkCode(content, language, filePath) {
    const iterator = await this._native.chunkCode(content, language, filePath);
    yield* makeAsyncGenerator(iterator);
  }

  async *chunkText(content, filePath) {
    const iterator = await this._native.chunkText(content, filePath);
    yield* makeAsyncGenerator(iterator);
  }
}

// Wrap the walkProject function to return async generator
async function* walkProject(path, maxChunkSize, tokenizer, hfModel, maxParallel) {
  const iterator = await native.walkProject(path, maxChunkSize, tokenizer, hfModel, maxParallel);
  yield* makeAsyncGenerator(iterator);
}

// Only export our wrapped interface
module.exports = {
  // Re-export only the enums/constants users need
  TokenizerType: native.TokenizerType,
  ChunkType: native.ChunkType,

  // Re-export language utility functions
  supportedLanguages: native.supportedLanguages,
  isLanguageSupported: native.isLanguageSupported,

  // Export our wrapped versions
  SemanticChunker,
  walkProject,
};