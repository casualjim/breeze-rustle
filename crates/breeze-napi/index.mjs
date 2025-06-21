// ESM wrapper for the NAPI module with async generator support
import { createRequire } from 'module';

const require = createRequire(import.meta.url);
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

// Re-export only the enums/constants users need
export const { TokenizerType, ChunkType } = native;

// Re-export language utility functions
export const { supportedLanguages, isLanguageSupported } = native;

// Export our wrapped versions
export { SemanticChunker, walkProject };

// Also export as default for convenience
export default {
  TokenizerType,
  ChunkType,
  supportedLanguages,
  isLanguageSupported,
  SemanticChunker,
  walkProject,
};
