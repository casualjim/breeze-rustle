// This file provides a thin wrapper to convert callback-based streaming to async generators
import native from './index.node';

/**
 * Creates an async generator from callback-based streaming function
 */
function createAsyncGenerator(fn) {
  return async function* (...args) {
    const chunks = [];
    let resolve;
    let reject;
    let done = false;
    let promise = new Promise((res, rej) => {
      resolve = res;
      reject = rej;
    });

    // Set up callbacks
    const onChunk = (chunk) => {
      chunks.push(chunk);
      if (resolve) {
        const currentResolve = resolve;
        resolve = null;
        currentResolve();
      }
    };

    const onError = (error) => {
      done = true;
      if (reject) {
        reject(new Error(error));
      }
    };

    const onComplete = () => {
      done = true;
      if (resolve) {
        resolve();
      }
    };

    // Start the streaming
    fn(...args, onChunk, onError, onComplete);

    // Yield chunks as they come
    while (!done || chunks.length > 0) {
      if (chunks.length > 0) {
        yield chunks.shift();
      } else if (!done) {
        // Wait for more chunks
        await promise;
        promise = new Promise((res, rej) => {
          resolve = res;
          reject = rej;
        });
      }
    }
  };
}

class SemanticChunker extends native.SemanticChunker {
  async* chunkCode(content, language, filePath) {
    yield* createAsyncGenerator(super.chunkCode.bind(this))(content, language, filePath);
  }

  async* chunkText(content, filePath) {
    yield* createAsyncGenerator(super.chunkText.bind(this))(content, filePath);
  }
}

async function* walkProject(path, maxChunkSize, tokenizer, hfModel, maxParallel) {
  yield* createAsyncGenerator(native.walkProject)(path, maxChunkSize, tokenizer, hfModel, maxParallel);
}

// Re-export everything from native, with our wrapped versions
export * from './index.node';
export { SemanticChunker, walkProject };