import test from 'node:test';
import assert from 'node:assert';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { SemanticChunker, TokenizerType, ChunkType, walkProject } from '../index.mjs';

// Test that ESM imports work correctly
test('ESM imports should work', () => {
    assert(SemanticChunker, 'SemanticChunker should be imported');
    assert(TokenizerType, 'TokenizerType should be imported');
    assert(ChunkType, 'ChunkType should be imported');
    assert(walkProject, 'walkProject should be imported');
});

// Test enum values
test('TokenizerType enum should have correct values', () => {
    assert.equal(TokenizerType.Characters, 0);
    assert.equal(TokenizerType.Tiktoken, 1);
    assert.equal(TokenizerType.HuggingFace, 2);
});

test('ChunkType enum should have correct values', () => {
    assert.equal(ChunkType.Semantic, 0);
    assert.equal(ChunkType.Text, 1);
});

// Test basic async iteration with for await
test('should support for await...of with chunkCode', async () => {
    const chunker = new SemanticChunker();
    const content = `
def hello():
    return "Hello, World!"

def goodbye():
    return "Goodbye!"
`;

    const chunks = [];
    for await (const chunk of chunker.chunkCode(content, 'python')) {
        chunks.push(chunk);
    }

    assert(chunks.length > 0, 'Should have generated chunks');
    assert(chunks[0].metadata.language === 'python', 'Should have python metadata');
});

test('should support for await...of with chunkText', async () => {
    const chunker = new SemanticChunker(50);
    const content = 'This is a test. '.repeat(20);

    const chunks = [];
    for await (const chunk of chunker.chunkText(content)) {
        chunks.push(chunk);
    }

    assert(chunks.length > 0, 'Should have generated text chunks');
    assert(chunks[0].chunkType === ChunkType.Text, 'Should be text chunks');
});

test('should support for await...of with walkProject', async () => {
    // Create temporary test directory
    const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'breeze-esm-test-'));

    try {
        // Create test files
        await fs.writeFile(path.join(tempDir, 'test.js'), 'console.log("test");');
        await fs.writeFile(path.join(tempDir, 'test.py'), 'print("test")');

        const chunks = [];
        for await (const chunk of walkProject(tempDir)) {
            chunks.push(chunk);
        }

        assert(chunks.length > 0, 'Should have found project chunks');
        assert(chunks[0].filePath, 'Should have file path');
        assert(chunks[0].chunk, 'Should have chunk data');
    } finally {
        await fs.rm(tempDir, { recursive: true });
    }
});

// Test multiple concurrent iterations
test('should support multiple concurrent async iterations', async () => {
    const chunker = new SemanticChunker();
    const pythonCode = 'def test(): pass';
    const jsCode = 'function test() {}';

    // Start multiple iterations concurrently
    const [pythonChunks, jsChunks] = await Promise.all([
        (async () => {
            const chunks = [];
            for await (const chunk of chunker.chunkCode(pythonCode, 'python')) {
                chunks.push(chunk);
            }
            return chunks;
        })(),
        (async () => {
            const chunks = [];
            for await (const chunk of chunker.chunkCode(jsCode, 'javascript')) {
                chunks.push(chunk);
            }
            return chunks;
        })()
    ]);

    assert(pythonChunks.length > 0, 'Should have Python chunks');
    assert(jsChunks.length > 0, 'Should have JavaScript chunks');
    assert(pythonChunks[0].metadata.language === 'python');
    assert(jsChunks[0].metadata.language === 'javascript');
});

// Test early termination
test('should handle early termination with break', async () => {
    const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'breeze-break-test-'));

    try {
        // Create many test files
        for (let i = 0; i < 10; i++) {
            await fs.writeFile(path.join(tempDir, `test${i}.py`), `def test${i}(): pass`);
        }

        let count = 0;
        for await (const chunk of walkProject(tempDir)) {
            count++;
            if (count >= 3) {
                break; // Early termination
            }
        }

        assert.equal(count, 3, 'Should have stopped at 3 chunks');
    } finally {
        await fs.rm(tempDir, { recursive: true });
    }
});

// Test error handling in async iteration
test('should handle errors gracefully in async iteration', async () => {
    const chunker = new SemanticChunker();

    // With our improved error handling, errors are now properly thrown
    let errorThrown = false;
    const chunks = [];

    try {
        for await (const chunk of chunker.chunkCode('code', 'UNSUPPORTED_LANG')) {
            chunks.push(chunk);
        }
    } catch (error) {
        errorThrown = true;
        assert(error.message.includes('UnsupportedLanguage'));
    }

    assert(errorThrown, 'Should throw error for unsupported language');
    assert.equal(chunks.length, 0, 'Should get no chunks for unsupported language');
});

// Test that async iterators can be passed around
test('async iterators should be passable to functions', async () => {
    async function collectChunks(iterator) {
        const chunks = [];
        for await (const chunk of iterator) {
            chunks.push(chunk);
        }
        return chunks;
    }

    const chunker = new SemanticChunker();
    const iterator = chunker.chunkCode('def test(): pass', 'python');

    const chunks = await collectChunks(iterator);
    assert(chunks.length > 0, 'Should have collected chunks');
});

// Test Symbol.asyncIterator property
test('iterators should have Symbol.asyncIterator', () => {
    const chunker = new SemanticChunker();
    const codeIterator = chunker.chunkCode('test', 'python');
    const textIterator = chunker.chunkText('test');
    const projectIterator = walkProject('.');

    assert(Symbol.asyncIterator in codeIterator, 'Code iterator should have Symbol.asyncIterator');
    assert(Symbol.asyncIterator in textIterator, 'Text iterator should have Symbol.asyncIterator');
    assert(Symbol.asyncIterator in projectIterator, 'Project iterator should have Symbol.asyncIterator');
});

console.log('ESM tests completed!');
