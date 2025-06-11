import { SemanticChunker, TokenizerType, walkProject } from '../index.mjs';

// Example 1: Basic async iteration
async function basicIteration() {
    console.log('=== Example 1: Basic async iteration ===');
    const chunker = new SemanticChunker();
    const code = `
function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

class Calculator {
    add(a, b) {
        return a + b;
    }
}`;

    console.log('Chunking JavaScript code...');
    for await (const chunk of chunker.chunkCode(code, 'javascript')) {
        console.log(`- ${chunk.metadata.nodeType}: ${chunk.metadata.nodeName || '(anonymous)'}`);
    }
}

// Example 2: Collecting chunks into arrays
async function collectingChunks() {
    console.log('\n=== Example 2: Collecting chunks ===');
    const chunker = new SemanticChunker(100); // Small chunks
    const text = 'This is a long text. '.repeat(50);
    
    // Collect all chunks into an array
    const chunks = [];
    for await (const chunk of chunker.chunkText(text)) {
        chunks.push(chunk);
    }
    
    console.log(`Collected ${chunks.length} text chunks`);
    console.log(`First chunk: "${chunks[0].text.substring(0, 50)}..."`);
    console.log(`Last chunk: "${chunks[chunks.length - 1].text.substring(0, 50)}..."`);
}

// Example 3: Early termination with filtering
async function earlyTermination() {
    console.log('\n=== Example 3: Early termination ===');
    const chunker = new SemanticChunker();
    
    // Find first function in a file
    const code = `
// Some comments
const x = 42;

function targetFunction() {
    return "found me!";
}

function anotherFunction() {
    return "you won't see me";
}`;

    for await (const chunk of chunker.chunkCode(code, 'javascript')) {
        if (chunk.metadata.nodeType === 'function_declaration') {
            console.log(`Found first function: ${chunk.metadata.nodeName}`);
            break; // Stop after first function
        }
    }
}

// Example 4: Concurrent processing
async function concurrentProcessing() {
    console.log('\n=== Example 4: Concurrent processing ===');
    const chunker = new SemanticChunker();
    
    const pythonCode = 'def hello(): return "Python"';
    const rustCode = 'fn hello() -> &str { "Rust" }';
    const jsCode = 'function hello() { return "JavaScript"; }';
    
    // Process multiple languages concurrently
    const [pythonResult, rustResult, jsResult] = await Promise.all([
        processLanguage(chunker, pythonCode, 'python'),
        processLanguage(chunker, rustCode, 'rust'),
        processLanguage(chunker, jsCode, 'javascript')
    ]);
    
    console.log('Concurrent results:', { pythonResult, rustResult, jsResult });
}

async function processLanguage(chunker, code, language) {
    let count = 0;
    for await (const _ of chunker.chunkCode(code, language)) {
        count++;
    }
    return `${language}: ${count} chunks`;
}

// Example 5: Project walking with filtering
async function projectWalking() {
    console.log('\n=== Example 5: Project walking ===');
    // Walk current directory, but only process first 5 JavaScript files
    let jsFileCount = 0;
    const jsChunks = [];
    
    for await (const projectChunk of walkProject('.', 500, TokenizerType.Characters)) {
        if (projectChunk.filePath.endsWith('.js') || projectChunk.filePath.endsWith('.mjs')) {
            jsChunks.push(projectChunk);
            if (projectChunk.chunk.metadata.nodeType !== 'text_chunk') {
                console.log(`JS: ${projectChunk.filePath} - ${projectChunk.chunk.metadata.nodeType}`);
            }
            
            jsFileCount++;
            if (jsFileCount >= 5) {
                console.log('(stopped after 5 JS files)');
                break;
            }
        }
    }
}

// Example 6: Transform chunks on the fly
async function transformChunks() {
    console.log('\n=== Example 6: Transform chunks ===');
    const chunker = new SemanticChunker();
    
    async function* addLineNumbers(iterator) {
        let index = 0;
        for await (const chunk of iterator) {
            yield {
                ...chunk,
                index: index++,
                preview: `${chunk.text.split('\n')[0]}...`
            };
        }
    }
    
    const code = `
def func1():
    pass

def func2():
    pass`;
    
    const iterator = chunker.chunkCode(code, 'python');
    const numberedIterator = addLineNumbers(iterator);
    
    for await (const chunk of numberedIterator) {
        console.log(`Chunk ${chunk.index}: ${chunk.preview}`);
    }
}


// Example 7: Error handling in iteration
async function errorHandling() {
    console.log("\n=== Example 7: Error handling ===");
	const chunker = new SemanticChunker();

	// Note: Current NAPI implementation silently stops iteration on errors
	const invalidChunks = [];
	for await (const chunk of chunker.chunkCode("code", "INVALID_LANGUAGE")) {
		invalidChunks.push(chunk);
	}
	console.log(
		`Invalid language produced ${invalidChunks.length} chunks (should be 0)`,
	);

	// Continue with valid language
	console.log("Continuing with valid language...");
	for await (const chunk of chunker.chunkCode('print("hello")', "python")) {
		console.log(`Valid chunk: ${chunk.metadata.nodeType}`);
	}
}


// Run all examples
async function runAllExamples() {
    try {
        await basicIteration();
        await collectingChunks();
        await earlyTermination();
        await concurrentProcessing();
        await projectWalking();
        await transformChunks();
        await errorHandling();
    } catch (error) {
        console.error('Example error:', error);
    }
}

runAllExamples();