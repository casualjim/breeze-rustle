# @breeze/chunkers

High-performance semantic code chunking for Node.js, powered by Rust and tree-sitter.

## Features

- **44+ Language Support**: Python, JavaScript, TypeScript, Rust, Go, Java, C++, and many more
- **Semantic Chunking**: Intelligently splits code at meaningful boundaries (functions, classes, etc.)
- **Async Iteration**: Modern async iterator API for streaming results
- **High Performance**: Rust-powered processing with parallel file handling
- **Multiple Tokenizers**: Character-based, Tiktoken, or HuggingFace tokenizers
- **Rich Metadata**: Extracts function names, class contexts, scope paths, and more
- **Cross-Platform**: Works on Windows, macOS, Linux, and more

## Installation

```bash
npm install @breeze/chunkers
```

## Usage

### Basic Code Chunking

```javascript
import { SemanticChunker } from '@breeze/chunkers';

const chunker = new SemanticChunker();

const code = `
def hello(name):
    return f"Hello, {name}!"

class Greeter:
    def greet(self, name):
        return hello(name)
`;

// Use async iteration to process chunks
for await (const chunk of chunker.chunkCode(code, 'python')) {
    console.log(`${chunk.metadata.nodeType}: ${chunk.metadata.nodeName}`);
    console.log(`Lines ${chunk.startLine}-${chunk.endLine}`);
    console.log(chunk.text);
}
```

### Walking a Project

```javascript
import { walkProject, TokenizerType } from '@breeze/chunkers';

// Walk all code files in a directory
for await (const projectChunk of walkProject('./src', 1500, TokenizerType.Characters)) {
    console.log(`File: ${projectChunk.filePath}`);
    console.log(`Type: ${projectChunk.chunk.metadata.nodeType}`);
    console.log(`Language: ${projectChunk.chunk.metadata.language}`);
}
```

### Text Chunking

For non-code content or unsupported languages:

```javascript
const chunker = new SemanticChunker(500); // 500 character chunks

const text = "Your long text content here...";

for await (const chunk of chunker.chunkText(text)) {
    console.log(`Text chunk: ${chunk.text.substring(0, 50)}...`);
}
```

## API Reference

### `SemanticChunker`

Main class for chunking code and text.

#### Constructor

```javascript
new SemanticChunker(maxChunkSize?, tokenizer?, hfModel?)
```

- `maxChunkSize` (optional): Maximum size for each chunk (default: 1500)
- `tokenizer` (optional): Tokenizer type to use
  - `TokenizerType.Characters` (default)
  - `TokenizerType.Tiktoken`
  - `TokenizerType.HuggingFace` (requires `hfModel`)
- `hfModel` (optional): HuggingFace model name when using HF tokenizer

#### Methods

##### `chunkCode(content, language, filePath?)`

Chunks source code with semantic understanding.

- Returns: Async iterator of `SemanticChunk` objects
- `content`: Source code string
- `language`: Programming language (case-insensitive)
- `filePath` (optional): Path for reference

##### `chunkText(content, filePath?)`

Chunks plain text without semantic parsing.

- Returns: Async iterator of `SemanticChunk` objects
- `content`: Text content
- `filePath` (optional): Path for reference

##### Static Methods

- `SemanticChunker.supportedLanguages()`: Get array of supported language names
- `SemanticChunker.isLanguageSupported(language)`: Check if a language is supported

### `walkProject(path, maxChunkSize?, tokenizer?, hfModel?, maxParallel?)`

Walk a project directory and chunk all files.

- Returns: Async iterator of `ProjectChunk` objects
- `path`: Directory path to walk
- `maxChunkSize` (optional): Maximum chunk size
- `tokenizer` (optional): Tokenizer type
- `hfModel` (optional): HuggingFace model name
- `maxParallel` (optional): Max parallel file processing (default: 8)

### Types

#### `SemanticChunk`

```typescript
interface SemanticChunk {
    chunkType: ChunkType;
    text: string;
    startByte: number;
    endByte: number;
    startLine: number;
    endLine: number;
    metadata: ChunkMetadata;
}
```

#### `ChunkMetadata`

```typescript
interface ChunkMetadata {
    nodeType: string;        // e.g., "function_definition", "class_definition"
    nodeName?: string;       // e.g., "myFunction", "MyClass"
    language: string;        // e.g., "Python", "JavaScript"
    parentContext?: string;  // e.g., "ParentClass" for methods
    scopePath: string[];     // Nested scope names
    definitions: string[];   // Defined symbols
    references: string[];    // Referenced external symbols
}
```

#### `ProjectChunk`

```typescript
interface ProjectChunk {
    filePath: string;
    chunk: SemanticChunk;
}
```

## Examples

### CommonJS Usage

```javascript
const { SemanticChunker, TokenizerType } = require('@breeze/chunkers');

async function processCode() {
    const chunker = new SemanticChunker();

    for await (const chunk of chunker.chunkCode(code, 'javascript')) {
        console.log(chunk);
    }
}
```

### ES Modules Usage

```javascript
import { SemanticChunker, TokenizerType } from '@breeze/chunkers';

const chunker = new SemanticChunker();
// ... use as shown above
```

### Concurrent Processing

```javascript
// Process multiple files concurrently
const files = ['file1.py', 'file2.js', 'file3.rs'];

const results = await Promise.all(
    files.map(async (file) => {
        const content = await fs.readFile(file, 'utf-8');
        const language = path.extname(file).slice(1);
        const chunks = [];

        for await (const chunk of chunker.chunkCode(content, language)) {
            chunks.push(chunk);
        }

        return { file, chunks };
    })
);
```

### Early Termination

```javascript
// Find the first class definition
for await (const chunk of chunker.chunkCode(code, 'python')) {
    if (chunk.metadata.nodeType === 'class_definition') {
        console.log(`Found class: ${chunk.metadata.nodeName}`);
        break; // Stop iteration
    }
}
```

## Supported Languages

Run `SemanticChunker.supportedLanguages()` to get the full list. Currently includes:

- **Top 10**: Python, JavaScript, Java, C++, C, C#, TypeScript, SQL, PHP, Go
- **Popular**: Rust, Swift, Kotlin, Ruby, R, Bash, Scala, Dart, PowerShell
- **Specialized**: Objective-C, Perl, Fortran, MATLAB, Lua, Haskell, Assembly, Pascal, Ada
- **Emerging**: Elixir, Erlang, Common Lisp, Julia, Clojure, Solidity, Apex, Groovy, OCaml, Scheme, Zig
- **Web**: HTML, CSS, TSX

## Performance

- Processes 1MB+ files in under 100ms
- Supports parallel processing of multiple files
- Zero-copy operations where possible
- Streaming results to minimize memory usage

## License

MIT
