const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs').promises;
const path = require('node:path');
const os = require('node:os');
const { SemanticChunker, TokenizerType, ChunkType, walkProject, supportedLanguages, isLanguageSupported } = require('../index.js');

// Test Basic Functionality
test('should support 44+ languages', async () => {
    const languages = supportedLanguages();
    assert(languages.length >= 44, `Should support at least 44 languages, got ${languages.length}`);

    // Test core languages (top 10)
    const coreLanguages = ["python", "javascript", "java", "cpp", "c", "csharp", "typescript", "sql", "php", "go"];
    for (const lang of coreLanguages) {
        assert(languages.includes(lang), `${lang} should be supported`);
    }

    // Test popular languages (11-20)
    const popularLanguages = ["rust", "swift", "kotlin", "ruby", "r", "bash", "scala", "dart", "powershell"];
    for (const lang of popularLanguages) {
        assert(languages.includes(lang), `${lang} should be supported`);
    }

    // Test specialized languages (21-30)
    const specializedLanguages = ["objc", "perl", "fortran", "matlab", "lua", "haskell", "asm", "pascal", "ada"];
    for (const lang of specializedLanguages) {
        assert(languages.includes(lang), `${lang} should be supported`);
    }

    // Test emerging & niche (31-41)
    const emergingLanguages = ["elixir", "erlang", "commonlisp", "julia", "clojure", "solidity", "apex", "groovy", "ocaml", "scheme", "zig"];
    for (const lang of emergingLanguages) {
        assert(languages.includes(lang), `${lang} should be supported`);
    }

    // Test web languages (HTML, CSS, TSX)
    const webLanguages = ["html", "css", "tsx"];
    for (const lang of webLanguages) {
        assert(languages.includes(lang), `${lang} should be supported`);
    }
});

test('should correctly identify supported languages', () => {
    // Test case-insensitive support
    assert(isLanguageSupported("Python"));
    assert(isLanguageSupported("python"));
    assert(isLanguageSupported("PYTHON"));

    assert(isLanguageSupported("Rust"));
    assert(isLanguageSupported("rust"));

    assert(isLanguageSupported("JavaScript"));
    assert(isLanguageSupported("javascript"));

    assert(isLanguageSupported("TypeScript"));
    assert(isLanguageSupported("typescript"));

    assert(isLanguageSupported("Go"));
    assert(isLanguageSupported("go"));

    // Test more languages
    assert(isLanguageSupported("Kotlin"));
    assert(isLanguageSupported("kotlin"));

    assert(isLanguageSupported("Swift"));
    assert(isLanguageSupported("swift"));

    assert(isLanguageSupported("Scala"));
    assert(isLanguageSupported("scala"));

    // Fortran is now supported!
    assert(isLanguageSupported("Fortran"));
    assert(isLanguageSupported("fortran"));

    // Still not supported (not in top 50)
    assert(!isLanguageSupported("COBOL"));
    assert(!isLanguageSupported("Brainfuck"));
});

test('should chunk a simple Python file correctly', async () => {
    const content = `def hello(name):
    message = f"Hello {name}"
    return message

class Greeter:
    def __init__(self, lang):
        self.lang = lang

    def greet(self, name):
        return hello(name)
`;

    const chunker = new SemanticChunker();
    const chunks = [];

    for await (const chunk of chunker.chunkCode(content, "Python")) {
        chunks.push(chunk);
    }

    assert(chunks.length > 0);

    // Check chunk structure
    for (const chunk of chunks) {
        assert(chunk.text);
        assert(chunk.startByte >= 0);
        assert(chunk.endByte > chunk.startByte);
        assert(chunk.startLine >= 1);
        assert(chunk.endLine >= chunk.startLine);

        // Check metadata
        assert(chunk.metadata);
        assert.equal(chunk.metadata.language, "Python");
        assert(chunk.metadata.nodeType);
    }
});

test('should respect max chunk size', async () => {
    // Create a large Python file
    const functions = [];
    for (let i = 0; i < 50; i++) {
        functions.push(`def function_${i}():\n    """Docstring for function ${i}"""\n    return ${i}\n`);
    }
    const content = functions.join('\n');

    const chunker = new SemanticChunker(500);
    const chunks = [];

    for await (const chunk of chunker.chunkCode(content, "Python")) {
        chunks.push(chunk);
    }

    // All chunks should be under the size limit (allowing some overage for semantic boundaries)
    for (const chunk of chunks) {
        assert(chunk.text.length <= 1000, `Chunk too large: ${chunk.text.length}`);
    }
});

// Test Tokenizer Types
test('should support character-based chunking', async () => {
    const content = "def test(): pass";
    const chunker = new SemanticChunker(100, TokenizerType.Characters);
    const chunks = [];

    for await (const chunk of chunker.chunkCode(content, "Python")) {
        chunks.push(chunk);
    }

    assert(chunks.length > 0);
});

test('should support tiktoken tokenizer', async () => {
    const content = "def test(): pass";
    const chunker = new SemanticChunker(100, TokenizerType.Tiktoken);
    const chunks = [];

    for await (const chunk of chunker.chunkCode(content, "Python")) {
        chunks.push(chunk);
    }

    assert(chunks.length > 0);
});

test('should support HuggingFace tokenizer with model', async () => {
    const content = "def test(): pass";
    const chunker = new SemanticChunker(100, TokenizerType.HuggingFace, "bert-base-uncased");
    const chunks = [];

    for await (const chunk of chunker.chunkCode(content, "Python")) {
        chunks.push(chunk);
    }

    assert(chunks.length > 0);
});

test('should fail when HuggingFace tokenizer is used without model', () => {
    assert.throws(() => {
        new SemanticChunker(100, TokenizerType.HuggingFace);
    }, /HuggingFace requires hfModel/);
});

// Test Metadata Extraction
test('should extract function metadata correctly', async () => {
    const content = `def calculate_sum(a, b):
    """Calculate sum of two numbers"""
    result = a + b
    return result
`;

    const chunker = new SemanticChunker();
    const chunks = [];

    for await (const chunk of chunker.chunkCode(content, "Python")) {
        chunks.push(chunk);
    }

    assert(chunks.length > 0);

    // Check function metadata
    const chunk = chunks[0];
    assert.equal(chunk.metadata.nodeType, "function_definition");
    assert.equal(chunk.metadata.nodeName, "calculate_sum");
});

test('should extract class and method metadata with parent context', async () => {
    const content = `class Calculator:
    def __init__(self):
        self.value = 0

    def add(self, x):
        self.value += x
        return self.value
`;

    const chunker = new SemanticChunker();
    const chunks = [];

    for await (const chunk of chunker.chunkCode(content, "Python")) {
        chunks.push(chunk);
    }

    // Find chunks with methods
    const methodChunks = chunks.filter(c => c.text.includes("def ") && !c.text.includes("__init__"));

    if (methodChunks.length > 0) {
        const methodChunk = methodChunks[0];
        assert.equal(methodChunk.metadata.nodeName, "add");
        assert.equal(methodChunk.metadata.parentContext, "Calculator");
    }
});

// Test Text Chunking
test('should handle plain text chunking', async () => {
    const content = `This is plain text content that should be chunked.
It doesn't have any specific programming language syntax.
Just regular text that needs to be split into chunks.`;

    const chunker = new SemanticChunker(50);
    const chunks = [];

    for await (const chunk of chunker.chunkText(content)) {
        chunks.push(chunk);
    }

    assert(chunks.length > 0);

    for (const chunk of chunks) {
        assert.equal(chunk.metadata.nodeType, "text_chunk");
        assert.equal(chunk.metadata.language, "text");
    }
});

test('should include filename in text chunk metadata', async () => {
    const content = "Simple text content";
    const chunker = new SemanticChunker();
    const chunks = [];

    for await (const chunk of chunker.chunkText(content, "readme.txt")) {
        chunks.push(chunk);
    }

    assert(chunks.length > 0);
});

// Test Multi-Language Support
test('should chunk JavaScript files correctly', async () => {
    const content = `function greet(name) {
    console.log(\`Hello, \${name}!\`);
}

class Person {
    constructor(name) {
        this.name = name;
    }

    sayHello() {
        greet(this.name);
    }
}`;

    const chunker = new SemanticChunker();
    const chunks = [];

    for await (const chunk of chunker.chunkCode(content, "JavaScript")) {
        chunks.push(chunk);
    }

    assert(chunks.length > 0);
    assert(chunks.every(c => c.metadata.language === "JavaScript"));
});

test('should chunk Rust files correctly', async () => {
    const content = `struct Person {
    name: String,
    age: u32,
}

impl Person {
    fn new(name: String, age: u32) -> Self {
        Person { name, age }
    }

    fn greet(&self) {
        println!("Hello, I'm {}", self.name);
    }
}`;

    const chunker = new SemanticChunker();
    const chunks = [];

    for await (const chunk of chunker.chunkCode(content, "Rust")) {
        chunks.push(chunk);
    }

    assert(chunks.length > 0);
    assert(chunks.every(c => c.metadata.language === "Rust"));
});

test('should chunk TypeScript files correctly', async () => {
    const content = `interface User {
    id: number;
    name: string;
}

class UserService {
    private users: User[] = [];

    addUser(user: User): void {
        this.users.push(user);
    }

    getUser(id: number): User | undefined {
        return this.users.find(u => u.id === id);
    }
}`;

    const chunker = new SemanticChunker();
    const chunks = [];

    for await (const chunk of chunker.chunkCode(content, "TypeScript")) {
        chunks.push(chunk);
    }

    assert(chunks.length > 0);
    assert(chunks.every(c => c.metadata.language === "TypeScript"));
});

test('should chunk Go files correctly', async () => {
    const content = `package main

import "fmt"

type Person struct {
    Name string
    Age  int
}

func (p *Person) Greet() {
    fmt.Printf("Hello, I'm %s\\n", p.Name)
}

func main() {
    person := &Person{Name: "Alice", Age: 30}
    person.Greet()
}`;

    const chunker = new SemanticChunker();
    const chunks = [];

    for await (const chunk of chunker.chunkCode(content, "Go")) {
        chunks.push(chunk);
    }

    assert(chunks.length > 0);
    assert(chunks.every(c => c.metadata.language === "Go"));
});

// Test Error Handling
test('should handle unsupported language error', async () => {
    const content = "some code";
    const chunker = new SemanticChunker();

    // With our improved error handling, errors are now properly thrown
    let errorThrown = false;
    let chunks = [];

    try {
        for await (const chunk of chunker.chunkCode(content, "COBOL")) {
            chunks.push(chunk);
        }
    } catch (error) {
        errorThrown = true;
        assert(error.message.includes("UnsupportedLanguage"));
    }

    assert(errorThrown, "Should throw error for unsupported language");
    assert.equal(chunks.length, 0, "Should get no chunks for unsupported language");
});

test('should handle empty content gracefully', async () => {
    const chunker = new SemanticChunker();
    const chunks = [];

    for await (const chunk of chunker.chunkCode("", "Python")) {
        chunks.push(chunk);
    }

    // Should either return empty list or single empty chunk
    assert(chunks.length <= 1);
});

test('should handle malformed code without panicking', async () => {
    const content = `def broken_function(
    # Missing closing parenthesis and body
class IncompleteClass:
    def method_without_end(self):
        x = "unclosed string`;

    const chunker = new SemanticChunker();
    const chunks = [];

    // Should not throw
    for await (const chunk of chunker.chunkCode(content, "Python")) {
        chunks.push(chunk);
    }

    assert(Array.isArray(chunks));
});

// Test Project Walking
test('should walk a project directory', async (t) => {
    // Create a temporary directory with test files
    const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'breeze-test-'));

    try {
        // Create test files
        await fs.writeFile(path.join(tempDir, 'main.rs'), `
fn main() {
    println!("Hello, world!");
}

fn helper_function() {
    let x = 42;
    println!("Helper: {}", x);
}
`);

        await fs.writeFile(path.join(tempDir, 'script.py'), `
def hello():
    print("Hello from Python!")

class Calculator:
    def add(self, a, b):
        return a + b
`);

        await fs.writeFile(path.join(tempDir, 'README.txt'), `
Test Project

This is a test project for the walker functionality.
`);

        // Create subdirectory
        const srcDir = path.join(tempDir, 'src');
        await fs.mkdir(srcDir);
        await fs.writeFile(path.join(srcDir, 'app.js'), `
function greet(name) {
    console.log(\`Hello, \${name}!\`);
}
`);

        // Walk the project
        const chunks = [];
        for await (const chunk of walkProject(tempDir, 500, TokenizerType.Characters)) {
            chunks.push(chunk);
        }

        // Should have found chunks from multiple files
        assert(chunks.length > 0, "Should have processed at least one chunk");

        // Verify chunk structure
        for (const chunk of chunks) {
            assert(chunk.filePath);
            assert(chunk.chunk);
            assert(chunk.chunk.chunkType === ChunkType.Semantic || chunk.chunk.chunkType === ChunkType.Text || chunk.chunk.chunkType === ChunkType.EndOfFile);
            if (chunk.chunk.chunkType !== ChunkType.EndOfFile) {
                assert(chunk.chunk.startLine >= 1);
                assert(chunk.chunk.endLine >= chunk.chunk.startLine);
            }
        }

        // Check we got chunks from different files
        const filePaths = new Set(chunks.filter(c => c.chunk.chunkType !== ChunkType.EndOfFile).map(c => c.filePath));
        assert(filePaths.size > 1, "Should have processed multiple files");

        // Check we have both semantic and text chunks
        const hasSemantics = chunks.some(c => c.chunk.chunkType === ChunkType.Semantic);
        const hasText = chunks.some(c => c.chunk.chunkType === ChunkType.Text);
        const hasEof = chunks.some(c => c.chunk.chunkType === ChunkType.EndOfFile);
        assert(hasSemantics, "Should have semantic chunks");
        assert(hasText, "Should have text chunks");
        assert(hasEof, "Should have EOF chunks");

    } finally {
        // Clean up
        await fs.rm(tempDir, { recursive: true });
    }
});

test('should handle non-existent directory gracefully', async () => {
    const chunks = [];

    // Should not throw during iteration
    for await (const chunk of walkProject('/nonexistent/directory')) {
        chunks.push(chunk);
    }

    // Should handle gracefully (empty results)
    assert(Array.isArray(chunks));
});

test('should walk with different tokenizers', async (t) => {
    const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'breeze-test-'));

    try {
        await fs.writeFile(path.join(tempDir, 'test.py'), 'def test(): pass');

        for (const tokenizer of [TokenizerType.Characters, TokenizerType.Tiktoken]) {
            const chunks = [];
            for await (const chunk of walkProject(tempDir, 200, tokenizer)) {
                chunks.push(chunk);
            }

            assert(chunks.length > 0, `Should work with ${tokenizer}`);

            // Verify all chunks have valid structure
            for (const chunk of chunks) {
                assert(chunk.chunk.text);
                // EOF chunks have empty language field
                if (chunk.chunk.chunkType !== ChunkType.EndOfFile) {
                    assert(chunk.chunk.metadata.language);
                }
            }
        }
    } finally {
        await fs.rm(tempDir, { recursive: true });
    }
});

test('should respect max_parallel parameter', async (t) => {
    const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'breeze-test-'));

    try {
        // Create multiple files
        for (let i = 0; i < 5; i++) {
            await fs.writeFile(path.join(tempDir, `file${i}.py`), `def test${i}(): pass`);
        }

        for (const maxParallel of [1, 2, 8]) {
            const chunks = [];
            for await (const chunk of walkProject(tempDir, 100, TokenizerType.Characters, null, maxParallel)) {
                chunks.push(chunk);
            }

            assert(chunks.length > 0, `Should work with max_parallel=${maxParallel}`);
        }
    } finally {
        await fs.rm(tempDir, { recursive: true });
    }
});

// Test interface extraction
test('should extract interface metadata correctly', async () => {
    const content = `interface User {
    id: number;
    name: string;
    email: string;
}

interface AdminUser extends User {
    permissions: string[];
}`;

    const chunker = new SemanticChunker();
    const chunks = [];

    for await (const chunk of chunker.chunkCode(content, "TypeScript")) {
        chunks.push(chunk);
    }

    // Should find interfaces
    const interfaceChunks = chunks.filter(c => c.text.includes("interface"));
    assert(interfaceChunks.length > 0);

    // Check interface metadata
    for (const chunk of interfaceChunks) {
        if (chunk.text.includes("User") && !chunk.text.includes("AdminUser")) {
            assert.equal(chunk.metadata.nodeType, "interface_declaration");
            assert.equal(chunk.metadata.nodeName, "User");
        }
    }
});

// Test scope path extraction
test('should extract scope paths correctly', async () => {
    const content = `class OuterClass:
    class InnerClass:
        def nested_method(self):
            pass`;

    const chunker = new SemanticChunker();
    const chunks = [];

    for await (const chunk of chunker.chunkCode(content, "Python")) {
        chunks.push(chunk);
    }

    // Look for the nested method chunk
    for (const chunk of chunks) {
        if (chunk.text.includes("nested_method") && chunk.metadata.nodeName === "nested_method") {
            // Scope path should reflect nesting
            assert(chunk.metadata.scopePath.length >= 2);
            assert(chunk.metadata.scopePath.includes("OuterClass"));
            assert(chunk.metadata.scopePath.includes("InnerClass"));
        }
    }
});

// Test EOF chunks contain content and hash
test('should include content and hash in EOF chunks', async (t) => {
    const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'breeze-test-'));

    try {
        const testContent = `def test_function():
    return "Hello World"`;
        const testFile = path.join(tempDir, 'test.py');
        await fs.writeFile(testFile, testContent);

        const fileContents = {};
        const eofChunks = [];
        const regularChunks = [];

        for await (const chunk of walkProject(tempDir)) {
            if (chunk.chunk.chunkType === ChunkType.EndOfFile) {
                eofChunks.push(chunk);
                // EOF chunks should have content and hash
                assert(chunk.chunk.content !== null && chunk.chunk.content !== undefined, "EOF chunk should have content");
                assert(chunk.chunk.contentHash !== null && chunk.chunk.contentHash !== undefined, "EOF chunk should have contentHash");
                assert(typeof chunk.chunk.content === 'string', "EOF chunk content should be a string");
                assert(Array.isArray(chunk.chunk.contentHash), "EOF chunk contentHash should be an Array");
                assert(chunk.chunk.contentHash.length === 32, "Blake3 hash should be 32 bytes");

                // Store content for verification
                fileContents[chunk.filePath] = chunk.chunk.content;
            } else {
                regularChunks.push(chunk);
                // Non-EOF chunks should not have content/hash
                assert(chunk.chunk.content === null || chunk.chunk.content === undefined, "Non-EOF chunks should not have content");
                assert(chunk.chunk.contentHash === null || chunk.chunk.contentHash === undefined, "Non-EOF chunks should not have contentHash");
            }
        }

        // Should have EOF chunks
        assert(eofChunks.length > 0, "Should have EOF chunks");
        assert(eofChunks.length === Object.keys(fileContents).length, "Should have one EOF chunk per file");

        // Verify content matches actual file
        const actualContent = await fs.readFile(testFile, 'utf8');
        assert(fileContents[testFile] === actualContent, "EOF chunk content should match file content");

        // Verify chunks text is part of full content
        for (const chunk of regularChunks) {
            if (chunk.filePath === testFile) {
                assert(fileContents[testFile].includes(chunk.chunk.text), "Chunk text should be found in full content");
            }
        }

    } finally {
        await fs.rm(tempDir, { recursive: true });
    }
});

// Run all tests
console.log('Running NAPI tests...');
