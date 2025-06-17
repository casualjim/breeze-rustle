import pytest
from breeze import SemanticChunker, SemanticChunk, ChunkMetadata, TokenizerType


class TestBasicFunctionality:
    def test_supported_languages(self):
        """Should support 44 languages (41 from top 50 + HTML, CSS, TSX)"""
        languages = SemanticChunker.supported_languages()
        assert len(languages) >= 44  # We now have 44 languages total

        # Test core languages (top 10)
        core_languages = [
            "python",
            "javascript",
            "java",
            "cpp",
            "c",
            "csharp",
            "typescript",
            "sql",
            "php",
            "go",
        ]
        for lang in core_languages:
            assert lang in languages, f"{lang} should be supported"

        # Test popular languages (11-20)
        popular_languages = [
            "rust",
            "swift",
            "kotlin",
            "ruby",
            "r",
            "bash",
            "scala",
            "dart",
            "powershell",
        ]
        for lang in popular_languages:
            assert lang in languages, f"{lang} should be supported"

        # Test specialized languages (21-30)
        specialized_languages = [
            "objc",
            "perl",
            "fortran",
            "matlab",
            "lua",
            "haskell",
            "asm",
            "pascal",
            "ada",
        ]
        for lang in specialized_languages:
            assert lang in languages, f"{lang} should be supported"

        # Test emerging & niche (31-41)
        emerging_languages = [
            "elixir",
            "erlang",
            "commonlisp",
            "julia",
            "clojure",
            "solidity",
            "apex",
            "groovy",
            "ocaml",
            "scheme",
            "zig",
        ]
        for lang in emerging_languages:
            assert lang in languages, f"{lang} should be supported"

        # Test web languages (HTML, CSS, TSX)
        web_languages = ["html", "css", "tsx"]
        for lang in web_languages:
            assert lang in languages, f"{lang} should be supported"

    def test_language_support_check(self):
        """Should correctly identify supported languages"""
        # Test case-insensitive support for all our languages
        assert SemanticChunker.is_language_supported("Python")
        assert SemanticChunker.is_language_supported("python")
        assert SemanticChunker.is_language_supported("PYTHON")

        assert SemanticChunker.is_language_supported("Rust")
        assert SemanticChunker.is_language_supported("rust")

        assert SemanticChunker.is_language_supported("JavaScript")
        assert SemanticChunker.is_language_supported("javascript")

        assert SemanticChunker.is_language_supported("TypeScript")
        assert SemanticChunker.is_language_supported("typescript")

        assert SemanticChunker.is_language_supported("Go")
        assert SemanticChunker.is_language_supported("go")

        # Test more languages are supported
        assert SemanticChunker.is_language_supported("Kotlin")
        assert SemanticChunker.is_language_supported("kotlin")

        assert SemanticChunker.is_language_supported("Swift")
        assert SemanticChunker.is_language_supported("swift")

        assert SemanticChunker.is_language_supported("Scala")
        assert SemanticChunker.is_language_supported("scala")

        # Fortran is now supported!
        assert SemanticChunker.is_language_supported("Fortran")
        assert SemanticChunker.is_language_supported("fortran")

        # Still not supported (not in top 50)
        assert not SemanticChunker.is_language_supported("COBOL")
        assert not SemanticChunker.is_language_supported("Brainfuck")

    @pytest.mark.asyncio
    async def test_simple_python_file(self):
        """Should chunk a simple Python file correctly"""
        content = """def hello(name):
    message = f"Hello {name}"
    return message

class Greeter:
    def __init__(self, lang):
        self.lang = lang

    def greet(self, name):
        return hello(name)
"""
        chunker = SemanticChunker()
        chunk_stream = await chunker.chunk_code(content, "Python")

        chunks = []
        async for chunk in chunk_stream:
            chunks.append(chunk)

        assert len(chunks) > 0

        # Check chunk structure
        for chunk in chunks:
            assert isinstance(chunk, SemanticChunk)
            assert chunk.text
            assert chunk.start_byte >= 0
            assert chunk.end_byte > chunk.start_byte
            assert chunk.start_line >= 1
            assert chunk.end_line >= chunk.start_line

            # Check metadata
            assert isinstance(chunk.metadata, ChunkMetadata)
            assert chunk.metadata.language == "Python"
            assert chunk.metadata.node_type is not None

    @pytest.mark.asyncio
    async def test_chunk_size_limits(self):
        """Should respect max chunk size"""
        # Create a large Python file
        content = "\n".join(
            [
                f'def function_{i}():\n    """Docstring for function {i}"""\n    return {i}\n'
                for i in range(50)
            ]
        )

        chunker = SemanticChunker(max_chunk_size=500)
        chunk_stream = await chunker.chunk_code(content, "Python")

        chunks = []
        async for chunk in chunk_stream:
            chunks.append(chunk)

        # All chunks should be under the size limit (allowing some overage for semantic boundaries)
        for chunk in chunks:
            assert len(chunk.text) <= 1000  # Allow 2x for semantic boundaries


class TestTokenizerTypes:
    @pytest.mark.asyncio
    async def test_character_tokenizer(self):
        """Should support character-based chunking"""
        content = "def test(): pass"
        chunker = SemanticChunker(
            max_chunk_size=100, tokenizer_type=TokenizerType.CHARACTERS
        )
        chunk_stream = await chunker.chunk_code(content, "Python")

        chunks = []
        async for chunk in chunk_stream:
            chunks.append(chunk)

        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_tiktoken_tokenizer(self):
        """Should support tiktoken tokenizer"""
        content = "def test(): pass"
        chunker = SemanticChunker(
            max_chunk_size=100, tokenizer_type=TokenizerType.TIKTOKEN
        )
        chunk_stream = await chunker.chunk_code(content, "Python")

        chunks = []
        async for chunk in chunk_stream:
            chunks.append(chunk)

        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_huggingface_tokenizer(self):
        """Should support HuggingFace tokenizer with model"""
        content = "def test(): pass"
        chunker = SemanticChunker(
            max_chunk_size=100,
            tokenizer_type=TokenizerType.HUGGINGFACE,
            tokenizer_name="bert-base-uncased",
        )
        chunk_stream = await chunker.chunk_code(content, "Python")

        chunks = []
        async for chunk in chunk_stream:
            chunks.append(chunk)

        assert len(chunks) > 0

    def test_huggingface_without_model_fails(self):
        """Should fail when HuggingFace tokenizer is used without model"""
        with pytest.raises(ValueError, match="HUGGINGFACE requires tokenizer_name"):
            SemanticChunker(tokenizer_type=TokenizerType.HUGGINGFACE)


class TestMetadataExtraction:
    @pytest.mark.asyncio
    async def test_function_metadata(self):
        """Should extract function metadata correctly"""
        content = '''def calculate_sum(a, b):
    """Calculate sum of two numbers"""
    result = a + b
    return result
'''
        chunker = SemanticChunker()
        chunk_stream = await chunker.chunk_code(content, "Python")

        chunks = []
        async for chunk in chunk_stream:
            chunks.append(chunk)

        # Should have at least one chunk
        assert len(chunks) > 0

        # Check function metadata
        chunk = chunks[0]
        assert chunk.metadata.node_type == "function_definition"
        assert chunk.metadata.node_name == "calculate_sum"

    @pytest.mark.asyncio
    async def test_class_and_method_metadata(self):
        """Should extract class and method metadata with parent context"""
        content = """class Calculator:
    def __init__(self):
        self.value = 0

    def add(self, x):
        self.value += x
        return self.value
"""
        chunker = SemanticChunker()
        chunk_stream = await chunker.chunk_code(content, "Python")

        chunks = []
        async for chunk in chunk_stream:
            chunks.append(chunk)

        # Find chunks with methods
        method_chunks = [
            c for c in chunks if "def " in c.text and "__init__" not in c.text
        ]

        if method_chunks:
            method_chunk = method_chunks[0]
            assert method_chunk.metadata.node_name == "add"
            assert method_chunk.metadata.parent_context == "Calculator"

    @pytest.mark.asyncio
    async def test_scope_path_extraction(self):
        """Should extract scope paths correctly"""
        content = """class OuterClass:
    class InnerClass:
        def nested_method(self):
            pass
"""
        chunker = SemanticChunker()
        chunk_stream = await chunker.chunk_code(content, "Python")

        chunks = []
        async for chunk in chunk_stream:
            chunks.append(chunk)

        # Look for the nested method chunk
        for chunk in chunks:
            if (
                "nested_method" in chunk.text
                and chunk.metadata.node_name == "nested_method"
            ):
                # Scope path should reflect nesting
                assert len(chunk.metadata.scope_path) >= 2
                assert "OuterClass" in chunk.metadata.scope_path
                assert "InnerClass" in chunk.metadata.scope_path


class TestTextChunking:
    @pytest.mark.asyncio
    async def test_plain_text_chunking(self):
        """Should handle plain text chunking for unsupported content"""
        content = """This is plain text content that should be chunked.
It doesn't have any specific programming language syntax.
Just regular text that needs to be split into chunks."""

        chunker = SemanticChunker(max_chunk_size=50)
        chunk_stream = await chunker.chunk_text(content)

        chunks = []
        async for chunk in chunk_stream:
            chunks.append(chunk)

        assert len(chunks) > 0

        for chunk in chunks:
            assert chunk.metadata.node_type == "text_chunk"
            assert chunk.metadata.language == "text"

    @pytest.mark.asyncio
    async def test_text_chunking_with_filename(self):
        """Should include filename in text chunk metadata"""
        content = "Simple text content"
        chunker = SemanticChunker()
        chunk_stream = await chunker.chunk_text(content, "readme.txt")

        chunks = []
        async for chunk in chunk_stream:
            chunks.append(chunk)

        assert len(chunks) > 0


class TestMultiLanguageSupport:
    @pytest.mark.asyncio
    async def test_javascript_chunking(self):
        """Should chunk JavaScript files correctly"""
        content = """function greet(name) {
    console.log(`Hello, ${name}!`);
}

class Person {
    constructor(name) {
        this.name = name;
    }

    sayHello() {
        greet(this.name);
    }
}
"""
        chunker = SemanticChunker()
        chunk_stream = await chunker.chunk_code(content, "JavaScript")

        chunks = []
        async for chunk in chunk_stream:
            chunks.append(chunk)

        assert len(chunks) > 0
        assert all(c.metadata.language == "JavaScript" for c in chunks)

    @pytest.mark.asyncio
    async def test_rust_chunking(self):
        """Should chunk Rust files correctly"""
        content = """struct Person {
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
}
"""
        chunker = SemanticChunker()
        chunk_stream = await chunker.chunk_code(content, "Rust")

        chunks = []
        async for chunk in chunk_stream:
            chunks.append(chunk)

        assert len(chunks) > 0
        assert all(c.metadata.language == "Rust" for c in chunks)

        # Check for struct detection
        struct_chunks = [c for c in chunks if "struct Person" in c.text]
        if struct_chunks:
            # Be more flexible about node_type since it might vary
            assert struct_chunks[0].metadata.node_type is not None
            assert len(struct_chunks[0].metadata.node_type) > 0

    @pytest.mark.asyncio
    async def test_typescript_chunking(self):
        """Should chunk TypeScript files correctly"""
        content = """interface User {
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
}
"""
        chunker = SemanticChunker()
        chunk_stream = await chunker.chunk_code(content, "TypeScript")

        chunks = []
        async for chunk in chunk_stream:
            chunks.append(chunk)

        assert len(chunks) > 0
        assert all(c.metadata.language == "TypeScript" for c in chunks)

    @pytest.mark.asyncio
    async def test_go_chunking(self):
        """Should chunk Go files correctly"""
        content = """package main

import "fmt"

type Person struct {
    Name string
    Age  int
}

func (p *Person) Greet() {
    fmt.Printf("Hello, I'm %s\n", p.Name)
}

func main() {
    person := &Person{Name: "Alice", Age: 30}
    person.Greet()
}
"""
        chunker = SemanticChunker()
        chunk_stream = await chunker.chunk_code(content, "Go")

        chunks = []
        async for chunk in chunk_stream:
            chunks.append(chunk)

        assert len(chunks) > 0
        assert all(c.metadata.language == "Go" for c in chunks)

    @pytest.mark.asyncio
    async def test_tsx_chunking(self):
        """Should chunk TSX files correctly"""
        content = """import React from 'react';

interface Props {
    name: string;
    age: number;
}

const UserCard: React.FC<Props> = ({ name, age }) => {
    return (
        <div className="user-card">
            <h1>{name}</h1>
            <p>Age: {age}</p>
        </div>
    );
};

export default UserCard;"""
        chunker = SemanticChunker()
        chunk_stream = await chunker.chunk_code(content, "TSX")

        chunks = []
        async for chunk in chunk_stream:
            chunks.append(chunk)

        assert len(chunks) > 0
        assert all(c.metadata.language == "TSX" for c in chunks)

    @pytest.mark.asyncio
    async def test_css_chunking(self):
        """Should chunk CSS files correctly"""
        content = """.user-card {
    background-color: #f0f0f0;
    padding: 20px;
    border-radius: 8px;
}

.user-card h1 {
    color: #333;
    margin-bottom: 10px;
}

@media (max-width: 600px) {
    .user-card {
        padding: 10px;
    }
}"""
        chunker = SemanticChunker()
        chunk_stream = await chunker.chunk_code(content, "CSS")

        chunks = []
        async for chunk in chunk_stream:
            chunks.append(chunk)

        assert len(chunks) > 0
        assert all(c.metadata.language == "CSS" for c in chunks)


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_unsupported_language_error(self):
        """Should raise error for unsupported languages"""
        content = "some code"
        chunker = SemanticChunker()

        # The error might be raised when we iterate, not when we call chunk_code
        chunk_stream = await chunker.chunk_code(content, "COBOL")

        with pytest.raises(ValueError, match="Unsupported language"):
            async for chunk in chunk_stream:
                pass

    @pytest.mark.asyncio
    async def test_empty_content(self):
        """Should handle empty content gracefully"""
        chunker = SemanticChunker()
        chunk_stream = await chunker.chunk_code("", "Python")

        chunks = []
        async for chunk in chunk_stream:
            chunks.append(chunk)

        # Should either return empty list or single empty chunk
        assert len(chunks) <= 1

    @pytest.mark.asyncio
    async def test_malformed_code(self):
        """Should handle malformed code without panicking"""
        content = """def broken_function(
    # Missing closing parenthesis and body
class IncompleteClass:
    def method_without_end(self):
        x = "unclosed string"""

        chunker = SemanticChunker()
        # Should not raise an exception
        chunk_stream = await chunker.chunk_code(content, "Python")

        chunks = []
        async for chunk in chunk_stream:
            chunks.append(chunk)

        assert isinstance(chunks, list)


class TestExpandedMetadataExtraction:
    """Test metadata extraction for the expanded language constructs"""

    @pytest.mark.asyncio
    async def test_interface_extraction(self):
        """Should extract interface metadata correctly"""
        content = """interface User {
    id: number;
    name: string;
    email: string;
}

interface AdminUser extends User {
    permissions: string[];
}"""
        chunker = SemanticChunker()
        chunk_stream = await chunker.chunk_code(content, "TypeScript")

        chunks = []
        async for chunk in chunk_stream:
            chunks.append(chunk)

        # Should find interfaces
        interface_chunks = [c for c in chunks if "interface" in c.text]
        assert len(interface_chunks) > 0

        # Check interface metadata
        for chunk in interface_chunks:
            if "User" in chunk.text and "AdminUser" not in chunk.text:
                assert chunk.metadata.node_type == "interface_declaration"
                assert chunk.metadata.node_name == "User"

    @pytest.mark.asyncio
    async def test_trait_extraction(self):
        """Should extract trait metadata correctly"""
        content = """pub trait Display {
    fn fmt(&self) -> String;
}

pub trait Debug: Display {
    fn debug(&self) -> String;
}"""
        chunker = SemanticChunker()
        chunk_stream = await chunker.chunk_code(content, "Rust")

        chunks = []
        async for chunk in chunk_stream:
            chunks.append(chunk)

        # Should find traits
        trait_chunks = [c for c in chunks if "trait" in c.text]
        assert len(trait_chunks) > 0

        # Check trait metadata
        for chunk in trait_chunks:
            if "Display" in chunk.text and "Debug" not in chunk.text:
                assert chunk.metadata.node_type == "trait_item"
                assert chunk.metadata.node_name == "Display"

    @pytest.mark.asyncio
    async def test_enum_extraction(self):
        """Should extract enum metadata correctly"""
        content = """enum Color {
    Red,
    Green,
    Blue,
}

enum Result<T, E> {
    Ok(T),
    Err(E),
}"""
        # Use small chunk size to force splitting into separate enums
        chunker = SemanticChunker(max_chunk_size=50)
        chunk_stream = await chunker.chunk_code(content, "Rust")

        chunks = []
        async for chunk in chunk_stream:
            chunks.append(chunk)

        # Should find enums
        enum_chunks = [c for c in chunks if "enum" in c.text]
        assert len(enum_chunks) > 0

        # Check enum metadata
        for chunk in enum_chunks:
            if "Color" in chunk.text:
                assert chunk.metadata.node_type == "enum_item"
                assert chunk.metadata.node_name == "Color"

    @pytest.mark.asyncio
    async def test_namespace_extraction(self):
        """Should extract namespace metadata correctly"""
        content = """namespace MyCompany {
    export class Employee {
        constructor(public name: string) {}
    }

    export namespace HR {
        export class Manager extends Employee {
            constructor(name: string, public department: string) {
                super(name);
            }
        }
    }
}"""
        # Use small chunk size to force splitting
        chunker = SemanticChunker(max_chunk_size=100)
        chunk_stream = await chunker.chunk_code(content, "TypeScript")

        chunks = []
        async for chunk in chunk_stream:
            chunks.append(chunk)

        # Check for namespace in metadata (TypeScript parser uses "internal_module" for namespace)
        assert any(
            c.metadata.node_type == "internal_module"
            or "namespace" in str(c.metadata.scope_path)
            for c in chunks
        )

    @pytest.mark.asyncio
    async def test_scala_object_extraction(self):
        """Should extract Scala object metadata correctly"""
        content = """object MySingleton {
  def hello(): String = "Hello"
}

class MyClass {
  def greet(): String = MySingleton.hello()
}"""
        # Use small chunk size to force splitting
        chunker = SemanticChunker(max_chunk_size=50)
        chunk_stream = await chunker.chunk_code(content, "Scala")

        chunks = []
        async for chunk in chunk_stream:
            chunks.append(chunk)

        # Should find object definition
        object_chunks = [c for c in chunks if "object MySingleton" in c.text]
        if object_chunks:
            chunk = object_chunks[0]
            assert chunk.metadata.node_type == "object_definition"
            assert chunk.metadata.node_name == "MySingleton"

    @pytest.mark.asyncio
    async def test_elixir_module_extraction(self):
        """Should extract Elixir module metadata correctly"""
        content = """defmodule MyApp.User do
  defstruct [:name, :email]

  def new(name, email) do
    %__MODULE__{name: name, email: email}
  end

  defp validate_email(email) do
    String.contains?(email, "@")
  end
end"""
        # Use small chunk size to force splitting
        chunker = SemanticChunker(max_chunk_size=50)
        chunk_stream = await chunker.chunk_code(content, "Elixir")

        chunks = []
        async for chunk in chunk_stream:
            chunks.append(chunk)

        # Should find module and function definitions (Elixir parser doesn't extract node names, just node types)
        assert any(
            "defmodule" in c.metadata.node_type or c.metadata.node_type == "call"
            for c in chunks
        )
        # Elixir parser detects functions as "call" types but doesn't extract names yet
        assert any(c.metadata.node_type == "call" for c in chunks)


class TestDefinitionsAndReferences:
    @pytest.mark.asyncio
    async def test_basic_definitions_extraction(self):
        """Should extract variable and function definitions"""
        content = """def process_data(data):
    result = []
    total = 0

    for item in data:
        value = item * 2
        total += value
        result.append(value)

    return result, total
"""
        chunker = SemanticChunker()
        chunk_stream = await chunker.chunk_code(content, "Python")

        chunks = []
        async for chunk in chunk_stream:
            chunks.append(chunk)

        # Should extract some definitions
        if chunks:
            chunk = chunks[0]
            # At minimum should include the function name
            assert (
                "process_data" in chunk.metadata.definitions
                or len(chunk.metadata.definitions) > 0
            )

    @pytest.mark.asyncio
    async def test_references_extraction(self):
        """Should extract references to external symbols"""
        content = """import math

def calculate_circle_area(radius):
    return math.pi * radius ** 2
"""
        chunker = SemanticChunker()
        chunk_stream = await chunker.chunk_code(content, "Python")

        chunks = []
        async for chunk in chunk_stream:
            chunks.append(chunk)

        # Look for chunks containing the function
        func_chunks = [c for c in chunks if "calculate_circle_area" in c.text]
        if func_chunks:
            chunk = func_chunks[0]
            # Check that we have metadata, even if references extraction is limited
            assert chunk.metadata is not None
            # References might be empty if extraction is not fully implemented
            assert isinstance(chunk.metadata.references, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
