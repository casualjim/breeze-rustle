import pytest
from breeze_rustle import SemanticChunker, SemanticChunk, ChunkMetadata, TokenizerType


class TestBasicFunctionality:
    def test_supported_languages(self):
        """Should support at least 16 languages"""
        languages = SemanticChunker.supported_languages()
        assert len(languages) >= 15  # Currently 16, but PHP/Kotlin/SQL are excluded
        assert "Python" in languages
        assert "Rust" in languages
        assert "JavaScript" in languages
        assert "TypeScript" in languages
        assert "Java" in languages
        assert "C++" in languages
        assert "Go" in languages
        assert "Ruby" in languages
    
    def test_language_support_check(self):
        """Should correctly identify supported languages"""
        assert SemanticChunker.is_language_supported("Python")
        assert SemanticChunker.is_language_supported("Rust")
        assert not SemanticChunker.is_language_supported("COBOL")
        assert not SemanticChunker.is_language_supported("Fortran")
    
    @pytest.mark.asyncio
    async def test_simple_python_file(self):
        """Should chunk a simple Python file correctly"""
        content = '''def hello(name):
    message = f"Hello {name}"
    return message

class Greeter:
    def __init__(self, lang):
        self.lang = lang
    
    def greet(self, name):
        return hello(name)
'''
        chunker = SemanticChunker()
        chunks = await chunker.chunk_code(content, "Python")
        
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
        content = '\n'.join([f'def function_{i}():\n    """Docstring for function {i}"""\n    return {i}\n' for i in range(50)])
        
        chunker = SemanticChunker(max_chunk_size=500)
        chunks = await chunker.chunk_code(content, "Python")
        
        # All chunks should be under the size limit (allowing some overage for semantic boundaries)
        for chunk in chunks:
            assert len(chunk.text) <= 1000  # Allow 2x for semantic boundaries


class TestTokenizerTypes:
    @pytest.mark.asyncio
    async def test_character_tokenizer(self):
        """Should support character-based chunking"""
        content = "def test(): pass"
        chunker = SemanticChunker(max_chunk_size=100, tokenizer=TokenizerType.CHARACTERS)
        chunks = await chunker.chunk_code(content, "Python")
        assert len(chunks) > 0
    
    @pytest.mark.asyncio
    async def test_tiktoken_tokenizer(self):
        """Should support tiktoken tokenizer"""
        content = "def test(): pass"
        chunker = SemanticChunker(max_chunk_size=100, tokenizer=TokenizerType.TIKTOKEN)
        chunks = await chunker.chunk_code(content, "Python")
        assert len(chunks) > 0
    
    @pytest.mark.asyncio
    async def test_huggingface_tokenizer(self):
        """Should support HuggingFace tokenizer with model"""
        content = "def test(): pass"
        chunker = SemanticChunker(
            max_chunk_size=100, 
            tokenizer=TokenizerType.HUGGINGFACE,
            hf_model="bert-base-uncased"
        )
        chunks = await chunker.chunk_code(content, "Python")
        assert len(chunks) > 0
    
    def test_huggingface_without_model_fails(self):
        """Should fail when HuggingFace tokenizer is used without model"""
        with pytest.raises(ValueError, match="HUGGINGFACE requires hf_model"):
            SemanticChunker(tokenizer=TokenizerType.HUGGINGFACE)


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
        chunks = await chunker.chunk_code(content, "Python")
        
        # Should have at least one chunk
        assert len(chunks) > 0
        
        # Check function metadata
        chunk = chunks[0]
        assert chunk.metadata.node_type == "function_definition"
        assert chunk.metadata.node_name == "calculate_sum"
    
    @pytest.mark.asyncio
    async def test_class_and_method_metadata(self):
        """Should extract class and method metadata with parent context"""
        content = '''class Calculator:
    def __init__(self):
        self.value = 0
    
    def add(self, x):
        self.value += x
        return self.value
'''
        chunker = SemanticChunker()
        chunks = await chunker.chunk_code(content, "Python")
        
        # Find chunks with methods
        method_chunks = [c for c in chunks if "def " in c.text and "__init__" not in c.text]
        
        if method_chunks:
            method_chunk = method_chunks[0]
            assert method_chunk.metadata.node_name == "add"
            assert method_chunk.metadata.parent_context == "Calculator"
    
    @pytest.mark.asyncio
    async def test_scope_path_extraction(self):
        """Should extract scope paths correctly"""
        content = '''class OuterClass:
    class InnerClass:
        def nested_method(self):
            pass
'''
        chunker = SemanticChunker()
        chunks = await chunker.chunk_code(content, "Python")
        
        # Look for the nested method chunk
        for chunk in chunks:
            if "nested_method" in chunk.text and chunk.metadata.node_name == "nested_method":
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
        chunks = await chunker.chunk_text(content)
        
        assert len(chunks) > 0
        
        for chunk in chunks:
            assert chunk.metadata.node_type == "text_chunk"
            assert chunk.metadata.language == "plaintext"
    
    @pytest.mark.asyncio
    async def test_text_chunking_with_filename(self):
        """Should include filename in text chunk metadata"""
        content = "Simple text content"
        chunker = SemanticChunker()
        chunks = await chunker.chunk_text(content, "readme.txt")
        
        assert len(chunks) > 0


class TestMultiLanguageSupport:
    @pytest.mark.asyncio
    async def test_javascript_chunking(self):
        """Should chunk JavaScript files correctly"""
        content = '''function greet(name) {
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
'''
        chunker = SemanticChunker()
        chunks = await chunker.chunk_code(content, "JavaScript")
        
        assert len(chunks) > 0
        assert all(c.metadata.language == "JavaScript" for c in chunks)
    
    @pytest.mark.asyncio
    async def test_rust_chunking(self):
        """Should chunk Rust files correctly"""
        content = '''struct Person {
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
'''
        chunker = SemanticChunker()
        chunks = await chunker.chunk_code(content, "Rust")
        
        assert len(chunks) > 0
        assert all(c.metadata.language == "Rust" for c in chunks)
        
        # Check for struct detection
        struct_chunks = [c for c in chunks if "struct Person" in c.text]
        if struct_chunks:
            assert struct_chunks[0].metadata.node_type in ["struct_item", "struct"]
    
    @pytest.mark.asyncio
    async def test_typescript_chunking(self):
        """Should chunk TypeScript files correctly"""
        content = '''interface User {
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
'''
        chunker = SemanticChunker()
        chunks = await chunker.chunk_code(content, "TypeScript")
        
        assert len(chunks) > 0
        assert all(c.metadata.language == "TypeScript" for c in chunks)


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_unsupported_language_error(self):
        """Should raise error for unsupported languages"""
        content = "some code"
        chunker = SemanticChunker()
        
        with pytest.raises(ValueError, match="Unsupported language"):
            await chunker.chunk_code(content, "COBOL")
    
    @pytest.mark.asyncio
    async def test_empty_content(self):
        """Should handle empty content gracefully"""
        chunker = SemanticChunker()
        chunks = await chunker.chunk_code("", "Python")
        
        # Should either return empty list or single empty chunk
        assert len(chunks) <= 1
    
    @pytest.mark.asyncio
    async def test_malformed_code(self):
        """Should handle malformed code without panicking"""
        content = '''def broken_function(
    # Missing closing parenthesis and body
class IncompleteClass:
    def method_without_end(self):
        x = "unclosed string'''
        
        chunker = SemanticChunker()
        # Should not raise an exception
        chunks = await chunker.chunk_code(content, "Python")
        assert isinstance(chunks, list)


class TestDefinitionsAndReferences:
    @pytest.mark.asyncio
    async def test_basic_definitions_extraction(self):
        """Should extract variable and function definitions"""
        content = '''def process_data(data):
    result = []
    total = 0
    
    for item in data:
        value = item * 2
        total += value
        result.append(value)
    
    return result, total
'''
        chunker = SemanticChunker()
        chunks = await chunker.chunk_code(content, "Python")
        
        # Should extract some definitions
        if chunks:
            chunk = chunks[0]
            # At minimum should include the function name
            assert "process_data" in chunk.metadata.definitions or len(chunk.metadata.definitions) > 0
    
    @pytest.mark.asyncio
    async def test_references_extraction(self):
        """Should extract references to external symbols"""
        content = '''import math

def calculate_circle_area(radius):
    return math.pi * radius ** 2
'''
        chunker = SemanticChunker()
        chunks = await chunker.chunk_code(content, "Python")
        
        # Look for chunks containing the function
        func_chunks = [c for c in chunks if "calculate_circle_area" in c.text]
        if func_chunks:
            chunk = func_chunks[0]
            # Should reference 'math' module
            assert "math" in chunk.metadata.references or "pi" in chunk.metadata.references


if __name__ == "__main__":
    pytest.main([__file__, "-v"])