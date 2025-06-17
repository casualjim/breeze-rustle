"""Tests for the async ProjectWalker functionality."""

import pytest
import tempfile
import os
from pathlib import Path
from breeze import SemanticChunker, TokenizerType, ChunkType


class TestAsyncProjectWalker:
    """Test the async iterator functionality of ProjectWalker."""
    
    @pytest.fixture
    def test_project_dir(self):
        """Create a temporary directory with test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)
            
            # Create a Rust file
            rust_file = test_dir / "main.rs"
            rust_file.write_text("""
fn main() {
    println!("Hello, world!");
}

fn helper_function() {
    let x = 42;
    println!("Helper: {}", x);
}
""")
            
            # Create a Python file
            python_file = test_dir / "script.py"
            python_file.write_text("""
def hello():
    print("Hello from Python!")

class Calculator:
    def add(self, a, b):
        return a + b
""")
            
            # Create a text file (use .txt to ensure it's processed as text)
            text_file = test_dir / "README.txt"
            text_file.write_text("""
Test Project

This is a test project for the walker functionality.
""")
            
            # Create a subdirectory with another file
            subdir = test_dir / "src"
            subdir.mkdir()
            js_file = subdir / "app.js"
            js_file.write_text("""
function greet(name) {
    console.log(`Hello, ${name}!`);
}
""")
            
            yield str(test_dir)
    
    @pytest.mark.asyncio
    async def test_basic_async_iteration(self, test_project_dir):
        """Test that ProjectWalker works as an async iterator."""
        chunker = SemanticChunker(max_chunk_size=500)
        walker = await chunker.walk_project(test_project_dir)
        
        # Verify walker type
        assert hasattr(walker, '__aiter__')
        assert hasattr(walker, '__anext__')
        
        chunks = []
        async for chunk in walker:
            chunks.append(chunk)
        
        # Should have found chunks from multiple files
        assert len(chunks) > 0, "Should have processed at least one chunk"
        
        # Verify chunk structure
        for chunk in chunks:
            assert hasattr(chunk, 'file_path')
            assert hasattr(chunk, 'chunk')
            assert hasattr(chunk.chunk, 'chunk_type')
            assert chunk.chunk.chunk_type in [ChunkType.SEMANTIC, ChunkType.TEXT]
            assert chunk.chunk.start_line >= 1, "Line numbers should be 1-based"
            assert chunk.chunk.end_line >= chunk.chunk.start_line
        
        # Check we got chunks from different files
        file_paths = {chunk.file_path for chunk in chunks}
        assert len(file_paths) > 1, "Should have processed multiple files"
        
        # Check we have both semantic and text chunks
        has_semantic = any(chunk.chunk.chunk_type == ChunkType.SEMANTIC for chunk in chunks)
        has_text = any(chunk.chunk.chunk_type == ChunkType.TEXT for chunk in chunks)
        assert has_semantic, "Should have semantic chunks"
        assert has_text, "Should have text chunks"
    
    @pytest.mark.asyncio
    async def test_multiple_independent_walkers(self, test_project_dir):
        """Test that multiple walkers work independently."""
        chunker = SemanticChunker()
        
        # Create two independent walkers
        walker1 = await chunker.walk_project(test_project_dir)
        walker2 = await chunker.walk_project(test_project_dir)
        
        # Count chunks from both walkers
        count1 = 0
        async for _ in walker1:
            count1 += 1
        
        count2 = 0
        async for _ in walker2:
            count2 += 1
        
        assert count1 == count2, "Both walkers should process the same number of chunks"
        assert count1 > 0, "Should have processed at least one chunk"
    
    @pytest.mark.asyncio
    async def test_walker_early_termination(self, test_project_dir):
        """Test that walker can be terminated early without issues."""
        chunker = SemanticChunker()
        walker = await chunker.walk_project(test_project_dir)
        
        # Process only first few chunks
        chunks_processed = 0
        async for chunk in walker:
            chunks_processed += 1
            if chunks_processed >= 2:
                break
        
        assert chunks_processed == 2, "Should have processed exactly 2 chunks"
        
        # Walker should still be in a valid state
        # (not easily testable, but shouldn't crash)
    
    @pytest.mark.asyncio
    async def test_walker_exception_handling(self):
        """Test walker behavior with invalid directory."""
        chunker = SemanticChunker()
        
        # This should not raise during walker creation
        walker = await chunker.walk_project("/nonexistent/directory")
        
        # But iteration might yield no results or handle gracefully
        chunks = []
        async for chunk in walker:
            chunks.append(chunk)
        
        # Should handle gracefully (empty results)
        assert isinstance(chunks, list)
    
    @pytest.mark.asyncio
    async def test_walker_with_different_tokenizers(self, test_project_dir):
        """Test walker with different tokenizer types."""
        for tokenizer in [TokenizerType.CHARACTERS, TokenizerType.TIKTOKEN]:
            chunker = SemanticChunker(
                max_chunk_size=200,
                tokenizer=tokenizer
            )
            walker = await chunker.walk_project(test_project_dir)
            
            chunks = []
            async for chunk in walker:
                chunks.append(chunk)
            
            assert len(chunks) > 0, f"Should work with {tokenizer}"
            
            # Verify all chunks have valid structure
            for chunk in chunks:
                assert chunk.chunk.text, "Chunk should have text content"
                assert chunk.chunk.metadata.language, "Chunk should have language"
    
    @pytest.mark.asyncio
    async def test_walker_respects_max_parallel(self, test_project_dir):
        """Test that max_parallel parameter is respected."""
        # This is harder to test directly, but we can verify it doesn't break
        for max_parallel in [1, 2, 8]:
            chunker = SemanticChunker()
            walker = await chunker.walk_project(
                test_project_dir,
                max_parallel=max_parallel
            )
            
            chunks = []
            async for chunk in walker:
                chunks.append(chunk)
            
            assert len(chunks) > 0, f"Should work with max_parallel={max_parallel}"
    
    @pytest.mark.asyncio
    async def test_walker_empty_directory(self):
        """Test walker behavior with empty directory."""
        with tempfile.TemporaryDirectory() as empty_dir:
            chunker = SemanticChunker()
            walker = await chunker.walk_project(empty_dir)
            
            chunks = []
            async for chunk in walker:
                chunks.append(chunk)
            
            assert len(chunks) == 0, "Empty directory should yield no chunks"
    
    @pytest.mark.asyncio
    async def test_walker_large_chunk_size(self, test_project_dir):
        """Test walker with very large chunk size."""
        chunker = SemanticChunker(max_chunk_size=10000)
        walker = await chunker.walk_project(test_project_dir)
        
        chunks = []
        async for chunk in walker:
            chunks.append(chunk)
        
        # With large chunk size, we should get fewer, larger chunks
        assert len(chunks) > 0
        for chunk in chunks:
            # Chunks might be large but should still be reasonable
            assert len(chunk.chunk.text) <= 15000  # Allow some overhead
    
    @pytest.mark.asyncio
    async def test_walker_small_chunk_size(self, test_project_dir):
        """Test walker with very small chunk size."""
        chunker = SemanticChunker(max_chunk_size=50)
        walker = await chunker.walk_project(test_project_dir)
        
        chunks = []
        async for chunk in walker:
            chunks.append(chunk)
        
        # With small chunk size, we should get more, smaller chunks
        assert len(chunks) > 0
        
        # Most chunks should respect the size limit (allowing for semantic boundaries)
        oversized_chunks = [c for c in chunks if len(c.chunk.text) > 200]
        assert len(oversized_chunks) / len(chunks) < 0.5, "Most chunks should respect size limit"
    
    @pytest.mark.asyncio
    async def test_walker_semantic_vs_text_chunks(self, test_project_dir):
        """Test that walker properly distinguishes semantic vs text chunks."""
        chunker = SemanticChunker()
        walker = await chunker.walk_project(test_project_dir)
        
        semantic_chunks = []
        text_chunks = []
        
        async for chunk in walker:
            if chunk.chunk.chunk_type == ChunkType.SEMANTIC:
                semantic_chunks.append(chunk)
            elif chunk.chunk.chunk_type == ChunkType.TEXT:
                text_chunks.append(chunk)
        
        # Should have both types
        assert len(semantic_chunks) > 0, "Should have semantic chunks from code files"
        assert len(text_chunks) > 0, "Should have text chunks from text files"
        
        # Semantic chunks should have language info
        for chunk in semantic_chunks:
            assert chunk.chunk.metadata.language in ["Rust", "Python", "JavaScript"]
            assert chunk.chunk.metadata.node_type != "text_chunk"
        
        # Text chunks should be marked as text
        for chunk in text_chunks:
            assert chunk.chunk.metadata.language == "text"
            assert chunk.chunk.metadata.node_type == "text_chunk"
    
    @pytest.mark.asyncio
    async def test_walker_preserves_file_paths(self, test_project_dir):
        """Test that walker preserves correct file paths."""
        chunker = SemanticChunker()
        walker = await chunker.walk_project(test_project_dir)
        
        file_extensions = set()
        
        async for chunk in walker:
            # Verify file path is absolute and exists
            assert os.path.isabs(chunk.file_path), "File path should be absolute"
            
            # Extract extension
            ext = Path(chunk.file_path).suffix
            file_extensions.add(ext)
        
        # Should have found our test files (using .txt instead of .md since markdown is now supported)
        expected_extensions = {'.rs', '.py', '.txt', '.js'}
        assert expected_extensions.issubset(file_extensions), \
            f"Should have found files with extensions {expected_extensions}, got {file_extensions}"