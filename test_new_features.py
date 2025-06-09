#!/usr/bin/env python3
"""Test script for new tokenizer and text chunking features."""

import asyncio
from breeze_rustle import SemanticChunker

async def test_tokenizers():
    """Test different tokenizer options."""
    
    # Test characters tokenizer (default)
    print("Testing characters tokenizer...")
    chunker_chars = SemanticChunker(max_chunk_size=100, tokenizer="characters")
    
    code = """
def hello(name):
    return f"Hello {name}!"

class Greeter:
    def greet(self, name):
        return hello(name)
"""
    
    chunks = await chunker_chars.chunk_file(code, "Python", "test.py")
    print(f"Characters tokenizer: {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
        print(f"  Chunk {i+1}: {len(chunk.text)} chars, type: {chunk.metadata.node_type}")
    
    # Test tiktoken tokenizer
    print("\nTesting tiktoken tokenizer...")
    try:
        chunker_tiktoken = SemanticChunker(max_chunk_size=50, tokenizer="tiktoken")
        chunks = await chunker_tiktoken.chunk_file(code, "Python", "test.py")
        print(f"Tiktoken tokenizer: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
            print(f"  Chunk {i+1}: {len(chunk.text)} chars, type: {chunk.metadata.node_type}")
    except Exception as e:
        print(f"Tiktoken test failed: {e}")

async def test_text_chunking():
    """Test text chunking for unsupported languages."""
    
    print("\nTesting text chunking...")
    chunker = SemanticChunker(max_chunk_size=50, tokenizer="characters")
    
    # Test with supported language first
    print("Checking language support:")
    print(f"Python supported: {SemanticChunker.is_language_supported('Python')}")
    print(f"COBOL supported: {SemanticChunker.is_language_supported('COBOL')}")
    
    # Test text chunking
    text = """
This is a long text document that we want to chunk into smaller pieces.
It doesn't matter what programming language this is, because we're using
text-based chunking instead of semantic parsing.

This could be documentation, plain text, or code in an unsupported language.
The chunker will split it based on the tokenizer settings rather than
trying to understand the syntax.
"""
    
    chunks = await chunker.chunk_text(text, "document.txt")
    print(f"\nText chunking: {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {len(chunk.text)} chars, type: {chunk.metadata.node_type}")
        print(f"    Language: {chunk.metadata.language}")
        print(f"    Text preview: {repr(chunk.text[:50])}...")

async def test_workflow():
    """Test the recommended workflow for handling unsupported languages."""
    
    print("\n" + "="*60)
    print("Testing recommended workflow:")
    print("1. Check if language is supported")
    print("2. Use semantic chunking if supported, text chunking if not")
    
    chunker = SemanticChunker(max_chunk_size=100)
    
    # Test cases
    test_cases = [
        ("Python", "def hello(): pass", "hello.py"),
        ("COBOL", "PROGRAM-ID. HELLO-WORLD.", "hello.cob"),
        ("Brainfuck", "++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>", "hello.bf")
    ]
    
    for language, code, filename in test_cases:
        print(f"\nTesting {language}:")
        
        if SemanticChunker.is_language_supported(language):
            print(f"  ✓ {language} is supported - using semantic chunking")
            try:
                chunks = await chunker.chunk_file(code, language, filename)
                print(f"  → Semantic chunks: {len(chunks)}")
                if chunks:
                    print(f"    First chunk type: {chunks[0].metadata.node_type}")
            except Exception as e:
                print(f"  ✗ Semantic chunking failed: {e}")
        else:
            print(f"  ⚠ {language} not supported - using text chunking")
            chunks = await chunker.chunk_text(code, filename)
            print(f"  → Text chunks: {len(chunks)}")
            if chunks:
                print(f"    First chunk type: {chunks[0].metadata.node_type}")

async def main():
    """Run all tests."""
    await test_tokenizers()
    await test_text_chunking()
    await test_workflow()
    
    print("\n" + "="*60)
    print("All tests completed! 🎉")

if __name__ == "__main__":
    asyncio.run(main())