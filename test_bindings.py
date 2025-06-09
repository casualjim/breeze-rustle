#!/usr/bin/env python3
"""Quick test of Python bindings"""
import asyncio
import breeze_rustle

async def main():
    # Test module attributes
    print(f"Version: {breeze_rustle.__version__}")
    
    # Test SemanticChunker class
    chunker = breeze_rustle.SemanticChunker()
    
    # Test static methods
    languages = breeze_rustle.SemanticChunker.supported_languages()
    print(f"\nSupported languages ({len(languages)}): {', '.join(languages)}")
    
    # Test language support check
    assert breeze_rustle.SemanticChunker.is_language_supported("Python")
    assert not breeze_rustle.SemanticChunker.is_language_supported("COBOL")
    
    # Test chunking some Python code
    python_code = '''
def hello(name):
    """Say hello to someone"""
    message = f"Hello {name}!"
    print(message)
    return message

class Greeter:
    def __init__(self, language="en"):
        self.language = language
    
    def greet(self, name):
        if self.language == "en":
            return hello(name)
        else:
            return f"Bonjour {name}!"
'''
    
    chunks = await chunker.chunk_file(python_code, "Python", "test.py")
    
    print(f"\nGot {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"  Lines {chunk.start_line}-{chunk.end_line}")
        print(f"  Type: {chunk.metadata.node_type}")
        print(f"  Language: {chunk.metadata.language}")
        print(f"  Text preview: {chunk.text[:50]}...")
    
    # Test error handling
    try:
        await chunker.chunk_file("code", "COBOL")
    except ValueError as e:
        print(f"\nExpected error for unsupported language: {e}")
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    asyncio.run(main())