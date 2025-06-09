# breeze-rustle Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [API Reference](#api-reference)
4. [Usage Patterns](#usage-patterns)
5. [Performance Guide](#performance-guide)
6. [Integration Guide](#integration-guide)
7. [Troubleshooting](#troubleshooting)

## Introduction

breeze-rustle is a high-performance code chunking library designed for RAG (Retrieval-Augmented Generation) applications. It intelligently splits source code into semantic units while preserving context and extracting metadata.

### Key Concepts

- **Semantic Chunking**: Splits code along natural boundaries (functions, classes, methods)
- **Metadata Extraction**: Captures node types, names, scopes, and relationships
- **Language Agnostic**: Supports 16+ programming languages via tree-sitter
- **Tokenizer Flexibility**: Choose the right tokenizer for your use case

## Architecture

### Component Overview

```
┌─────────────────────┐
│   Python API        │  (breeze_rustle module)
├─────────────────────┤
│   PyO3 Bindings     │  (Rust-Python interface)
├─────────────────────┤
│   Rust Core         │
│  ├─ Chunker         │  (Wraps text-splitter)
│  ├─ Languages       │  (Language registry)
│  ├─ Metadata        │  (AST analysis)
│  └─ Types           │  (Data structures)
├─────────────────────┤
│   Dependencies      │
│  ├─ text-splitter   │  (Core chunking logic)
│  ├─ tree-sitter-*   │  (Language parsers)
│  └─ tokenizers      │  (Size measurement)
└─────────────────────┘
```

### Data Flow

1. **Input**: Source code + language identification
2. **Parsing**: Tree-sitter creates AST
3. **Chunking**: text-splitter divides code semantically
4. **Metadata**: Extract node information from AST
5. **Output**: Chunks with rich metadata

## API Reference

### Core Classes

#### SemanticChunker

The main interface for code chunking.

```python
class SemanticChunker:
    def __init__(
        self,
        max_chunk_size: int = 1500,
        tokenizer: TokenizerType = TokenizerType.CHARACTERS,
        hf_model: Optional[str] = None
    ): ...
    
    async def chunk_file(
        self,
        content: str,
        language: str,
        file_path: Optional[str] = None
    ) -> List[SemanticChunk]: ...
    
    async def chunk_text(
        self,
        content: str,
        file_path: Optional[str] = None
    ) -> List[SemanticChunk]: ...
    
    @staticmethod
    def supported_languages() -> List[str]: ...
    
    @staticmethod
    def is_language_supported(language: str) -> bool: ...
```

#### SemanticChunk

Represents a single chunk of code.

```python
@dataclass
class SemanticChunk:
    text: str                # The chunk content
    start_byte: int          # Byte offset in original
    end_byte: int            # End byte offset
    start_line: int          # Line number (1-indexed)
    end_line: int            # End line number
    metadata: ChunkMetadata  # Additional information
```

#### ChunkMetadata

Rich metadata about each chunk.

```python
@dataclass
class ChunkMetadata:
    node_type: str                    # e.g., "function_definition"
    node_name: Optional[str]          # e.g., "calculate_sum"
    language: str                     # e.g., "Python"
    parent_context: Optional[str]     # e.g., "MyClass"
    scope_path: List[str]            # e.g., ["module", "MyClass", "method"]
    definitions: List[str]           # Symbols defined
    references: List[str]            # Symbols referenced
```

#### TokenizerType

Enum for tokenizer selection.

```python
class TokenizerType(Enum):
    CHARACTERS = "CHARACTERS"      # Count characters
    TIKTOKEN = "TIKTOKEN"         # OpenAI's tokenizer
    HUGGINGFACE = "HUGGINGFACE"   # HF tokenizers
```

## Usage Patterns

### Pattern 1: Simple File Processing

```python
async def process_single_file(file_path: str):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Detect language from extension
    lang = detect_language_from_extension(file_path)
    
    chunker = SemanticChunker()
    chunks = await chunker.chunk_file(content, lang, file_path)
    
    return chunks
```

### Pattern 2: Batch Processing with Progress

```python
async def process_directory_with_progress(directory: Path):
    from tqdm.asyncio import tqdm
    
    chunker = SemanticChunker(max_chunk_size=1000)
    files = list(directory.rglob("*.py"))
    
    async def process_file(file_path):
        try:
            content = file_path.read_text()
            return await chunker.chunk_file(content, "Python", str(file_path))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []
    
    # Process files concurrently
    tasks = [process_file(f) for f in files]
    results = await tqdm.gather(*tasks, desc="Processing files")
    
    # Flatten results
    all_chunks = [chunk for chunks in results for chunk in chunks]
    return all_chunks
```

### Pattern 3: Custom Tokenizer Configuration

```python
# For OpenAI embeddings
openai_chunker = SemanticChunker(
    max_chunk_size=8191,  # OpenAI's limit
    tokenizer=TokenizerType.TIKTOKEN
)

# For BERT-based models
bert_chunker = SemanticChunker(
    max_chunk_size=512,
    tokenizer=TokenizerType.HUGGINGFACE,
    hf_model="bert-base-uncased"
)

# For code-specific models
code_chunker = SemanticChunker(
    max_chunk_size=2048,
    tokenizer=TokenizerType.HUGGINGFACE,
    hf_model="microsoft/codebert-base"
)
```

### Pattern 4: Metadata-Based Filtering

```python
async def extract_classes_and_functions(code: str, language: str):
    chunker = SemanticChunker()
    chunks = await chunker.chunk_file(code, language)
    
    classes = []
    functions = []
    
    for chunk in chunks:
        if chunk.metadata.node_type == "class_definition":
            classes.append({
                "name": chunk.metadata.node_name,
                "code": chunk.text,
                "methods": []
            })
        elif chunk.metadata.node_type in ["function_definition", "method_definition"]:
            func_info = {
                "name": chunk.metadata.node_name,
                "code": chunk.text,
                "parent": chunk.metadata.parent_context
            }
            
            if chunk.metadata.parent_context:
                # It's a method, add to its class
                for cls in classes:
                    if cls["name"] == chunk.metadata.parent_context:
                        cls["methods"].append(func_info)
            else:
                # It's a standalone function
                functions.append(func_info)
    
    return {"classes": classes, "functions": functions}
```

### Pattern 5: Language Detection Integration

```python
from breeze_langdetect import detect_language

async def smart_chunk_file(file_path: str):
    """Intelligently chunk a file with automatic language detection."""
    # Read content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Detect language
    detected_lang = detect_language(file_path)
    
    chunker = SemanticChunker()
    
    # Try semantic chunking first
    if detected_lang and SemanticChunker.is_language_supported(detected_lang):
        try:
            return await chunker.chunk_file(content, detected_lang, file_path)
        except Exception as e:
            print(f"Semantic chunking failed: {e}, falling back to text chunking")
    
    # Fall back to text chunking
    return await chunker.chunk_text(content, file_path)
```

## Performance Guide

### Optimization Tips

1. **Choose the Right Tokenizer**
   - Characters: Fastest, good for small chunks
   - Tiktoken: Best for OpenAI models
   - HuggingFace: Most accurate for specific models

2. **Batch Processing**
   ```python
   # Process files concurrently
   async def batch_process(files, max_concurrent=10):
       semaphore = asyncio.Semaphore(max_concurrent)
       
       async def process_with_limit(file):
           async with semaphore:
               return await process_file(file)
       
       tasks = [process_with_limit(f) for f in files]
       return await asyncio.gather(*tasks)
   ```

3. **Memory Management**
   ```python
   # Process large files in streaming fashion
   async def process_large_file(file_path, chunk_size=1024*1024):
       chunker = SemanticChunker()
       all_chunks = []
       
       with open(file_path, 'r') as f:
           while True:
               content = f.read(chunk_size)
               if not content:
                   break
               
               # Process partial content
               chunks = await chunker.chunk_text(content)
               all_chunks.extend(chunks)
       
       return all_chunks
   ```

### Benchmarking

```python
import time
import asyncio

async def benchmark_chunker(content, language, iterations=100):
    chunker = SemanticChunker()
    
    # Warm up
    await chunker.chunk_file(content, language)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        await chunker.chunk_file(content, language)
    end = time.perf_counter()
    
    avg_time = (end - start) / iterations
    print(f"Average time: {avg_time*1000:.2f}ms")
    print(f"Throughput: {len(content) / avg_time / 1024:.2f} KB/s")
```

## Integration Guide

### RAG Pipeline Integration

```python
from typing import List, Dict
import numpy as np

class CodeRAGPipeline:
    def __init__(self, embedding_model, vector_store):
        self.chunker = SemanticChunker(
            max_chunk_size=512,
            tokenizer=TokenizerType.HUGGINGFACE,
            hf_model=embedding_model.model_name
        )
        self.embedding_model = embedding_model
        self.vector_store = vector_store
    
    async def index_codebase(self, codebase_path: str):
        """Index an entire codebase for RAG."""
        all_chunks = []
        
        # Collect all code files
        for file_path in Path(codebase_path).rglob("*"):
            if file_path.is_file() and self.is_code_file(file_path):
                chunks = await self.process_file(file_path)
                all_chunks.extend(chunks)
        
        # Generate embeddings
        texts = [c.text for c in all_chunks]
        embeddings = self.embedding_model.encode(texts)
        
        # Store in vector database
        metadata = [{
            "file": c.file_path,
            "node_type": c.metadata.node_type,
            "node_name": c.metadata.node_name,
            "language": c.metadata.language,
            "scope": "/".join(c.metadata.scope_path)
        } for c in all_chunks]
        
        self.vector_store.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadata=metadata
        )
    
    async def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for relevant code chunks."""
        query_embedding = self.embedding_model.encode([query])[0]
        
        results = self.vector_store.similarity_search(
            query_embedding,
            k=k
        )
        
        return results
```

### LangChain Integration

```python
from langchain.text_splitter import TextSplitter
from typing import List

class BreezeRustleSplitter(TextSplitter):
    """LangChain-compatible splitter using breeze-rustle."""
    
    def __init__(self, language: str = "Python", **kwargs):
        super().__init__(**kwargs)
        self.language = language
        self.chunker = SemanticChunker(
            max_chunk_size=self.chunk_size
        )
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        # Run async in sync context
        import asyncio
        chunks = asyncio.run(
            self.chunker.chunk_file(text, self.language)
        )
        return [chunk.text for chunk in chunks]

# Usage with LangChain
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS

# Load and split
loader = TextLoader("code.py")
documents = loader.load()
splitter = BreezeRustleSplitter(language="Python", chunk_size=1000)
splits = splitter.split_documents(documents)

# Create vector store
vectorstore = FAISS.from_documents(splits, embedding_model)
```

## Troubleshooting

### Common Issues

1. **Unsupported Language Error**
   ```python
   # Solution: Check language support first
   if SemanticChunker.is_language_supported(language):
       chunks = await chunker.chunk_file(content, language)
   else:
       chunks = await chunker.chunk_text(content)
   ```

2. **Large File Performance**
   ```python
   # Solution: Use smaller chunk sizes or process in parts
   chunker = SemanticChunker(max_chunk_size=500)
   ```

3. **Memory Issues with HuggingFace Tokenizer**
   ```python
   # Solution: Use batch processing
   async def process_in_batches(files, batch_size=10):
       for i in range(0, len(files), batch_size):
           batch = files[i:i+batch_size]
           await process_batch(batch)
           # Force garbage collection if needed
           import gc
           gc.collect()
   ```

4. **Async Context Errors**
   ```python
   # Solution: Proper async context management
   async def main():
       chunker = SemanticChunker()
       # Your async code here
   
   # Run properly
   asyncio.run(main())
   ```

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Verbose chunk inspection
async def debug_chunk(content, language):
    chunker = SemanticChunker()
    chunks = await chunker.chunk_file(content, language)
    
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i} ---")
        print(f"Type: {chunk.metadata.node_type}")
        print(f"Name: {chunk.metadata.node_name}")
        print(f"Parent: {chunk.metadata.parent_context}")
        print(f"Scope: {' > '.join(chunk.metadata.scope_path)}")
        print(f"Lines: {chunk.start_line}-{chunk.end_line}")
        print(f"Text preview: {chunk.text[:100]}...")
        print(f"Definitions: {chunk.metadata.definitions}")
        print(f"References: {chunk.metadata.references}")
```

### Performance Profiling

```python
import cProfile
import pstats
from pstats import SortKey

async def profile_chunking():
    chunker = SemanticChunker()
    
    # Read a large file
    with open("large_file.py", "r") as f:
        content = f.read()
    
    # Profile the chunking
    profiler = cProfile.Profile()
    profiler.enable()
    
    chunks = await chunker.chunk_file(content, "Python")
    
    profiler.disable()
    
    # Print stats
    stats = pstats.Stats(profiler)
    stats.sort_stats(SortKey.TIME)
    stats.print_stats(10)  # Top 10 time-consuming functions
```