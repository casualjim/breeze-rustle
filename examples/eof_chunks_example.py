"""Example demonstrating how to use EOF chunks to access file content and hash"""
import asyncio
import hashlib
from breeze import SemanticChunker, ChunkType


async def demonstrate_eof_chunks():
    print("Walking current directory to demonstrate EOF chunks...\n")
    
    chunker = SemanticChunker(max_chunk_size=1000)
    file_data = {}
    chunks = []
    
    # Walk the project directory
    walker = await chunker.walk_project(".")
    async for project_chunk in walker:
        file_path = project_chunk.file_path
        chunk = project_chunk.chunk
        
        if chunk.chunk_type == ChunkType.EOF:
            # EOF chunk contains the full file content and hash
            content_hash = chunk.content_hash.hex() if chunk.content_hash else ""
            file_data[file_path] = {
                'content': chunk.content,
                'hash': content_hash,
                'chunk_count': len([c for c in chunks if c.file_path == file_path])
            }
            
            print(f"📄 File: {file_path}")
            print(f"   Hash: {content_hash[:16]}...")
            print(f"   Size: {len(chunk.content)} bytes")
            print(f"   Chunks: {file_data[file_path]['chunk_count']}")
            print()
        else:
            chunks.append(project_chunk)
    
    print("\nSummary:")
    print(f"Total files processed: {len(file_data)}")
    print(f"Total chunks created: {len(chunks)}")
    
    # Example: Verify chunk content is part of file content
    if chunks and file_data:
        first_chunk = chunks[0]
        file_info = file_data.get(first_chunk.file_path)
        if file_info:
            print(f"\nVerifying first chunk is part of its file content:")
            print(f"Chunk text: \"{first_chunk.chunk.text[:50]}...\"")
            print(f"Is part of file: {first_chunk.chunk.text in file_info['content']}")
            
            # Verify hash is correct
            computed_hash = hashlib.blake3(file_info['content'].encode()).hexdigest()
            print(f"\nHash verification:")
            print(f"EOF chunk hash:  {file_info['hash']}")
            print(f"Computed hash:   {computed_hash}")
            print(f"Hashes match:    {file_info['hash'] == computed_hash}")


if __name__ == "__main__":
    asyncio.run(demonstrate_eof_chunks())