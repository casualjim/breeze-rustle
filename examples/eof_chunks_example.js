// Example demonstrating how to use EOF chunks to access file content and hash
const { walkProject, ChunkType } = require("../crates/breeze-napi");

async function demonstrateEOFChunks() {
  console.log("Walking current directory to demonstrate EOF chunks...\n");

  const fileData = new Map();
  const chunks = [];

  // Walk the project directory
  for await (const projectChunk of await walkProject(".", 1000)) {
    const { filePath, chunk } = projectChunk;

    if (chunk.chunkType === ChunkType.EndOfFile) {
      // EOF chunk contains the full file content and hash
      const contentHash = Buffer.from(chunk.contentHash).toString("hex");
      fileData.set(filePath, {
        content: chunk.content,
        hash: contentHash,
        chunkCount: chunks.filter((c) => c.filePath === filePath).length,
      });

      console.log(`📄 File: ${filePath}`);
      console.log(`   Hash: ${contentHash.substring(0, 16)}...`);
      console.log(`   Size: ${chunk.content.length} bytes`);
      console.log(`   Chunks: ${fileData.get(filePath).chunkCount}`);
      console.log();
    } else {
      chunks.push(projectChunk);
    }
  }

  console.log("\nSummary:");
  console.log(`Total files processed: ${fileData.size}`);
  console.log(`Total chunks created: ${chunks.length}`);

  // Example: Verify chunk content is part of file content
  if (chunks.length > 0 && fileData.size > 0) {
    const firstChunk = chunks[0];
    const fileInfo = fileData.get(firstChunk.filePath);
    if (fileInfo) {
      console.log(`\nVerifying first chunk is part of its file content:`);
      console.log(`Chunk text: "${firstChunk.chunk.text.substring(0, 50)}..."`);
      console.log(`Is part of file: ${fileInfo.content.includes(firstChunk.chunk.text)}`);
    }
  }
}

demonstrateEOFChunks().catch(console.error);
