import { SemanticChunker, TokenizerType, walkProject } from "../index.mjs";

async function testAsyncGenerators() {
  console.log("Testing async generators with for-await-of (ESM)...\n");

  const chunker = new SemanticChunker();
  const code = `
function hello() {
    console.log("Hello world");
}

function goodbye() {
    console.log("Goodbye");
}
`;

  console.log("Testing chunkCode:");
  let count = 0;
  for await (const chunk of chunker.chunkCode(code, "javascript")) {
    count++;
    console.log(`Chunk ${count}:`, {
      type: chunk.chunkType,
      text: chunk.text.trim().slice(0, 50) + "...",
      lines: `${chunk.startLine}-${chunk.endLine}`,
    });
  }
  console.log(`Total chunks: ${count}\n`);

  console.log("Testing chunkText:");
  const text = `This is a test document.
It has multiple paragraphs.

Each paragraph should be a separate chunk.
This helps with text processing.`;

  count = 0;
  for await (const chunk of chunker.chunkText(text)) {
    count++;
    console.log(`Chunk ${count}:`, {
      type: chunk.chunkType,
      text: chunk.text.trim().slice(0, 50) + "...",
      lines: `${chunk.startLine}-${chunk.endLine}`,
    });
  }
  console.log(`Total chunks: ${count}\n`);

  console.log("Testing walkProject:");
  count = 0;
  const maxChunks = 5;

  for await (const projectChunk of walkProject(".", 500, TokenizerType.Characters)) {
    count++;
    console.log(`Chunk ${count} from ${projectChunk.filePath}:`, {
      type: projectChunk.chunk.chunkType,
      text: projectChunk.chunk.text.slice(0, 50) + "...",
      lines: `${projectChunk.chunk.startLine}-${projectChunk.chunk.endLine}`,
      language: projectChunk.chunk.metadata.language,
    });

    if (count >= maxChunks) {
      console.log(`(Stopping after ${maxChunks} chunks)`);
      break;
    }
  }

  console.log("\nAll async generator tests passed!");
}

testAsyncGenerators().catch(console.error);
