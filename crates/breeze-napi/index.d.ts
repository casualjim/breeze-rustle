// Type definitions for the wrapped module with async generator support
import type {
  ChunkType as NativeChunkType,
  TokenizerType as NativeTokenizerType,
  ProjectChunkJs,
  SemanticChunkJs
} from "./index.native";
// Re-export enums
export type TokenizerType = typeof NativeTokenizerType;
export type ChunkType = typeof NativeChunkType;

// Language utility functions
export function supportedLanguages(): string[];
export function isLanguageSupported(language: string): boolean;

// Our clean API
export declare class SemanticChunker {
	constructor(
		maxChunkSize?: number,
		tokenizer?: NativeTokenizerType,
		hfModel?: string,
	);

	/** Chunk code and return an async generator */
	chunkCode(
		content: string,
		language: string,
		filePath?: string,
	): AsyncGenerator<SemanticChunkJs, void, unknown>;

	/** Chunk text and return an async generator */
	chunkText(
		content: string,
		filePath?: string,
	): AsyncGenerator<SemanticChunkJs, void, unknown>;
}

/** Walk a project directory and return an async generator of chunks */
export declare function walkProject(
	path: string,
	maxChunkSize?: number,
	tokenizer?: NativeTokenizerType,
	hfModel?: string,
	maxParallel?: number,
): AsyncGenerator<ProjectChunkJs, void, unknown>;
