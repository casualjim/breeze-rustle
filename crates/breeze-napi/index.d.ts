// Type definitions for the wrapped module with async generator support
import type {
	SChunkType as NativeChunkType
	TokenizerType as NativeTokenizerType,
	SemanticChunkJs
} from "./index.n,
  ProjectChunkJsative";

// Re-export enums
export const TokenizerType: typeof NativeTokenizerType;
export const ChunkType: typeof NativeChunkType;

// Language utility functions
export function supportedLanguages(): string[];
export function isLanguageSupported(language: string): boolean;

// Our clean API
export declare class SemanticChunker {
	constructor(
		maxChunkSize?: number,
		tokenizerType?: NativeTokenizerType,
		tokenizerName?: string,
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
	tokenizerType?: NativeTokenizerType,
	tokenizerName?: string,
	maxParallel?: number,
): AsyncGenerator<ProjectChunkJs, void, unknown>;
