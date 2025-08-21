// Type definitions for the wrapped module with async generator support
import type {
  ChunkType as NativeChunkType,
  TokenizerType as NativeTokenizerType,
  ProjectChunkJs,
	SemanticChunkJs,
	SearchGranularity as NativeSearchGranularity,
} from "./index.native";
// Re-export enums
export type TokenizerType = typeof NativeTokenizerType;
export type ChunkType = typeof NativeChunkType;
export type SearchGranularity = typeof NativeSearchGranularity;

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

// ------------ Indexer & Search typings ------------

export interface ScopeDepth {
	min?: number;
	max?: number;
}

export interface SearchOptions {
	languages?: string[];
	fileLimit?: number;
	chunksPerFile?: number;
	granularity?: NativeSearchGranularity;
	nodeTypes?: string[];
	nodeNamePattern?: string;
	parentContextPattern?: string;
	scopeDepth?: ScopeDepth;
	hasDefinitions?: string[];
	hasReferences?: string[];
}

export interface ChunkResult {
	content: string;
	startLine: number;
	endLine: number;
	startByte: number;
	endByte: number;
	relevanceScore: number;
	nodeType: string;
	nodeName?: string;
	language: string;
	parentContext?: string;
	scopePath: string[];
	definitions: string[];
	references: string[];
}

export interface SearchResult {
	id: string;
	filePath: string;
	relevanceScore: number;
	chunkCount: number;
	chunks: ChunkResult[];
	fileSize: number;
	lastModifiedUs: number; // microseconds since epoch
	indexedAtUs: number; // microseconds since epoch
	languages: string[];
	primaryLanguage?: string;
}

export interface Project {
	id: string;
	name: string;
	directory: string;
	description?: string;
	status: string;
	createdAtUs: number;
	updatedAtUs: number;
	rescanIntervalUs?: number;
	lastIndexedAtUs?: number;
}

export interface IndexerCreationOptions {
	/** JSON string matching the Rust Config in breeze-indexer */
	configJson: string;
}

export declare class Indexer {
	static fromJson(options: IndexerCreationOptions): Promise<Indexer>;

	start(): Promise<void>;
	stop(): void;

	createProject(
		name: string,
		directory: string,
		description?: string,
		rescanIntervalSecs?: number,
	): Promise<Project>;

	listProjects(): Promise<Project[]>;
	getProject(id: string): Promise<Project | undefined>;

	indexProject(projectId: string): Promise<string>; // returns task id
	indexFile(projectId: string, filePath: string): Promise<string>; // returns task id

	search(query: string, options?: SearchOptions, projectId?: string): Promise<SearchResult[]>;
}
