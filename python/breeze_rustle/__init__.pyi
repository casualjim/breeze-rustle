"""Type stubs for breeze-rustle"""

from typing import List, Optional, Coroutine, Any
from enum import Enum

__version__: str

class TokenizerType(Enum):
    """Tokenizer types for chunking."""
    CHARACTERS = "characters"
    TIKTOKEN = "tiktoken"
    HUGGINGFACE = "huggingface"
    
    @property
    def value(self) -> str:
        """Get the string value of the tokenizer type."""
        ...

class ChunkMetadata:
    """Metadata about a semantic code chunk."""
    node_type: str
    node_name: Optional[str]
    language: str
    parent_context: Optional[str]
    scope_path: List[str]
    definitions: List[str]
    references: List[str]

class SemanticChunk:
    """A semantic chunk of code with metadata."""
    text: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    metadata: ChunkMetadata

class SemanticChunker:
    """High-performance semantic code chunker."""
    
    def __init__(
        self,
        max_chunk_size: Optional[int] = None,
        tokenizer: Optional[TokenizerType] = None,
        hf_model: Optional[str] = None
    ) -> None:
        """
        Initialize a new SemanticChunker.
        
        Args:
            max_chunk_size: Maximum chunk size in tokens/characters (default: 1500)
            tokenizer: Tokenization method to use:
                - TokenizerType.CHARACTERS (default): Character-based chunking
                - TokenizerType.TIKTOKEN: OpenAI's tiktoken (cl100k_base) tokenizer
                - TokenizerType.HUGGINGFACE: HuggingFace tokenizer (requires hf_model)
            hf_model: HuggingFace model name (required when tokenizer is TokenizerType.HUGGINGFACE)
        
        Raises:
            ValueError: If TokenizerType.HUGGINGFACE is used without hf_model
            RuntimeError: If tokenizer initialization fails
        """
        ...
    
    def chunk_file(
        self,
        content: str,
        language: str,
        file_path: Optional[str] = None
    ) -> Coroutine[Any, Any, List[SemanticChunk]]:
        """
        Asynchronously chunk a file into semantic units.
        
        Args:
            content: The source code content to chunk
            language: Programming language of the content (e.g., "Python", "JavaScript")
            file_path: Optional path to the source file
        
        Returns:
            A coroutine that resolves to a list of semantic chunks
        
        Raises:
            ValueError: If the language is not supported
            RuntimeError: If parsing fails
        """
        ...
    
    def chunk_text(
        self,
        content: str,
        file_path: Optional[str] = None
    ) -> Coroutine[Any, Any, List[SemanticChunk]]:
        """
        Asynchronously chunk plain text into semantic units.
        
        This method provides text-based chunking for any content, regardless
        of programming language support. Use this for unsupported languages
        or when you want simple text chunking.
        
        Args:
            content: The text content to chunk
            file_path: Optional path to the source file
        
        Returns:
            A coroutine that resolves to a list of text chunks with minimal metadata
        
        Raises:
            RuntimeError: If chunking fails
        """
        ...
    
    @staticmethod
    def supported_languages() -> List[str]:
        """
        Get a list of supported programming languages.
        
        Returns:
            List of language names (e.g., ["Python", "JavaScript", "Rust"])
        """
        ...
    
    @staticmethod
    def is_language_supported(language: str) -> bool:
        """
        Check if a language is supported.
        
        Args:
            language: Language name to check (case-sensitive)
        
        Returns:
            True if the language is supported, False otherwise
        """
        ...