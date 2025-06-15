import pytest
import tempfile
import os
from pathlib import Path
from breeze_langdetect import detect_language
from breeze import SemanticChunker


class TestHyperpolyglotIntegration:
    """Test integration between breeze-langdetect (hyperpolyglot) and breeze-rustle"""

    @pytest.mark.asyncio
    async def test_python_file_detection_and_chunking(self):
        """Should detect Python language and chunk accordingly"""
        content = '''#!/usr/bin/env python3
"""A simple Python module for testing."""

def factorial(n):
    """Calculate factorial of n."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

class Calculator:
    def add(self, a, b):
        return a + b
'''
        # Create temporary file for detection
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            # Detect language using hyperpolyglot
            detected_lang = detect_language(temp_path)
            assert detected_lang == "Python"

            # Use detected language for chunking
            chunker = SemanticChunker()
            chunk_stream = await chunker.chunk_code(content, detected_lang)
            chunks = []
            async for chunk in chunk_stream:
                chunks.append(chunk)

            assert len(chunks) > 0
            assert all(chunk.metadata.language == "Python" for chunk in chunks)
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_javascript_detection_and_chunking(self):
        """Should detect JavaScript and chunk accordingly"""
        content = """// JavaScript test file
const greeting = (name) => {
    return `Hello, ${name}!`;
};

class Person {
    constructor(name) {
        this.name = name;
    }

    greet() {
        return greeting(this.name);
    }
}

module.exports = { Person, greeting };
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            detected_lang = detect_language(temp_path)
            assert detected_lang == "JavaScript"

            chunker = SemanticChunker()
            chunk_stream = await chunker.chunk_code(content, detected_lang)
            chunks = []
            async for chunk in chunk_stream:
                chunks.append(chunk)

            assert len(chunks) > 0
            assert all(chunk.metadata.language == "JavaScript" for chunk in chunks)
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_rust_detection_and_chunking(self):
        """Should detect Rust and chunk accordingly"""
        content = """//! A simple Rust module

use std::collections::HashMap;

pub struct Cache<T> {
    data: HashMap<String, T>,
}

impl<T> Cache<T> {
    pub fn new() -> Self {
        Cache {
            data: HashMap::new(),
        }
    }

    pub fn insert(&mut self, key: String, value: T) {
        self.data.insert(key, value);
    }
}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".rs", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            detected_lang = detect_language(temp_path)
            assert detected_lang == "Rust"

            chunker = SemanticChunker()
            chunk_stream = await chunker.chunk_code(content, detected_lang)
            chunks = []
            async for chunk in chunk_stream:
                chunks.append(chunk)

            assert len(chunks) > 0
            assert all(chunk.metadata.language == "Rust" for chunk in chunks)
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_typescript_detection_and_chunking(self):
        """Should detect TypeScript and chunk accordingly"""
        content = """interface User {
    id: number;
    name: string;
    email?: string;
}

export class UserService {
    private users: Map<number, User> = new Map();

    addUser(user: User): void {
        this.users.set(user.id, user);
    }

    getUser(id: number): User | undefined {
        return this.users.get(id);
    }
}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ts", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            detected_lang = detect_language(temp_path)
            assert detected_lang == "TypeScript"

            chunker = SemanticChunker()
            chunk_stream = await chunker.chunk_code(content, detected_lang)
            chunks = []
            async for chunk in chunk_stream:
                chunks.append(chunk)

            assert len(chunks) > 0
            assert all(chunk.metadata.language == "TypeScript" for chunk in chunks)
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_ambiguous_file_with_shebang(self):
        """Should use shebang to disambiguate language"""
        content = """#!/usr/bin/env python3
# Python script
def hello(name):
    print(f"Hello, {name}!")

class Greeter:
    def __init__(self, language):
        self.language = language

    def greet(self, name):
        hello(name)
"""
        # Create file without extension
        with tempfile.NamedTemporaryFile(mode="w", suffix="", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            detected_lang = detect_language(temp_path)
            # Since it's Python with a shebang, it should be detected
            assert detected_lang == "Python"

            chunker = SemanticChunker()
            chunk_stream = await chunker.chunk_code(content, detected_lang)
            chunks = []
            async for chunk in chunk_stream:
                chunks.append(chunk)

            assert len(chunks) > 0
            assert all(chunk.metadata.language == "Python" for chunk in chunks)
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_unsupported_language_fallback(self):
        """Should handle unsupported languages gracefully"""
        content = """IDENTIFICATION DIVISION.
PROGRAM-ID. HELLO-WORLD.
PROCEDURE DIVISION.
    DISPLAY "Hello, World!".
    STOP RUN.
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cob", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            detected_lang = detect_language(temp_path)
            chunker = SemanticChunker()

            if detected_lang and SemanticChunker.is_language_supported(detected_lang):
                # If somehow supported, use regular chunking
                chunk_stream = await chunker.chunk_code(content, detected_lang)
            else:
                # Fall back to text chunking
                chunk_stream = await chunker.chunk_text(content, "hello.cob")

            chunks = []
            async for chunk in chunk_stream:
                chunks.append(chunk)

            assert len(chunks) > 0
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_mixed_language_handling(self):
        """Should handle files with mixed content appropriately"""
        # HTML with embedded JavaScript
        content = """<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
    <script>
        function initPage() {
            console.log("Page initialized");
        }
    </script>
</head>
<body onload="initPage()">
    <h1>Hello World</h1>
</body>
</html>
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            detected_lang = detect_language(temp_path)
            chunker = SemanticChunker()

            if detected_lang and SemanticChunker.is_language_supported(detected_lang):
                chunk_stream = await chunker.chunk_code(content, detected_lang)
            else:
                chunk_stream = await chunker.chunk_text(content, "index.html")

            chunks = []
            async for chunk in chunk_stream:
                chunks.append(chunk)

            assert len(chunks) > 0
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_language_confidence_threshold(self):
        """Should handle low-confidence detections appropriately"""
        # Generic config file that might have low confidence
        content = """# Configuration
debug = true
port = 8080
host = "localhost"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix="", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            detected_lang = detect_language(temp_path)
            chunker = SemanticChunker()

            # If language detection confidence is low or unsupported, use text chunking
            if detected_lang and SemanticChunker.is_language_supported(detected_lang):
                chunk_stream = await chunker.chunk_code(content, detected_lang)
            else:
                chunk_stream = await chunker.chunk_text(content, "config")

            chunks = []
            async for chunk in chunk_stream:
                chunks.append(chunk)

            assert len(chunks) > 0
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_batch_processing_workflow(self):
        """Test a realistic workflow of processing multiple files"""
        test_files = [
            (
                "utils.py",
                """def format_date(date):
    return date.strftime("%Y-%m-%d")
""",
            ),
            (
                "helpers.js",
                """export const formatCurrency = (amount) => {
    return `$${amount.toFixed(2)}`;
};
""",
            ),
            (
                "types.ts",
                """export interface Product {
    id: string;
    name: string;
    price: number;
}
""",
            ),
            (
                "README.md",
                """# Project Documentation
This is a test project.
""",
            ),
        ]

        chunker = SemanticChunker()
        results = []
        temp_files = []

        try:
            # Create all temp files first
            for filename, content in test_files:
                suffix = Path(filename).suffix or ""
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=suffix, delete=False
                ) as f:
                    f.write(content)
                    temp_files.append((f.name, filename, content))

            # Process files
            for temp_path, original_name, content in temp_files:
                detected_lang = detect_language(temp_path)

                if detected_lang and SemanticChunker.is_language_supported(
                    detected_lang
                ):
                    chunk_stream = await chunker.chunk_code(content, detected_lang)
                    chunks = []
                    async for chunk in chunk_stream:
                        chunks.append(chunk)
                    results.append((original_name, detected_lang, "code", len(chunks)))
                else:
                    chunk_stream = await chunker.chunk_text(content, original_name)
                    chunks = []
                    async for chunk in chunk_stream:
                        chunks.append(chunk)
                    results.append(
                        (original_name, detected_lang or "unknown", "text", len(chunks))
                    )

            # Verify we processed all files
            assert len(results) == len(test_files)

            # Check specific results
            py_results = [r for r in results if r[0] == "utils.py"]
            assert py_results[0][1] == "Python"
            assert py_results[0][2] == "code"

            js_results = [r for r in results if r[0] == "helpers.js"]
            assert js_results[0][1] == "JavaScript"
            assert js_results[0][2] == "code"

            ts_results = [r for r in results if r[0] == "types.ts"]
            assert ts_results[0][1] == "TypeScript"
            assert ts_results[0][2] == "code"

            # Markdown is now supported as a semantic language
            md_results = [r for r in results if r[0] == "README.md"]
            assert md_results[0][2] == "code"
            assert md_results[0][1] == "Markdown"
        finally:
            # Clean up temp files
            for temp_path, _, _ in temp_files:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
