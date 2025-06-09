#!/usr/bin/env python3
"""Test metadata extraction capabilities"""
import asyncio
import breeze_rustle

async def main():
    chunker = breeze_rustle.SemanticChunker(max_chunk_size=500)
    
    # Test Python code with various structures
    python_code = '''
import os
import sys

def hello(name: str) -> str:
    """Say hello to someone"""
    message = f"Hello {name}!"
    print(message)
    return message

class Greeter:
    """A class for greeting people"""
    
    def __init__(self, language="en"):
        self.language = language
        self.greetings = {
            "en": "Hello",
            "fr": "Bonjour",
            "es": "Hola"
        }
    
    def greet(self, name: str) -> str:
        """Greet someone in the configured language"""
        greeting = self.greetings.get(self.language, "Hello")
        return f"{greeting} {name}!"
    
    def set_language(self, lang: str):
        """Change the greeting language"""
        if lang in self.greetings:
            self.language = lang
        else:
            raise ValueError(f"Unsupported language: {lang}")

def main():
    """Main entry point"""
    greeter = Greeter()
    print(greeter.greet("World"))
    
    greeter.set_language("fr")
    print(greeter.greet("Monde"))

if __name__ == "__main__":
    main()
'''
    
    chunks = await chunker.chunk_file(python_code, "Python", "test_greeting.py")
    
    print(f"Got {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(f"  Lines: {chunk.start_line}-{chunk.end_line}")
        print(f"  Node type: {chunk.metadata.node_type}")
        print(f"  Node name: {chunk.metadata.node_name}")
        print(f"  Parent context: {chunk.metadata.parent_context}")
        print(f"  Scope path: {chunk.metadata.scope_path}")
        print(f"  Definitions: {chunk.metadata.definitions}")
        print(f"  References: {chunk.metadata.references}")
        print(f"  Text ({len(chunk.text)} chars): {chunk.text[:60]}...")
        print()
    
    # Test JavaScript code
    print("\n" + "="*60 + "\n")
    
    js_code = '''
const express = require('express');

class UserService {
    constructor(database) {
        this.db = database;
        this.cache = new Map();
    }
    
    async getUser(id) {
        if (this.cache.has(id)) {
            return this.cache.get(id);
        }
        
        const user = await this.db.findUser(id);
        this.cache.set(id, user);
        return user;
    }
    
    async createUser(userData) {
        const user = await this.db.createUser(userData);
        this.cache.set(user.id, user);
        return user;
    }
}

function setupRoutes(app, userService) {
    app.get('/users/:id', async (req, res) => {
        try {
            const user = await userService.getUser(req.params.id);
            res.json(user);
        } catch (error) {
            res.status(404).json({ error: 'User not found' });
        }
    });
    
    app.post('/users', async (req, res) => {
        try {
            const user = await userService.createUser(req.body);
            res.status(201).json(user);
        } catch (error) {
            res.status(400).json({ error: error.message });
        }
    });
}

module.exports = { UserService, setupRoutes };
'''
    
    chunks = await chunker.chunk_file(js_code, "JavaScript", "user_service.js")
    
    print(f"JavaScript: Got {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(f"  Lines: {chunk.start_line}-{chunk.end_line}")
        print(f"  Node type: {chunk.metadata.node_type}")
        print(f"  Node name: {chunk.metadata.node_name}")
        print(f"  Parent context: {chunk.metadata.parent_context}")
        print(f"  Definitions: {chunk.metadata.definitions[:5]}...")
        print()

if __name__ == "__main__":
    asyncio.run(main())