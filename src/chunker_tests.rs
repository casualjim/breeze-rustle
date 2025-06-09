#[cfg(test)]
mod comprehensive_tests {
    use crate::chunker::{InnerChunker, TokenizerType};
    
    #[tokio::test]
    async fn test_python_class_chunking() {
        let chunker = InnerChunker::new(300, TokenizerType::Characters).unwrap();
        
        let code = r#"
class Calculator:
    """A simple calculator class"""
    
    def __init__(self):
        self.memory = 0
    
    def add(self, a, b):
        """Add two numbers"""
        result = a + b
        self.memory = result
        return result
    
    def subtract(self, a, b):
        """Subtract b from a"""
        return a - b
    
    def clear_memory(self):
        """Clear the memory"""
        self.memory = 0
"#;
        
        let chunks = chunker.chunk_file(code, "Python", Some("calculator.py"))
            .await
            .expect("Should chunk Python code");
        
        // Should have multiple chunks due to size limit
        assert!(chunks.len() >= 2);
        
        // Check first chunk contains class definition
        let first_chunk = &chunks[0];
        assert!(first_chunk.text.contains("class Calculator"));
        assert_eq!(first_chunk.metadata.language, "Python");
        
        // Check metadata extraction
        let has_calculator_def = chunks.iter().any(|c| 
            c.metadata.definitions.contains(&"Calculator".to_string())
        );
        assert!(has_calculator_def, "Should find Calculator class definition");
        
        // Check method definitions are found
        let all_definitions: Vec<_> = chunks.iter()
            .flat_map(|c| &c.metadata.definitions)
            .collect();
        
        assert!(all_definitions.iter().any(|d| d.contains("add")));
        assert!(all_definitions.iter().any(|d| d.contains("subtract")));
    }
    
    #[tokio::test]
    async fn test_javascript_async_chunking() {
        let chunker = InnerChunker::new(200, TokenizerType::Characters).unwrap();
        
        let code = r#"
async function fetchUserData(userId) {
    const response = await fetch(`/api/users/${userId}`);
    const data = await response.json();
    return data;
}

class UserManager {
    constructor(apiClient) {
        this.client = apiClient;
        this.cache = new Map();
    }
    
    async getUser(id) {
        if (this.cache.has(id)) {
            return this.cache.get(id);
        }
        
        const user = await fetchUserData(id);
        this.cache.set(id, user);
        return user;
    }
}

const manager = new UserManager(apiClient);
const user = await manager.getUser(123);
"#;
        
        let chunks = chunker.chunk_file(code, "JavaScript", Some("user_manager.js"))
            .await
            .expect("Should chunk JavaScript code");
        
        assert!(chunks.len() >= 3, "Should have multiple chunks");
        
        // Verify function and class definitions are captured
        let all_definitions: Vec<_> = chunks.iter()
            .flat_map(|c| &c.metadata.definitions)
            .collect();
        
        assert!(all_definitions.iter().any(|d| d == &"fetchUserData"));
        assert!(all_definitions.iter().any(|d| d == &"UserManager"));
        
        // Verify references are captured
        let all_references: Vec<_> = chunks.iter()
            .flat_map(|c| &c.metadata.references)
            .collect();
        
        assert!(all_references.iter().any(|r| r.contains("fetch")));
    }
    
    #[tokio::test]
    async fn test_rust_impl_chunking() {
        let chunker = InnerChunker::new(250, TokenizerType::Characters).unwrap();
        
        let code = r#"
use std::collections::HashMap;

pub struct Cache<K, V> {
    storage: HashMap<K, V>,
    capacity: usize,
}

impl<K, V> Cache<K, V> 
where
    K: Eq + std::hash::Hash,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            storage: HashMap::with_capacity(capacity),
            capacity,
        }
    }
    
    pub fn get(&self, key: &K) -> Option<&V> {
        self.storage.get(key)
    }
    
    pub fn insert(&mut self, key: K, value: V) {
        if self.storage.len() >= self.capacity {
            // Simple eviction: remove first item
            if let Some(first_key) = self.storage.keys().next().cloned() {
                self.storage.remove(&first_key);
            }
        }
        self.storage.insert(key, value);
    }
}
"#;
        
        let chunks = chunker.chunk_file(code, "Rust", Some("cache.rs"))
            .await
            .expect("Should chunk Rust code");
        
        // Verify struct and impl blocks are identified
        let node_types: Vec<_> = chunks.iter()
            .map(|c| c.metadata.node_type.as_str())
            .collect();
        
        assert!(node_types.iter().any(|&t| t.contains("struct") || t.contains("impl")));
        
        // Check that we have proper Rust code parsing
        let has_struct_or_impl = chunks.iter().any(|c| 
            c.metadata.definitions.iter().any(|d| d == "Cache" || d == "new" || d == "get" || d == "insert")
        );
        assert!(has_struct_or_impl, "Should find Rust struct or method definitions");
    }
    
    #[tokio::test]
    async fn test_nested_scope_extraction() {
        let chunker = InnerChunker::new(500, TokenizerType::Characters).unwrap();
        
        let code = r#"
module OuterModule {
    export namespace InnerNamespace {
        export class NestedClass {
            private data: string[];
            
            constructor() {
                this.data = [];
            }
            
            public addItem(item: string): void {
                this.data.push(item);
            }
            
            public getItems(): string[] {
                return [...this.data];
            }
        }
        
        export function helperFunction(): NestedClass {
            return new NestedClass();
        }
    }
}
"#;
        
        let chunks = chunker.chunk_file(code, "TypeScript", Some("nested.ts"))
            .await
            .expect("Should chunk TypeScript code");
        
        // Check for nested class definition
        let has_nested_class = chunks.iter().any(|c| {
            c.metadata.definitions.contains(&"NestedClass".to_string()) ||
            c.metadata.node_name.as_ref().map_or(false, |n| n == "NestedClass")
        });
        
        assert!(has_nested_class, "Should find NestedClass definition");
    }
    
    #[tokio::test]
    async fn test_chunk_boundaries_preserve_semantics() {
        let chunker = InnerChunker::new(150, TokenizerType::Characters).unwrap(); // Small chunks to force splitting
        
        let code = r#"
def process_data(items):
    """Process a list of items"""
    results = []
    
    for item in items:
        # This is a long comment that explains what we're doing
        # It might cause the chunk to split at an interesting boundary
        processed = transform(item)
        validated = validate(processed)
        
        if validated:
            results.append(validated)
        else:
            log_error(f"Invalid item: {item}")
    
    return results

def transform(item):
    """Transform an item"""
    return item.upper()

def validate(item):
    """Validate an item"""
    return len(item) > 0
"#;
        
        let chunks = chunker.chunk_file(code, "Python", Some("processor.py"))
            .await
            .expect("Should chunk Python code");
        
        // Verify we get multiple chunks
        assert!(chunks.len() >= 3, "Should split into multiple chunks");
        
        // Each chunk should have valid Python syntax (no partial statements)
        for chunk in &chunks {
            // Basic check: balanced parentheses
            let open_parens = chunk.text.matches('(').count();
            let close_parens = chunk.text.matches(')').count();
            assert_eq!(open_parens, close_parens, 
                "Chunk should have balanced parentheses: {}", chunk.text);
        }
    }
    
    #[tokio::test]
    async fn test_large_file_performance() {
        use std::time::Instant;
        
        let chunker = InnerChunker::new(1500, TokenizerType::Characters).unwrap();
        
        // Generate a large Python file (approximately 1MB)
        let mut code = String::with_capacity(1_000_000);
        for i in 0..1000 {
            code.push_str(&format!(r#"
def function_{i}(param1, param2):
    """Function number {i}"""
    result = param1 + param2
    for j in range(100):
        result += j * {i}
    return result

class Class_{i}:
    def __init__(self):
        self.value = {i}
    
    def method1(self, x):
        return x * self.value
    
    def method2(self, y):
        return y + self.value

"#, i = i));
        }
        
        let start = Instant::now();
        let chunks = chunker.chunk_file(&code, "Python", Some("large_file.py"))
            .await
            .expect("Should chunk large file");
        let duration = start.elapsed();
        
        // Should complete within 10 seconds for 1MB file (more realistic with full AST parsing)
        assert!(duration.as_secs() < 10, 
            "Chunking took {}ms, should be under 10s", duration.as_millis());
        
        // Should produce reasonable number of chunks
        assert!(chunks.len() > 50, "Should produce many chunks for large file");
        assert!(chunks.len() < 2000, "Should not produce too many chunks");
        
        // Verify metadata is still extracted
        let has_definitions = chunks.iter().any(|c| !c.metadata.definitions.is_empty());
        assert!(has_definitions, "Should extract definitions even for large files");
    }
}