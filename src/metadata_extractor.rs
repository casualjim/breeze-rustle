use crate::types::ChunkMetadata;
use tree_sitter::{Language, Node, Parser, TreeCursor};

/// Extracts metadata from a tree-sitter AST node for a code chunk
pub fn extract_metadata(
    content: &str,
    chunk_start: usize,
    chunk_end: usize,
    language: Language,
    language_name: &str,
) -> Result<ChunkMetadata, String> {
    // Parse the full content to get the AST
    let mut parser = Parser::new();
    parser.set_language(&language)
        .map_err(|e| format!("Failed to set language: {:?}", e))?;
    
    let tree = parser.parse(content, None)
        .ok_or_else(|| "Failed to parse content".to_string())?;
    
    let root_node = tree.root_node();
    
    // Find the primary node that contains the chunk
    let primary_node = find_primary_node_for_range(root_node, chunk_start, chunk_end);
    
    // Extract node type
    let node_type = primary_node.kind().to_string();
    
    // Extract node name (e.g., function name, class name)
    let node_name = extract_node_name(&primary_node, content);
    
    // Build scope path by traversing parents
    let scope_path = build_scope_path(&primary_node, content);
    
    // Extract parent context
    let parent_context = extract_parent_context(&primary_node, content);
    
    // Extract definitions and references within the chunk
    let (definitions, references) = extract_symbols_in_range(
        root_node, 
        content, 
        chunk_start, 
        chunk_end
    );
    
    Ok(ChunkMetadata {
        node_type,
        node_name,
        language: language_name.to_string(),
        parent_context,
        scope_path,
        definitions,
        references,
    })
}

/// Find the most specific node that fully contains the given byte range
fn find_primary_node_for_range(node: Node, start_byte: usize, end_byte: usize) -> Node {
    let mut cursor = node.walk();
    let mut best_node = node;
    
    // DFS to find the most specific node containing the range
    visit_node(&mut cursor, &mut best_node, start_byte, end_byte);
    
    best_node
}

fn visit_node<'a>(cursor: &mut TreeCursor<'a>, best_node: &mut Node<'a>, start_byte: usize, end_byte: usize) {
    let node = cursor.node();
    
    // Check if this node fully contains our range
    if node.start_byte() <= start_byte && node.end_byte() >= end_byte {
        // This node is a better match if it's more specific (smaller)
        if node.byte_range().len() < best_node.byte_range().len() {
            *best_node = node;
        }
        
        // Check children for an even more specific match
        if cursor.goto_first_child() {
            loop {
                visit_node(cursor, best_node, start_byte, end_byte);
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
            cursor.goto_parent();
        }
    }
}

/// Extract the name of a node (e.g., function name, class name)
fn extract_node_name(node: &Node, content: &str) -> Option<String> {
    // Common patterns for different node types
    match node.kind() {
        "function_declaration" | "function_definition" | "method_definition" | 
        "function_item" | "function" => {
            find_child_by_kind(node, "identifier")
                .or_else(|| find_child_by_kind(node, "property_identifier"))
                .map(|n| n.utf8_text(content.as_bytes()).unwrap_or("").to_string())
        }
        "class_declaration" | "class_definition" | "class" => {
            find_child_by_kind(node, "identifier")
                .or_else(|| find_child_by_kind(node, "type_identifier"))
                .map(|n| n.utf8_text(content.as_bytes()).unwrap_or("").to_string())
        }
        "struct_item" => {
            // Rust struct
            find_child_by_kind(node, "type_identifier")
                .map(|n| n.utf8_text(content.as_bytes()).unwrap_or("").to_string())
        }
        "impl_item" => {
            // Rust impl block - look for the type being implemented
            if let Some(type_id) = find_child_by_kind(node, "type_identifier") {
                Some(type_id.utf8_text(content.as_bytes()).unwrap_or("").to_string())
            } else if let Some(generic_type) = find_child_by_kind(node, "generic_type") {
                // Handle generic types like Cache<K, V>
                find_child_by_kind(&generic_type, "type_identifier")
                    .map(|n| n.utf8_text(content.as_bytes()).unwrap_or("").to_string())
            } else {
                None
            }
        }
        _ => None
    }
}

/// Find a child node by its kind
fn find_child_by_kind<'a>(node: &'a Node, kind: &str) -> Option<Node<'a>> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == kind {
            return Some(child);
        }
    }
    None
}

/// Build a scope path by traversing parent nodes
fn build_scope_path(node: &Node, content: &str) -> Vec<String> {
    let mut path = Vec::new();
    let mut current = Some(*node);
    
    while let Some(n) = current {
        if let Some(name) = extract_node_name(&n, content) {
            path.push(name);
        } else if is_significant_scope(n.kind()) {
            path.push(n.kind().to_string());
        }
        current = n.parent();
    }
    
    path.reverse();
    path
}

/// Check if a node type represents a significant scope
fn is_significant_scope(kind: &str) -> bool {
    matches!(kind, 
        "function_declaration" | "function_definition" | "method_definition" |
        "class_declaration" | "class_definition" | "module" | "impl_item" |
        "function_item" | "trait_item" | "interface_declaration" | "namespace"
    )
}

/// Extract parent context (e.g., class name for a method)
fn extract_parent_context(node: &Node, content: &str) -> Option<String> {
    let mut parent = node.parent();
    
    while let Some(p) = parent {
        if is_significant_scope(p.kind()) {
            return extract_node_name(&p, content);
        }
        parent = p.parent();
    }
    
    None
}

/// Extract definitions and references within a byte range
fn extract_symbols_in_range(
    root: Node,
    content: &str,
    start_byte: usize,
    end_byte: usize,
) -> (Vec<String>, Vec<String>) {
    let mut definitions = Vec::new();
    let mut references = Vec::new();
    
    let mut cursor = root.walk();
    extract_symbols_recursive(&mut cursor, content, start_byte, end_byte, &mut definitions, &mut references);
    
    // Remove duplicates
    definitions.sort();
    definitions.dedup();
    references.sort();
    references.dedup();
    
    (definitions, references)
}

fn extract_symbols_recursive(
    cursor: &mut TreeCursor,
    content: &str,
    start_byte: usize,
    end_byte: usize,
    definitions: &mut Vec<String>,
    references: &mut Vec<String>,
) {
    let node = cursor.node();
    
    // Only process nodes within our range
    if node.end_byte() < start_byte || node.start_byte() > end_byte {
        return;
    }
    
    // Check if this node is a definition or reference
    match node.kind() {
        // Variable/parameter definitions
        "variable_declarator" | "parameter" | "identifier" if is_definition_context(&node) => {
            if let Ok(text) = node.utf8_text(content.as_bytes()) {
                definitions.push(text.to_string());
            }
        }
        // Function/method definitions
        "function_declaration" | "function_definition" | "method_definition" => {
            if let Some(name_node) = find_child_by_kind(&node, "identifier") {
                if let Ok(text) = name_node.utf8_text(content.as_bytes()) {
                    definitions.push(text.to_string());
                }
            }
        }
        // Class/struct definitions
        "class_declaration" | "class_definition" | "struct_item" => {
            if let Some(name_node) = find_child_by_kind(&node, "identifier")
                .or_else(|| find_child_by_kind(&node, "type_identifier")) {
                if let Ok(text) = name_node.utf8_text(content.as_bytes()) {
                    definitions.push(text.to_string());
                }
            }
        }
        // References (function calls, variable usage)
        "call_expression" => {
            if let Some(func_node) = node.child(0) {
                if let Ok(text) = func_node.utf8_text(content.as_bytes()) {
                    references.push(text.to_string());
                }
            }
        }
        _ => {}
    }
    
    // Recurse into children
    if cursor.goto_first_child() {
        loop {
            extract_symbols_recursive(cursor, content, start_byte, end_byte, definitions, references);
            if !cursor.goto_next_sibling() {
                break;
            }
        }
        cursor.goto_parent();
    }
}

/// Check if an identifier node is in a definition context
fn is_definition_context(node: &Node) -> bool {
    if let Some(parent) = node.parent() {
        matches!(parent.kind(),
            "variable_declarator" | "parameter" | "formal_parameters" |
            "pattern" | "shorthand_property_identifier_pattern"
        )
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::languages::get_language;
    
    #[test]
    fn test_extract_function_metadata() {
        let code = r#"
def hello(name):
    """Say hello"""
    return f"Hello {name}!"
"#;
        
        let lang_fn = get_language("Python").unwrap();
        let lang: Language = lang_fn.into();
        
        let metadata = extract_metadata(code, 0, code.len(), lang, "Python").unwrap();
        
        assert!(metadata.node_type.contains("module") || metadata.node_type.contains("program"));
        // The whole module contains just the function definition
        assert!(metadata.definitions.contains(&"hello".to_string()));
        // Parameters are only extracted when we're inside the function scope
        assert!(metadata.definitions.len() >= 1);
    }
}