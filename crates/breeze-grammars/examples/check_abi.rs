fn main() {
    println!("Tree-sitter expected ABI version: {}", tree_sitter::LANGUAGE_VERSION);
    println!("Tree-sitter MIN ABI version: {}", tree_sitter::MIN_COMPATIBLE_LANGUAGE_VERSION);
}