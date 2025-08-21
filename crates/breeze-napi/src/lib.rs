pub mod chunking;
pub mod indexer;

// Re-export N-API surface from both modules
pub use chunking::*;
pub use indexer::*;
