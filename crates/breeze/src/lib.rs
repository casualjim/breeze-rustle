mod app;
pub mod cli;
pub mod config;
mod logging;

pub use app::App;
pub use config::Config;
pub use logging::init as init_logging;

// Re-export from breeze-indexer
pub use breeze_indexer::{SearchResult, ensure_ort_initialized, hybrid_search};
