use breeze_indexer::Indexer;
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
  pub indexer: Arc<Indexer>,
}

impl AppState {
  pub async fn new(indexer: Indexer) -> Self {
    AppState {
      indexer: Arc::new(indexer),
    }
  }
}
