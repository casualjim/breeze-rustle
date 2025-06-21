use breeze_indexer::Indexer;
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
  pub indexer: Arc<Indexer>,
}

impl AppState {
  pub async fn new(indexer: Arc<Indexer>) -> Self {
    AppState {
      indexer,
    }
  }
}
