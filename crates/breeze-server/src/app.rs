use breeze_indexer::Indexer;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

#[derive(Clone)]
pub struct AppState {
  pub indexer: Arc<Indexer>,
  pub shutdown_token: CancellationToken,
}

impl AppState {
  pub async fn new(indexer: Indexer, shutdown_token: Option<CancellationToken>) -> Self {
    AppState {
      indexer: Arc::new(indexer),
      shutdown_token: shutdown_token.unwrap_or_default(),
    }
  }
}
