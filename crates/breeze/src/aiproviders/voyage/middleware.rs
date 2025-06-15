use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use governor::clock::{QuantaClock, QuantaInstant};
use governor::middleware::NoOpMiddleware;
use governor::state::{InMemoryState, NotKeyed};
use governor::{Quota, RateLimiter};
use http::Extensions;
use reqwest::{Request, Response};
use reqwest_middleware::{Error, Middleware, Next};
use tracing::debug;

use super::models::EmbeddingModel;
use super::types::Tier;

/// Token count information to pass via request extensions
#[derive(Debug, Clone)]
pub struct TokenCount(pub u32);

/// Rate limiter configuration
#[derive(Debug, Clone)]
pub struct RateLimiterConfig {
  pub tier: Tier,
  pub model: EmbeddingModel,
}

/// Rate limiter that tracks both requests and tokens
pub struct RateLimitMiddleware {
  /// Rate limiter for requests per minute
  request_limiter:
    Arc<RateLimiter<NotKeyed, InMemoryState, QuantaClock, NoOpMiddleware<QuantaInstant>>>,
  /// Rate limiter for tokens per minute
  token_limiter:
    Arc<RateLimiter<NotKeyed, InMemoryState, QuantaClock, NoOpMiddleware<QuantaInstant>>>,
}

impl RateLimitMiddleware {
  /// Create a new rate limit middleware for a given tier and model
  pub fn new(config: RateLimiterConfig) -> Self {
    // Create request limiter
    let requests_per_minute = config.tier.safe_requests_per_minute();
    let request_quota = Quota::per_minute(NonZeroU32::new(requests_per_minute).unwrap());
    let request_limiter = Arc::new(RateLimiter::direct(request_quota));

    // Create token limiter
    let tokens_per_minute = config.tier.safe_tokens_per_minute(config.model);
    let token_quota = Quota::per_minute(NonZeroU32::new(tokens_per_minute).unwrap());
    let token_limiter = Arc::new(RateLimiter::direct(token_quota));

    debug!(
      "Created rate limiter for tier {:?} with model {:?}: {} req/min, {} tokens/min",
      config.tier, config.model, requests_per_minute, tokens_per_minute
    );

    Self {
      request_limiter,
      token_limiter,
    }
  }

  /// Acquire tokens for a request
  /// Returns Ok(()) when tokens are acquired
  async fn acquire_tokens(&self, token_count: u32) -> Result<()> {
    // This implemenation is not exactly optimal, but it works well enough for now in combination with the safety margins.
    // We first wait for a request slot to ensure we don't exceed requests per minute

    // Wait for request slot using governor's built-in async waiting
    self
      .request_limiter
      .until_ready_with_jitter(governor::Jitter::up_to(Duration::from_millis(100)))
      .await;

    // Wait for token capacity
    let tokens_needed = NonZeroU32::new(token_count).unwrap_or(NonZeroU32::new(1).unwrap());
    debug!("Waiting for {} tokens...", token_count);

    // For multiple tokens, we need to check if we can acquire them
    loop {
      match self.token_limiter.check_n(tokens_needed) {
        Ok(_) => {
          debug!("Acquired {} tokens", token_count);
          return Ok(());
        }
        Err(_) => {
          // Wait for at least one token to be available
          self
            .token_limiter
            .until_ready_with_jitter(governor::Jitter::up_to(Duration::from_millis(100)))
            .await;
          // Continue the loop to try again
        }
      }
    }
  }
}

#[async_trait]
impl Middleware for RateLimitMiddleware {
  async fn handle(
    &self,
    req: Request,
    extensions: &mut Extensions,
    next: Next<'_>,
  ) -> Result<Response, Error> {
    // Extract token count from request extensions
    let token_count = if let Some(tc) = extensions.get::<TokenCount>() {
      tc.0
    } else {
      // Panic as this is a programmer error
      panic!("TokenCount must be provided in request extensions");
    };

    // Wait for rate limit clearance
    match self.acquire_tokens(token_count).await {
      Ok(_) => {
        debug!("Rate limit passed, proceeding with request");
        next.run(req, extensions).await
      }
      Err(e) => Err(Error::Middleware(e)),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::time::Instant;

  #[tokio::test]
  async fn test_rate_limiter_creation() {
    let config = RateLimiterConfig {
      tier: Tier::Tier2,
      model: EmbeddingModel::VoyageCode3,
    };
    let middleware = RateLimitMiddleware::new(config);

    // Should be able to make initial requests without delay
    assert!(middleware.acquire_tokens(1000).await.is_ok());
  }

  #[tokio::test]
  async fn test_rate_limit_enforcement() {
    let config = RateLimiterConfig {
      tier: Tier::Free,
      model: EmbeddingModel::VoyageCode3,
    };
    let middleware = RateLimitMiddleware::new(config);

    // Free tier: 3 requests/min, 120k tokens/min
    // Should allow 3 quick requests
    let _start = Instant::now();

    // First request should succeed
    assert!(middleware.acquire_tokens(40_000).await.is_ok());

    // Second request should succeed
    assert!(middleware.acquire_tokens(40_000).await.is_ok());

    // Third request should succeed (but exhausts most tokens)
    assert!(middleware.acquire_tokens(28_000).await.is_ok());

    // Fourth request should be rate limited (exceeds safe token limit)
    // This should take some time due to rate limiting
    let before_limited = Instant::now();
    assert!(middleware.acquire_tokens(10_000).await.is_ok());
    let after_limited = Instant::now();

    // Should have waited at least a bit (rate limiting in effect)
    assert!(after_limited.duration_since(before_limited) > Duration::from_millis(10));
  }
}
