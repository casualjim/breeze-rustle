use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures::{Stream, StreamExt};
use http::header::{AUTHORIZATION, CONTENT_TYPE};
use http::{HeaderMap, HeaderValue};
use pin_project_lite::pin_project;
use reqwest::{Client, Response};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, OwnedSemaphorePermit, Semaphore};
use tokio::time::sleep;
use tracing::{debug, trace};

/// Configuration for API client
#[derive(Debug, Clone)]
pub struct ApiClientConfig {
  pub base_url: String,
  pub api_key: Option<String>,
  pub max_concurrent_requests: usize,
  pub max_requests_per_minute: usize,
  pub max_tokens_per_minute: usize,
  pub max_retries: usize,
  pub timeout: Duration,
}

impl Default for ApiClientConfig {
  fn default() -> Self {
    Self {
      base_url: String::new(),
      api_key: None,
      max_concurrent_requests: 300,
      max_requests_per_minute: 1000,
      max_tokens_per_minute: 1_000_000,
      max_retries: 3,
      timeout: Duration::from_secs(90),
    }
  }
}

/// Token bucket for smooth rate limiting
#[derive(Debug)]
struct TokenBucket {
  /// Maximum tokens in the bucket
  capacity: f64,
  /// Current tokens available
  tokens: f64,
  /// Rate at which tokens are refilled (tokens per second)
  refill_rate: f64,
  /// Last time tokens were refilled
  last_refill: Instant,
}

impl TokenBucket {
  fn new(capacity: f64, refill_rate: f64) -> Self {
    Self {
      capacity,
      tokens: capacity,
      refill_rate,
      last_refill: Instant::now(),
    }
  }

  /// Try to consume tokens, returns Ok(()) if successful, Err(wait_duration) if not
  fn try_consume(&mut self, tokens_needed: f64) -> Result<(), Duration> {
    self.refill();

    if self.tokens >= tokens_needed {
      self.tokens -= tokens_needed;
      Ok(())
    } else {
      // Calculate how long to wait for enough tokens
      let tokens_short = tokens_needed - self.tokens;
      let wait_seconds = tokens_short / self.refill_rate;
      Err(Duration::from_secs_f64(wait_seconds))
    }
  }

  /// Refill tokens based on elapsed time
  fn refill(&mut self) {
    let now = Instant::now();
    let elapsed = now.duration_since(self.last_refill).as_secs_f64();
    let new_tokens = elapsed * self.refill_rate;

    self.tokens = (self.tokens + new_tokens).min(self.capacity);
    self.last_refill = now;
  }
}

/// Rate limiter using dual token buckets for requests and tokens
#[derive(Debug)]
struct RateLimiter {
  request_bucket: TokenBucket,
  token_bucket: TokenBucket,
}

/// Simple API client wrapper around reqwest with rate limiting and retry
pub struct ApiClient {
  client: Client,
  config: ApiClientConfig,
  concurrent_semaphore: Arc<Semaphore>,
  rate_limiter: Arc<Mutex<RateLimiter>>,
}

impl ApiClient {
  pub fn new(config: ApiClientConfig) -> Result<Self, reqwest::Error> {
    let client = Client::builder()
      .default_headers({
        let mut headers = HeaderMap::new();
        if let Some(api_key) = &config.api_key {
          headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", api_key)).unwrap(),
          );
        }
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers
      })
      .user_agent("Breeze/ApiClient")
      .timeout(config.timeout)
      .build()?;

    let concurrent_semaphore = Arc::new(Semaphore::new(config.max_concurrent_requests));

    // Create token buckets with per-second rates
    let request_rate = config.max_requests_per_minute as f64 / 60.0;
    let token_rate = config.max_tokens_per_minute as f64 / 60.0;

    // Start with smaller initial capacity for smoother rate limiting
    // Allow burst of up to 10 seconds worth of requests
    let request_capacity = (request_rate * 10.0).min(config.max_requests_per_minute as f64);
    let token_capacity = (token_rate * 10.0).min(config.max_tokens_per_minute as f64);

    let rate_limiter = Arc::new(Mutex::new(RateLimiter {
      request_bucket: TokenBucket::new(request_capacity, request_rate),
      token_bucket: TokenBucket::new(token_capacity, token_rate),
    }));

    Ok(Self {
      client,
      config,
      concurrent_semaphore,
      rate_limiter,
    })
  }

  /// Make a POST request with JSON payload
  pub async fn post_json<Req, Res>(
    &self,
    path: &str,
    payload: &Req,
    token_count: u32,
  ) -> Result<Res, Box<dyn std::error::Error + Send + Sync>>
  where
    Req: Serialize,
    Res: for<'de> Deserialize<'de>,
  {
    let url = format!("{}{}", self.config.base_url, path);
    let body_bytes = serde_json::to_vec(payload)?;

    let mut retries = 0;
    loop {
      debug!("Attempting request to {} (attempt {})", url, retries + 1);

      // Wait for rate limits
      self.wait_for_rate_limit(token_count).await?;

      // Acquire concurrent request permit
      let request_permit = self.concurrent_semaphore.clone().acquire_owned().await?;

      // Make request
      let mut request = self
        .client
        .post(&url)
        .header("Content-Type", "application/json")
        .body(body_bytes.clone());

      if let Some(api_key) = &self.config.api_key {
        request = request.header("Authorization", format!("Bearer {}", api_key));
      }

      let result = request.send().await;

      match result {
        Ok(response) => {
          let status = response.status();

          if status.is_success() {
            // Wrap the response body to hold permit until consumed
            let body_bytes = read_body_with_permit(response, request_permit).await?;
            let result = serde_json::from_slice(&body_bytes)?;
            debug!("Request to {} succeeded with status {}", url, status);
            return Ok(result);
          } else if should_retry(status.as_u16()) && retries < self.config.max_retries {
            // Release permit for retry
            drop(request_permit);

            retries += 1;
            let backoff = calculate_backoff(retries);
            debug!(
              "Retrying after {} seconds due to status {}",
              backoff.as_secs(),
              status
            );
            sleep(backoff).await;
            continue;
          } else {
            let error_text = response.text().await.unwrap_or_default();
            return Err(
              format!("API request failed with status {}: {}", status, error_text).into(),
            );
          }
        }
        Err(e) => {
          // Release permit
          drop(request_permit);

          if is_retryable_error(&e) && retries < self.config.max_retries {
            retries += 1;
            let backoff = calculate_backoff(retries);
            debug!(
              "Retrying after {} seconds due to error: {}",
              backoff.as_secs(),
              e
            );
            sleep(backoff).await;
            continue;
          } else {
            return Err(e.into());
          }
        }
      }
    }
  }

  /// Wait until we can make a request without exceeding rate limits
  async fn wait_for_rate_limit(
    &self,
    token_count: u32,
  ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    loop {
      let wait_duration = {
        let mut limiter = self.rate_limiter.lock().await;

        // Check if we can consume from both buckets
        let request_result = limiter.request_bucket.try_consume(1.0);
        let token_result = limiter.token_bucket.try_consume(token_count as f64);

        match (request_result, token_result) {
          (Ok(()), Ok(())) => {
            // Both succeeded, we're good to go
            return Ok(());
          }
          (Ok(()), Err(token_wait)) => {
            // Request succeeded but tokens failed, refund request
            limiter.request_bucket.tokens += 1.0;
            token_wait
          }
          (Err(request_wait), Ok(())) => {
            // Tokens succeeded but request failed, refund tokens
            limiter.token_bucket.tokens += token_count as f64;
            request_wait
          }
          (Err(request_wait), Err(token_wait)) => {
            // Both failed, return the maximum wait time
            request_wait.max(token_wait)
          }
        }
      };

      // Add a small buffer to avoid tight loops
      let wait_with_buffer = wait_duration + Duration::from_millis(10);

      if wait_with_buffer > Duration::from_millis(100) {
        debug!(
          "Rate limit: waiting {:?} before next request",
          wait_with_buffer
        );
      }

      sleep(wait_with_buffer).await;
    }
  }
}

/// Read response body while holding permit
async fn read_body_with_permit(
  response: Response,
  request_permit: OwnedSemaphorePermit,
) -> Result<Bytes, Box<dyn std::error::Error + Send + Sync>> {
  // Create a stream that holds the permit
  let stream = response.bytes_stream();
  let guarded_stream = GuardedStream::new(stream, request_permit);

  // Collect all bytes
  let chunks: Vec<_> = guarded_stream.collect::<Vec<_>>().await;

  // Combine chunks into single Bytes
  let mut combined = Vec::new();
  for chunk in chunks {
    combined.extend_from_slice(&chunk?);
  }

  Ok(Bytes::from(combined))
}

pin_project! {
    /// Stream wrapper that holds permit until the stream is fully consumed
    struct GuardedStream<S> {
        #[pin]
        inner: S,
        _request_permit: Option<OwnedSemaphorePermit>,
    }
}

impl<S> GuardedStream<S> {
  fn new(inner: S, request_permit: OwnedSemaphorePermit) -> Self {
    Self {
      inner,
      _request_permit: Some(request_permit),
    }
  }
}

impl<S, E> Stream for GuardedStream<S>
where
  S: Stream<Item = Result<Bytes, E>>,
{
  type Item = Result<Bytes, E>;

  fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
    let this = self.project();

    match this.inner.poll_next(cx) {
      Poll::Ready(None) => {
        // Stream is done, drop permit
        trace!("Response body fully consumed, releasing permit");
        *this._request_permit = None;
        Poll::Ready(None)
      }
      other => other,
    }
  }
}

/// Check if status code indicates we should retry
fn should_retry(status: u16) -> bool {
  matches!(status, 429 | 500 | 502 | 503 | 504)
}

/// Check if error is retryable
fn is_retryable_error(error: &reqwest::Error) -> bool {
  error.is_timeout() || error.is_connect()
}

/// Calculate exponential backoff duration
fn calculate_backoff(retry_count: usize) -> Duration {
  // Allow tests to use milliseconds for faster execution
  #[cfg(test)]
  {
    let base = 2u64;
    let millis = base.pow(retry_count as u32).min(60) * 10; // 10ms base instead of 1s
    Duration::from_millis(millis)
  }

  #[cfg(not(test))]
  {
    let base = 2u64;
    let seconds = base.pow(retry_count as u32).min(60);
    Duration::from_secs(seconds)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use wiremock::matchers::{header, method, path};
  use wiremock::{Mock, MockServer, ResponseTemplate};

  #[derive(Serialize)]
  struct TestRequest {
    message: String,
  }

  #[derive(Deserialize, PartialEq, Debug)]
  struct TestResponse {
    result: String,
  }

  #[tokio::test]
  async fn test_successful_request() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
      .and(path("/test"))
      .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
          "result": "success"
      })))
      .mount(&mock_server)
      .await;

    let config = ApiClientConfig {
      base_url: mock_server.uri(),
      max_requests_per_minute: 100,
      max_tokens_per_minute: 10000,
      ..Default::default()
    };

    let client = ApiClient::new(config).unwrap();
    let request = TestRequest {
      message: "test".to_string(),
    };

    let response: TestResponse = client.post_json("/test", &request, 10).await.unwrap();
    assert_eq!(response.result, "success");
  }

  #[tokio::test]
  async fn test_retry_on_server_error() {
    let mock_server = MockServer::start().await;

    let counter = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
    let c = counter.clone();

    Mock::given(method("POST"))
      .and(path("/test"))
      .respond_with(move |_: &wiremock::Request| {
        let count = c.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if count == 0 {
          ResponseTemplate::new(503)
        } else {
          ResponseTemplate::new(200).set_body_json(serde_json::json!({
              "result": "retry_success"
          }))
        }
      })
      .mount(&mock_server)
      .await;

    let config = ApiClientConfig {
      base_url: mock_server.uri(),
      max_retries: 3,
      max_requests_per_minute: 100,
      max_tokens_per_minute: 10000,
      ..Default::default()
    };

    let client = ApiClient::new(config).unwrap();
    let request = TestRequest {
      message: "test".to_string(),
    };

    let start = Instant::now();
    let response: TestResponse = client.post_json("/test", &request, 10).await.unwrap();
    let elapsed = start.elapsed();

    assert_eq!(response.result, "retry_success");
    assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 2);
    // Should have waited at least 20ms due to backoff (2^1 * 10ms)
    assert!(elapsed >= Duration::from_millis(20));
  }

  #[tokio::test]
  async fn test_rate_limiting() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
      .and(path("/test"))
      .respond_with(
        ResponseTemplate::new(200)
          .set_body_json(serde_json::json!({"result": "rate_limited"}))
          .set_delay(Duration::from_millis(100)),
      )
      .mount(&mock_server)
      .await;

    let config = ApiClientConfig {
      base_url: mock_server.uri(),
      max_concurrent_requests: 1,
      max_requests_per_minute: 100,
      max_tokens_per_minute: 10000,
      ..Default::default()
    };

    let client = ApiClient::new(config).unwrap();
    let request = TestRequest {
      message: "test".to_string(),
    };

    let start = Instant::now();

    // Make two concurrent requests - second should wait for first
    let client_ref = &client;
    let request_ref = &request;
    let (result1, result2) = tokio::join!(
      client_ref.post_json::<_, TestResponse>("/test", request_ref, 10),
      client_ref.post_json::<_, TestResponse>("/test", request_ref, 10)
    );

    let elapsed = start.elapsed();

    assert!(result1.is_ok());
    assert!(result2.is_ok());

    // Should take at least 200ms (two sequential 100ms requests)
    assert!(elapsed >= Duration::from_millis(200));
  }

  #[tokio::test]
  async fn test_api_key_header() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
      .and(path("/test"))
      .and(header("Authorization", "Bearer test_key"))
      .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
          "result": "authorized"
      })))
      .mount(&mock_server)
      .await;

    let config = ApiClientConfig {
      base_url: mock_server.uri(),
      api_key: Some("test_key".to_string()),
      max_requests_per_minute: 100,
      max_tokens_per_minute: 10000,
      ..Default::default()
    };

    let client = ApiClient::new(config).unwrap();
    let request = TestRequest {
      message: "test".to_string(),
    };

    let response: TestResponse = client.post_json("/test", &request, 10).await.unwrap();
    assert_eq!(response.result, "authorized");
  }

  #[tokio::test]
  async fn test_token_bucket_rate_limiting() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
      .and(path("/test"))
      .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
          "result": "token_bucket"
      })))
      .mount(&mock_server)
      .await;

    // High rate to make tests fast but still demonstrate rate limiting
    let config = ApiClientConfig {
      base_url: mock_server.uri(),
      max_concurrent_requests: 10,
      max_requests_per_minute: 600, // 10 per second
      max_tokens_per_minute: 6000,  // 100 tokens per second
      ..Default::default()
    };

    let client = ApiClient::new(config).unwrap();
    let request = TestRequest {
      message: "test".to_string(),
    };

    // The bucket starts with capacity for ~100 requests (10 seconds * 10/sec)
    // Let's consume enough to see rate limiting

    // Make rapid requests to consume the burst capacity
    let start = Instant::now();
    let mut request_count = 0;

    // Keep making requests until we hit rate limiting
    while start.elapsed() < Duration::from_millis(200) {
      match client
        .post_json::<_, TestResponse>("/test", &request, 10)
        .await
      {
        Ok(response) => {
          assert_eq!(response.result, "token_bucket");
          request_count += 1;
        }
        Err(_) => break, // Hit some limit
      }
    }

    // We should have made multiple requests quickly
    assert!(
      request_count >= 2,
      "Expected at least 2 requests, got {}",
      request_count
    );

    // Now verify rate limiting is working by timing subsequent requests
    let before_limited = Instant::now();
    let response: TestResponse = client.post_json("/test", &request, 10).await.unwrap();
    let response2: TestResponse = client.post_json("/test", &request, 10).await.unwrap();
    let limited_elapsed = before_limited.elapsed();

    assert_eq!(response.result, "token_bucket");
    assert_eq!(response2.result, "token_bucket");

    // With 10 requests per second, spacing should be ~100ms between requests
    // We made 2 requests, so should take at least 100ms total
    assert!(
      limited_elapsed >= Duration::from_millis(90),
      "Expected rate limiting spacing, but got {:?}",
      limited_elapsed
    );
  }
}
