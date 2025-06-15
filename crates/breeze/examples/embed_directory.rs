use breeze_chunkers::{Chunk, Tokenizer, WalkOptions, walk_project};
use embed_anything::embeddings::embed::EmbedderBuilder;
use embed_anything::embeddings::local::text_embedding::ONNXModel;
use futures::StreamExt;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use tracing::{debug, error, info};

// Structure to hold chunk data for embedding
#[derive(Clone)]
struct ChunkData {
  text: String,
}

// Shared statistics
#[derive(Clone)]
struct Stats {
  files: Arc<AtomicUsize>,
  chunks: Arc<AtomicUsize>,
  embeddings: Arc<AtomicUsize>,
  tokens: Arc<AtomicUsize>,
}

impl Stats {
  fn new() -> Self {
    Self {
      files: Arc::new(AtomicUsize::new(0)),
      chunks: Arc::new(AtomicUsize::new(0)),
      embeddings: Arc::new(AtomicUsize::new(0)),
      tokens: Arc::new(AtomicUsize::new(0)),
    }
  }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
  // Run the async work in a runtime that we control
  let runtime = tokio::runtime::Runtime::new()?;
  let result = runtime.block_on(run_embedding());

  // Force clean exit to avoid ONNX cleanup issues
  match result {
    Ok(_) => std::process::exit(0),
    Err(_) => std::process::exit(1),
  }
}

async fn run_embedding() -> Result<(), Box<dyn std::error::Error>> {
  // Initialize logging
  tracing_subscriber::fmt::init();

  // Get directory from command line args
  let args: Vec<String> = std::env::args().collect();
  if args.len() < 2 {
    eprintln!("Usage: {} <directory>", args[0]);
    std::process::exit(1);
  }
  let directory = PathBuf::from(&args[1]);

  println!(
    "Benchmarking embeddings for directory: {}",
    directory.display()
  );
  let start_time = Instant::now();

  // Don't create embedder here - each worker will create its own

  // Configure walker
  let walk_options = WalkOptions {
    max_chunk_size: 512,
    tokenizer: Tokenizer::Characters,
    max_parallel: 4,                      // Parallel file processing
    max_file_size: Some(5 * 1024 * 1024), // 5MB limit
  };

  // Create flume channel for SPMC (single producer, multiple consumer)
  let (chunk_tx, chunk_rx) = flume::bounded::<Vec<ChunkData>>(100);

  // Shared statistics
  let stats = Stats::new();

  // Spawn stream processor task
  let stream_handle = spawn_stream_processor(
    directory.clone(),
    walk_options,
    chunk_tx,
    stats.clone(),
    512,
  );

  let stats_clone = stats.clone();
  let rx_clone = chunk_rx.clone();

  // Spawn embedder in a separate OS thread to avoid async issues
  let handle = std::thread::spawn(move || {
    // Create embedder in the thread
    let embedder = match EmbedderBuilder::new()
      .model_architecture("bert")
      .onnx_model_id(Some(ONNXModel::AllMiniLML6V2))
      .from_pretrained_onnx()
    {
      Ok(e) => e,
      Err(e) => {
        panic!("Failed to create embedder: {}", e);
      }
    };

    // Run the embedding worker with sync operations
    embedding_worker_sync(1, embedder, rx_clone, stats_clone);
  });

  // Wait for stream processor to complete
  stream_handle.await?;

  // Wait for embedding thread to complete
  handle.join().expect("Embedding thread panicked");

  // Give ONNX time to clean up
  tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

  let elapsed = start_time.elapsed();

  // Print summary statistics
  let total_files = stats.files.load(Ordering::Relaxed);
  let total_chunks = stats.chunks.load(Ordering::Relaxed);
  let total_embeddings = stats.embeddings.load(Ordering::Relaxed);
  let total_tokens = stats.tokens.load(Ordering::Relaxed);

  println!("\n=== Embedding Benchmark Summary ===");
  println!("Directory: {}", directory.display());
  println!("Total files processed: {}", total_files);
  println!("Total chunks generated: {}", total_chunks);
  println!("Total embeddings created: {}", total_embeddings);
  println!("Total tokens processed: {}", total_tokens);
  println!("Time elapsed: {:.2}s", elapsed.as_secs_f64());
  println!("Workers used: {}", 1);

  if total_chunks > 0 {
    println!(
      "Chunks per second: {:.0}",
      total_chunks as f64 / elapsed.as_secs_f64()
    );
    println!(
      "Average chunks per file: {:.1}",
      total_chunks as f64 / total_files as f64
    );
  }

  if total_tokens > 0 {
    println!(
      "Tokens per second: {:.0}",
      total_tokens as f64 / elapsed.as_secs_f64()
    );
  }

  Ok(())
}

fn spawn_stream_processor(
  directory: PathBuf,
  walk_options: WalkOptions,
  chunk_tx: flume::Sender<Vec<ChunkData>>,
  stats: Stats,
  batch_size: usize,
) -> tokio::task::JoinHandle<()> {
  tokio::spawn(async move {
    info!("Stream processor started");

    // Get chunk stream from walk_project
    let chunk_stream = walk_project(directory, walk_options);
    let mut chunk_stream = Box::pin(chunk_stream);

    let mut batch_buffer = Vec::new();

    while let Some(result) = chunk_stream.next().await {
      match result {
        Ok(project_chunk) => {
          match project_chunk.chunk {
            Chunk::Semantic(sc) | Chunk::Text(sc) => {
              // Accumulate chunk data
              batch_buffer.push(ChunkData { text: sc.text });

              stats.chunks.fetch_add(1, Ordering::Relaxed);

              // Send batch when it reaches the batch size
              if batch_buffer.len() >= batch_size {
                debug!("Sending batch of {} chunks", batch_buffer.len());
                if chunk_tx.send(batch_buffer.clone()).is_err() {
                  error!("Failed to send batch - receivers dropped");
                  return;
                }
                batch_buffer.clear();
              }
            }
            Chunk::EndOfFile { .. } => {
              stats.files.fetch_add(1, Ordering::Relaxed);
            }
          }
        }
        Err(e) => {
          error!("Error processing chunk: {}", e);
        }
      }
    }

    // Send remaining chunks
    if !batch_buffer.is_empty() {
      debug!("Sending final batch of {} chunks", batch_buffer.len());
      let _ = chunk_tx.send(batch_buffer);
    }

    info!(
      "Stream processor completed. Files: {}, Chunks: {}",
      stats.files.load(Ordering::Relaxed),
      stats.chunks.load(Ordering::Relaxed)
    );
  })
}

fn embedding_worker_sync(
  worker_id: usize,
  embedder: embed_anything::embeddings::embed::Embedder,
  chunk_rx: flume::Receiver<Vec<ChunkData>>,
  stats: Stats,
) {
  info!("Worker {} started", worker_id);
  let mut batches_processed = 0;
  let mut first_batch = true;

  // Buffer to accumulate exactly 256 chunks
  let mut chunk_buffer: Vec<ChunkData> = Vec::new();

  loop {
    // Try to receive a batch (sync)
    match chunk_rx.recv() {
      Ok(chunk_batch) => {
        // Add chunks to our buffer
        for chunk in chunk_batch {
          chunk_buffer.push(chunk);

          // Process when we have exactly 256 chunks
          if chunk_buffer.len() == 256 {
            process_chunk_batch_sync(
              &embedder,
              &chunk_buffer,
              worker_id,
              &mut batches_processed,
              &stats,
              &mut first_batch,
            );
            chunk_buffer.clear();
          }
        }
      }
      Err(_) => {
        // Channel closed, process remaining chunks
        if !chunk_buffer.is_empty() {
          process_chunk_batch_sync(
            &embedder,
            &chunk_buffer,
            worker_id,
            &mut batches_processed,
            &stats,
            &mut first_batch,
          );
        }
        break;
      }
    }
  }
  // Embedder will be dropped when function exits
  info!(
    "Worker {} completed. Processed {} batches",
    worker_id, batches_processed
  );
}

fn process_chunk_batch_sync(
  embedder: &embed_anything::embeddings::embed::Embedder,
  chunks: &[ChunkData],
  worker_id: usize,
  batches_processed: &mut usize,
  stats: &Stats,
  first_batch: &mut bool,
) {
  *batches_processed += 1;
  debug!(
    "Worker {} processing batch {} with {} chunks",
    worker_id,
    batches_processed,
    chunks.len()
  );

  // Extract texts for embedding
  let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();

  // Create a small runtime just for this async call
  let rt = tokio::runtime::Runtime::new().unwrap();

  // Generate embeddings
  match rt.block_on(embedder.embed(&texts, None, None)) {
    Ok(embeddings) => {
      stats
        .embeddings
        .fetch_add(embeddings.len(), Ordering::Relaxed);

      // Count approximate tokens
      for chunk in chunks {
        stats
          .tokens
          .fetch_add(chunk.text.len() / 4, Ordering::Relaxed);
      }

      // Print dimension info only for first batch of first worker
      if *first_batch && worker_id == 0 {
        if let Some(first_embedding) = embeddings.first() {
          match first_embedding {
            embed_anything::embeddings::embed::EmbeddingResult::DenseVector(dense) => {
              info!("Embedding dimension: {}", dense.len());
            }
            embed_anything::embeddings::embed::EmbeddingResult::MultiVector(multi) => {
              info!("Got multi-vector embedding with {} vectors", multi.len());
            }
          }
        }
        *first_batch = false;
      }

      debug!(
        "Worker {} successfully embedded {} chunks",
        worker_id,
        embeddings.len()
      );
    }
    Err(e) => {
      error!("Worker {} failed to embed batch: {}", worker_id, e);
    }
  }
}
