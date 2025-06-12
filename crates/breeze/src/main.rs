use std::path::Path;


#[tokio::main]
async fn main() {
    pretty_env_logger::init();

    let config = breeze::Config::default();
    let app = breeze::App::new(config).await.unwrap();
    app.index(&Path::new("/Users/ivan/github/kuzudb/kuzu")).await.unwrap();
    println!("Indexing completed successfully!");
    // let mut chunker = breeze_chunkers::walk_project("/Users/ivan/github/kuzudb/kuzu", WalkOptions {
    //     max_chunk_size: 2048,
    //     tokenizer: breeze_chunkers::Tokenizer::default(),
    //     max_parallel: 16,
    //     max_file_size: Some(1024 * 1024 * 5), // 5MB
    // });

    // let mut file_chunk_counts: HashMap<String, usize> = HashMap::new();
    // let mut total_chunks = 0;

    // while let Some(chunk) = chunker.next().await {
    //     match chunk {
    //         Ok(chunk) => {
    //             let file_path = chunk.file_path.clone();
    //             *file_chunk_counts.entry(file_path).or_insert(0) += 1;
    //             total_chunks += 1;
    //             if total_chunks % 1000 == 0 {
    //                 println!("Processed {} chunks so far...", total_chunks);
    //             }
    //             // println!("Chunk: {:?}", chunk);
    //         }
    //         Err(e) => {
    //             eprintln!("Error processing chunk: {}", e);
    //         }
    //     }
    // }

    // // Print summary statistics
    // println!("\n=== Summary ===");
    // println!("Total files processed: {}", file_chunk_counts.len());
    // println!("Total chunks: {}", total_chunks);
    // println!("\nChunks per file:");


}
