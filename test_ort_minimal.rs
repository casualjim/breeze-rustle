fn main() {
  println!("Initializing ORT...");
  match ort::init().with_name("test").commit() {
    Ok(_) => println!("ORT initialized successfully"),
    Err(e) => eprintln!("Failed to initialize ORT: {}", e),
  }
  println!("Program ending...");
}
