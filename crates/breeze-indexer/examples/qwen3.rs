// use anyhow::Result;
// use ndarray::{Array1, Array2, Array3, Axis, s};
// use ort::{Environment, GraphOptimizationLevel, Session, SessionBuilder, Value};
// use tokenizers::Tokenizer;

// pub struct Qwen3Embedder {
//   session: Session,
//   tokenizer: Tokenizer,
//   eod_token_id: u32,
//   max_length: usize,
// }

// impl Qwen3Embedder {
//   pub fn new(model_path: &str, tokenizer_path: &str) -> Result<Self> {
//     let env = Environment::builder().with_name("qwen3_embedder").build()?;

//     let session = SessionBuilder::new(&env)?
//       .with_optimization_level(GraphOptimizationLevel::Level3)?
//       .with_intra_threads(0)? // Use all cores
//       .with_memory_pattern_optimization(true)?
//       .with_model_from_file(model_path)?;

//     let tokenizer = Tokenizer::from_file(tokenizer_path)?;
//     let eod_token_id = tokenizer
//       .token_to_id("<|endoftext|>")
//       .ok_or_else(|| anyhow::anyhow!("EOD token not found"))?;

//     Ok(Self {
//       session,
//       tokenizer,
//       eod_token_id,
//       max_length: 512, // Adjust based on your needs
//     })
//   }

//   pub fn embed_batch(&self, texts: &[String]) -> Result<Array2<f32>> {
//     // Tokenize with special handling
//     let mut all_input_ids = Vec::new();
//     let mut all_attention_masks = Vec::new();

//     for text in texts {
//       let encoding = self.tokenizer.encode(text, false)?;
//       let mut ids = encoding.get_ids().to_vec();
//       let mut mask = encoding.get_attention_mask().to_vec();

//       // Critical: Append EOD token
//       ids.push(self.eod_token_id);
//       mask.push(1);

//       // Pad or truncate
//       if ids.len() > self.max_length {
//         ids.truncate(self.max_length);
//         mask.truncate(self.max_length);
//       } else {
//         // Left padding for Qwen3
//         let pad_length = self.max_length - ids.len();
//         ids = vec![0; pad_length].into_iter().chain(ids).collect();
//         mask = vec![0; pad_length].into_iter().chain(mask).collect();
//       }

//       all_input_ids.extend(ids.iter().map(|&x| x as i64));
//       all_attention_masks.extend(mask.iter().map(|&x| x as i64));
//     }

//     let batch_size = texts.len();

//     // Create tensors
//     let input_ids = Array2::from_shape_vec((batch_size, self.max_length), all_input_ids)?;

//     let attention_mask =
//       Array2::from_shape_vec((batch_size, self.max_length), all_attention_masks)?;

//     // Run inference
//     let outputs = self.session.run(vec![
//       Value::from_array(self.session.allocator(), &input_ids)?,
//       Value::from_array(self.session.allocator(), &attention_mask)?,
//     ])?;

//     // Extract last hidden states
//     let last_hidden_states = outputs[0].try_extract::<f32>()?.view().to_owned();
//     let last_hidden_states = last_hidden_states.into_dimensionality::<ndarray::Ix3>()?;

//     // Apply last token pooling
//     let mut embeddings = self.last_token_pool(&last_hidden_states, &attention_mask);

//     // L2 normalize
//     self.l2_normalize(&mut embeddings);

//     Ok(embeddings)
//   }

//   fn last_token_pool(
//     &self,
//     hidden_states: &Array3<f32>,
//     attention_mask: &Array2<i64>,
//   ) -> Array2<f32> {
//     let batch_size = hidden_states.shape()[0];
//     let hidden_dim = hidden_states.shape()[2];
//     let mut pooled = Array2::zeros((batch_size, hidden_dim));

//     // With left padding, last token is always at position -1
//     pooled.assign(&hidden_states.slice(s![.., -1, ..]));

//     pooled
//   }

//   fn l2_normalize(&self, embeddings: &mut Array2<f32>) {
//     for mut row in embeddings.axis_iter_mut(Axis(0)) {
//       let norm = row.mapv(|x| x * x).sum().sqrt();
//       if norm > 0.0 {
//         row /= norm;
//       }
//     }
//   }
// }

// // Usage example
// fn main() -> Result<()> {
//   let embedder = Qwen3Embedder::new("path/to/qwen3-0.6b.onnx", "path/to/tokenizer.json")?;

//   let texts = vec![
//     "Hello world".to_string(),
//     "Rust ONNX Runtime example".to_string(),
//   ];

//   let embeddings = embedder.embed_batch(&texts)?;
//   println!("Embeddings shape: {:?}", embeddings.shape());

//   Ok(())
// }

fn main() {
  // This is a placeholder for the main function.
  // The actual implementation would involve initializing the Qwen3Embedder,
  // loading the model, and processing input texts to generate embeddings.
  println!("Qwen3 Embedder example");
}
