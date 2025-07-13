mod app;
mod duration;
mod error;
mod mcp;
mod routes;
mod server;
pub mod types;

pub use server::{Config, run};

#[cfg(test)]
mod tests {
  use crate::types::CreateProjectRequest;
  use schemars::schema_for;

  #[test]
  fn test_schema_version() {
    let schema = schema_for!(CreateProjectRequest);
    let json = serde_json::to_value(&schema).unwrap();

    // Check what schema version is being generated
    if let Some(schema_version) = json.get("$schema") {
      println!("Schema version: {}", schema_version);
      assert!(
        schema_version.as_str().unwrap().contains("2020-12"),
        "Expected draft 2020-12, got: {}",
        schema_version
      );
    } else {
      panic!("No $schema field found in generated schema");
    }
  }
}
