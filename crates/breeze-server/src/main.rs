use breeze_server::Config;
use tracing::error;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  rustls::crypto::aws_lc_rs::default_provider().install_default()
    .map_err(|_| anyhow::anyhow!("Failed to install crypto provider"))?;
  setup_panic_hook();

  let config = Config {
    http_port: 8080,
    https_port: 8443,
    ..Default::default()
  };
  breeze_server::run(config).await?;
  tracing::info!("Breeze server started successfully");
  Ok(())
}

fn setup_panic_hook() {
  std::panic::set_hook(Box::new(|panic| {
    if let Some(location) = panic.location() {
      error!(
        message = %panic,
        panic.file = location.file(),
        panic.line = location.line(),
        panic.column = location.column(),
      );
    } else {
      error!(message = %panic);
    }
    futures::executor::block_on(async {
      opentelemetry::global::tracer_provider();
    });
  }));
}
