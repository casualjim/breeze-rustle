use tracing::Subscriber;
use tracing_subscriber::{
  EnvFilter, Layer, layer::SubscriberExt, registry::LookupSpan, util::SubscriberInitExt,
};

pub fn init(app_name: &str) -> anyhow::Result<()> {
  tracing_subscriber::registry()
    .with(build_loglevel_filter_layer(format!(
      "info,{app_name}=debug"
    )))
    .with(build_logger_text())
    .init();
  Ok(())
}

pub fn build_logger_text<S>() -> Box<dyn Layer<S> + Send + Sync + 'static>
where
  S: Subscriber + for<'a> LookupSpan<'a>,
{
  // if cfg!(debug_assertions) {
  Box::new(
    tracing_subscriber::fmt::layer()
      .pretty()
      .with_timer(tracing_subscriber::fmt::time::time()),
  )
  // } else {
  //   Box::new(tracing_subscriber::fmt::layer().compact().without_time())
  // }
}

pub fn build_loglevel_filter_layer<S: Into<String>>(default_log: S) -> EnvFilter {
  // filter what is output on log (fmt)
  let lg = std::env::var("RUST_LOG").unwrap_or_else(|_| default_log.into());
  unsafe {
    std::env::set_var(
      "RUST_LOG",
      format!(
        // `otel::tracing` should be a level trace to emit opentelemetry trace & span
        // `otel::setup` set to debug to log detected resources, configuration read and infered
        "{lg},otel::tracing=trace,otel::setup=debug",
      ),
    );
  }
  EnvFilter::from_default_env()
}
