use std::fmt;
use std::ops::Deref;
use std::str::FromStr;
use std::time::Duration;

use schemars::{JsonSchema, json_schema};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A wrapper around Duration that serializes/deserializes as human-readable strings
/// like "1h", "30m", "2h30m", etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HumanDuration(pub Duration);

impl HumanDuration {
  pub fn new(duration: Duration) -> Self {
    HumanDuration(duration)
  }

  pub fn inner(&self) -> Duration {
    self.0
  }
}

impl Deref for HumanDuration {
  type Target = Duration;

  fn deref(&self) -> &Self::Target {
    &self.0
  }
}

impl From<Duration> for HumanDuration {
  fn from(duration: Duration) -> Self {
    HumanDuration(duration)
  }
}

impl From<HumanDuration> for Duration {
  fn from(human: HumanDuration) -> Self {
    human.0
  }
}

impl fmt::Display for HumanDuration {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", humantime::format_duration(self.0))
  }
}

impl FromStr for HumanDuration {
  type Err = humantime::DurationError;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    humantime::parse_duration(s).map(HumanDuration)
  }
}

impl Serialize for HumanDuration {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: Serializer,
  {
    serializer.serialize_str(&self.to_string())
  }
}

impl<'de> Deserialize<'de> for HumanDuration {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: Deserializer<'de>,
  {
    let s = String::deserialize(deserializer)?;
    HumanDuration::from_str(&s).map_err(serde::de::Error::custom)
  }
}

impl JsonSchema for HumanDuration {
  fn schema_name() -> std::borrow::Cow<'static, str> {
    std::borrow::Cow::Borrowed("HumanDuration")
  }

  fn json_schema(_gen: &mut schemars::SchemaGenerator) -> schemars::Schema {
    json_schema!({
      "type": "string",
      "description": "Human-readable duration string (e.g. '1h', '30m', '1h 30m', '90s')",
      "examples": ["1h", "30m", "1h 30m", "90s", "2d 3h", "1w"]
    })
  }
}
