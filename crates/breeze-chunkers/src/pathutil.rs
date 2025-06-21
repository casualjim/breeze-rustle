// This file is derived from the ripgrep project (https://github.com/BurntSushi/ripgrep),
// specifically from the ignore crate.
// Original source: https://github.com/BurntSushi/ripgrep/tree/master/crates/ignore
// License: MIT OR Unlicense
use std::{ffi::OsStr, path::Path};

/// Returns true if and only if this entry is considered to be hidden.
///
/// This only returns true if the base name of the path starts with a `.`.
///
/// On Unix, this implements a more optimized check.
#[cfg(unix)]
pub(crate) fn is_hidden<P: AsRef<Path>>(dent: P) -> bool {
  use std::os::unix::ffi::OsStrExt;

  if let Some(name) = file_name(&dent) {
    name.as_bytes().get(0) == Some(&b'.')
  } else {
    false
  }
}

/// Returns true if and only if this entry is considered to be hidden.
///
/// On Windows, this returns true if one of the following is true:
///
/// * The base name of the path starts with a `.`.
/// * The file attributes have the `HIDDEN` property set.
#[cfg(windows)]
pub(crate) fn is_hidden<P: AsRef<Path>>(dent: P) -> bool {
  use std::os::windows::fs::MetadataExt;
  use winapi_util::file;

  // This looks like we're doing an extra stat call, but on Windows, the
  // directory traverser reuses the metadata retrieved from each directory
  // entry and stores it on the DirEntry itself. So this is "free."
  if let Ok(md) = std::fs::metadata(dent) {
    if file::is_hidden(md.file_attributes() as u64) {
      return true;
    }
  }
  if let Some(name) = file_name(dent) {
    name.to_str().map(|s| s.starts_with(".")).unwrap_or(false)
  } else {
    false
  }
}

/// Returns true if and only if this entry is considered to be hidden.
///
/// This only returns true if the base name of the path starts with a `.`.
#[cfg(not(any(unix, windows)))]
pub(crate) fn is_hidden<P: AsRef<Path>>(dent: P) -> bool {
  if let Some(name) = file_name(dent) {
    name.to_str().map(|s| s.starts_with(".")).unwrap_or(false)
  } else {
    false
  }
}

/// Strip `prefix` from the `path` and return the remainder.
///
/// If `path` doesn't have a prefix `prefix`, then return `None`.
#[cfg(unix)]
pub(crate) fn strip_prefix<'a, P: AsRef<Path> + ?Sized>(
  prefix: &'a P,
  path: &'a Path,
) -> Option<&'a Path> {
  use std::os::unix::ffi::OsStrExt;

  let prefix = prefix.as_ref().as_os_str().as_bytes();
  let path = path.as_os_str().as_bytes();
  if prefix.len() > path.len() || prefix != &path[0..prefix.len()] {
    None
  } else {
    Some(&Path::new(OsStr::from_bytes(&path[prefix.len()..])))
  }
}

/// Strip `prefix` from the `path` and return the remainder.
///
/// If `path` doesn't have a prefix `prefix`, then return `None`.
#[cfg(not(unix))]
pub(crate) fn strip_prefix<'a, P: AsRef<Path> + ?Sized>(
  prefix: &'a P,
  path: &'a Path,
) -> Option<&'a Path> {
  path.strip_prefix(prefix).ok()
}

/// The final component of the path, if it is a normal file.
///
/// If the path terminates in ., .., or consists solely of a root of prefix,
/// file_name will return None.
#[cfg(unix)]
pub(crate) fn file_name<'a, P: AsRef<Path> + ?Sized>(path: &'a P) -> Option<&'a OsStr> {
  use memchr::memrchr;
  use std::os::unix::ffi::OsStrExt;

  let path = path.as_ref().as_os_str().as_bytes();
  if path.is_empty() {
    return None;
  } else if path.len() == 1 && path[0] == b'.' {
    return None;
  } else if path.last() == Some(&b'.') {
    return None;
  } else if path.len() >= 2 && &path[path.len() - 2..] == &b".."[..] {
    return None;
  }
  let last_slash = memrchr(b'/', path).map(|i| i + 1).unwrap_or(0);
  Some(OsStr::from_bytes(&path[last_slash..]))
}

/// The final component of the path, if it is a normal file.
///
/// If the path terminates in ., .., or consists solely of a root of prefix,
/// file_name will return None.
#[cfg(not(unix))]
pub(crate) fn file_name<'a, P: AsRef<Path> + ?Sized>(path: &'a P) -> Option<&'a OsStr> {
  path.as_ref().file_name()
}
