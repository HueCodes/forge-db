//! Persistence layer for saving and loading forge-db indexes.
//!
//! This module provides serialization and deserialization of indexes to disk,
//! with support for checksums, versioning, and memory-mapped loading.
//!
//! # File Format
//!
//! ```text
//! [MAGIC 8B "FORGEDB\0"][VERSION u32][INDEX_TYPE u32][FLAGS u32][CHECKSUM u32]
//! [METADATA_LEN u64][METADATA bincode]
//! [DATA sections... (index-specific)]
//! ```
//!
//! # Example
//!
//! ```ignore
//! use forge_db::persistence::Persistable;
//!
//! // Save an index
//! index.save("my_index.fdb")?;
//!
//! // Load an index
//! let loaded = IVFPQIndex::load("my_index.fdb")?;
//! ```

mod format;

pub use format::{FileHeader, IndexType, FORMAT_VERSION, MAGIC};

use crate::error::{ForgeDbError, Result};
use std::path::Path;
use tracing::{instrument, warn};

/// Trait for types that can be persisted to disk.
pub trait Persistable: Sized {
    /// Save the index to a file.
    ///
    /// # Arguments
    /// * `path` - Path to save the index to
    ///
    /// # Errors
    /// Returns an error if the file cannot be written or serialization fails.
    fn save(&self, path: impl AsRef<Path>) -> Result<()>;

    /// Load an index from a file.
    ///
    /// # Arguments
    /// * `path` - Path to load the index from
    ///
    /// # Errors
    /// Returns an error if the file cannot be read, is corrupted, or has an
    /// incompatible format.
    fn load(path: impl AsRef<Path>) -> Result<Self>;

    /// Load an index from a memory-mapped file for reduced memory usage.
    ///
    /// This is useful for very large indexes where loading the entire file
    /// into memory would be prohibitive. The index data remains on disk and
    /// is paged in as needed.
    ///
    /// # Arguments
    /// * `path` - Path to load the index from
    ///
    /// # Errors
    /// Returns an error if the file cannot be memory-mapped or is corrupted.
    ///
    /// # Note
    /// Not all index types support memory-mapped loading. If not supported,
    /// this falls back to regular loading.
    fn load_mmap(path: impl AsRef<Path>) -> Result<Self> {
        // Default implementation falls back to regular load
        Self::load(path)
    }
}

/// Verify file header and return the data section.
pub(crate) fn verify_header(data: &[u8], expected_type: IndexType) -> Result<&[u8]> {
    if data.len() < FileHeader::SIZE {
        return Err(ForgeDbError::invalid_format("file too small for header"));
    }

    let header = FileHeader::from_bytes(&data[..FileHeader::SIZE])?;
    header.verify(expected_type)?;

    // Verify checksum of the data section
    let data_section = &data[FileHeader::SIZE..];
    let computed_checksum = crc32fast::hash(data_section);

    if computed_checksum != header.checksum {
        return Err(ForgeDbError::ChecksumMismatch);
    }

    Ok(data_section)
}

/// Check that the filesystem containing `path` has at least `required_bytes` free.
///
/// On Unix, this uses `statvfs` to query available space. On non-Unix platforms
/// the check is skipped with a warning.
///
/// A 10% safety margin is applied: the check requires `required_bytes * 1.1`
/// to be available, so the filesystem is not filled to the very last byte.
pub fn check_disk_space(path: &Path, required_bytes: u64) -> Result<()> {
    // Apply a 10% safety margin
    let required_with_margin = required_bytes + required_bytes / 10;

    #[cfg(unix)]
    {
        use std::ffi::CString;

        // Resolve to the parent directory (the path itself may not exist yet)
        let check_path = path
            .parent()
            .filter(|p| p.exists())
            .unwrap_or_else(|| Path::new("."));

        let c_path = CString::new(
            check_path
                .to_str()
                .ok_or_else(|| ForgeDbError::invalid_parameter("path is not valid UTF-8"))?,
        )
        .map_err(|e| ForgeDbError::invalid_parameter(format!("invalid path: {e}")))?;

        let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
        let ret = unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) };

        if ret != 0 {
            let io_err = std::io::Error::last_os_error();
            warn!(
                path = %check_path.display(),
                error = %io_err,
                "failed to check disk space, proceeding with write"
            );
            return Ok(());
        }

        #[allow(clippy::unnecessary_cast)] // Types differ across platforms
        let available = stat.f_bavail as u64 * stat.f_frsize as u64;

        if available < required_with_margin {
            return Err(ForgeDbError::insufficient_disk_space(
                required_with_margin,
                available,
            ));
        }

        tracing::debug!(
            available_bytes = available,
            required_bytes = required_with_margin,
            "disk space check passed"
        );
    }

    #[cfg(not(unix))]
    {
        let _ = required_with_margin;
        warn!(
            path = %path.display(),
            required_bytes,
            "disk space check not supported on this platform, skipping"
        );
    }

    Ok(())
}

/// Write header and data to file.
#[instrument(skip(data), fields(path = %path.as_ref().display(), bytes = data.len()))]
pub(crate) fn write_with_header(
    path: impl AsRef<Path>,
    index_type: IndexType,
    data: &[u8],
) -> Result<()> {
    use std::io::Write;

    let total_bytes = (FileHeader::SIZE + data.len()) as u64;
    check_disk_space(path.as_ref(), total_bytes)?;

    let checksum = crc32fast::hash(data);
    let header = FileHeader::new(index_type, checksum);

    let mut file = std::fs::File::create(path)?;
    file.write_all(&header.to_bytes())?;
    file.write_all(data)?;
    file.sync_all()?;

    Ok(())
}

#[cfg(test)]
mod tests_disk_space {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_check_disk_space_small_write_succeeds() {
        let path = PathBuf::from("/tmp/forge_test_disk_check.fdb");
        let result = check_disk_space(&path, 1024);
        assert!(result.is_ok());
    }

    #[test]
    fn test_check_disk_space_impossibly_large_write_fails() {
        let path = PathBuf::from("/tmp/forge_test_disk_check.fdb");
        let result = check_disk_space(&path, u64::MAX / 2);
        assert!(result.is_err());
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("insufficient disk space"),
            "expected disk space error, got: {msg}"
        );
    }

    #[test]
    fn test_check_disk_space_nonexistent_parent_falls_back() {
        let path = PathBuf::from("/nonexistent_dir_abc123/test.fdb");
        let result = check_disk_space(&path, 1024);
        assert!(result.is_ok());
    }
}
