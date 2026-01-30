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

/// Write header and data to file.
pub(crate) fn write_with_header(
    path: impl AsRef<Path>,
    index_type: IndexType,
    data: &[u8],
) -> Result<()> {
    use std::io::Write;

    let checksum = crc32fast::hash(data);
    let header = FileHeader::new(index_type, checksum);

    let mut file = std::fs::File::create(path)?;
    file.write_all(&header.to_bytes())?;
    file.write_all(data)?;
    file.sync_all()?;

    Ok(())
}
