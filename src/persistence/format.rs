//! File format definitions for forge-db persistence.

use crate::error::{ForgeDbError, Result};

/// Magic bytes identifying a forge-db file: "FORGEDB\0"
pub const MAGIC: [u8; 8] = *b"FORGEDB\0";

/// Current format version.
pub const FORMAT_VERSION: u32 = 1;

/// Index type identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum IndexType {
    /// IVF-PQ index
    IvfPq = 1,
    /// HNSW index
    Hnsw = 2,
    /// IVF index (without PQ)
    Ivf = 3,
    /// Brute force index
    BruteForce = 4,
}

impl IndexType {
    /// Convert from u32.
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            1 => Some(Self::IvfPq),
            2 => Some(Self::Hnsw),
            3 => Some(Self::Ivf),
            4 => Some(Self::BruteForce),
            _ => None,
        }
    }
}

/// File format flags.
#[derive(Debug, Clone, Copy, Default)]
pub struct FormatFlags {
    bits: u32,
}

impl FormatFlags {
    /// No special flags.
    pub const NONE: u32 = 0;
    /// Index includes original vectors for reranking.
    pub const HAS_RERANK_VECTORS: u32 = 1 << 0;
    /// Index includes metadata.
    pub const HAS_METADATA: u32 = 1 << 1;
    /// Data is compressed with LZ4.
    pub const LZ4_COMPRESSED: u32 = 1 << 2;

    /// Create new flags from bits.
    pub fn new(bits: u32) -> Self {
        Self { bits }
    }

    /// Get the raw bits.
    pub fn bits(&self) -> u32 {
        self.bits
    }

    /// Check if a flag is set.
    pub fn has(&self, flag: u32) -> bool {
        self.bits & flag != 0
    }

    /// Set a flag.
    pub fn set(&mut self, flag: u32) {
        self.bits |= flag;
    }
}

/// File header structure.
///
/// Total size: 24 bytes
/// ```text
/// [MAGIC 8B][VERSION u32][INDEX_TYPE u32][FLAGS u32][CHECKSUM u32]
/// ```
#[derive(Debug, Clone)]
pub struct FileHeader {
    /// Magic bytes (must be MAGIC)
    pub magic: [u8; 8],
    /// Format version
    pub version: u32,
    /// Index type
    pub index_type: IndexType,
    /// Format flags
    pub flags: FormatFlags,
    /// CRC32 checksum of the data section (everything after header)
    pub checksum: u32,
}

impl FileHeader {
    /// Header size in bytes.
    pub const SIZE: usize = 24;

    /// Create a new header.
    pub fn new(index_type: IndexType, checksum: u32) -> Self {
        Self {
            magic: MAGIC,
            version: FORMAT_VERSION,
            index_type,
            flags: FormatFlags::default(),
            checksum,
        }
    }

    /// Create a new header with flags.
    pub fn with_flags(index_type: IndexType, flags: FormatFlags, checksum: u32) -> Self {
        Self {
            magic: MAGIC,
            version: FORMAT_VERSION,
            index_type,
            flags,
            checksum,
        }
    }

    /// Serialize header to bytes.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut bytes = [0u8; Self::SIZE];
        bytes[0..8].copy_from_slice(&self.magic);
        bytes[8..12].copy_from_slice(&self.version.to_le_bytes());
        bytes[12..16].copy_from_slice(&(self.index_type as u32).to_le_bytes());
        bytes[16..20].copy_from_slice(&self.flags.bits().to_le_bytes());
        bytes[20..24].copy_from_slice(&self.checksum.to_le_bytes());
        bytes
    }

    /// Deserialize header from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < Self::SIZE {
            return Err(ForgeDbError::invalid_format("header too small"));
        }

        let mut magic = [0u8; 8];
        magic.copy_from_slice(&bytes[0..8]);

        if magic != MAGIC {
            return Err(ForgeDbError::invalid_format("invalid magic bytes"));
        }

        let version = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let index_type_raw = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        let flags_raw = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        let checksum = u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);

        let index_type = IndexType::from_u32(index_type_raw)
            .ok_or_else(|| ForgeDbError::invalid_format("unknown index type"))?;

        Ok(Self {
            magic,
            version,
            index_type,
            flags: FormatFlags::new(flags_raw),
            checksum,
        })
    }

    /// Verify the header is valid and matches expected type.
    pub fn verify(&self, expected_type: IndexType) -> Result<()> {
        if self.magic != MAGIC {
            return Err(ForgeDbError::invalid_format("invalid magic bytes"));
        }

        if self.version > FORMAT_VERSION {
            return Err(ForgeDbError::invalid_format(format!(
                "unsupported version {} (max supported: {})",
                self.version, FORMAT_VERSION
            )));
        }

        if self.index_type != expected_type {
            return Err(ForgeDbError::invalid_format(format!(
                "index type mismatch: expected {:?}, got {:?}",
                expected_type, self.index_type
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let header = FileHeader::new(IndexType::IvfPq, 0x12345678);
        let bytes = header.to_bytes();
        let parsed = FileHeader::from_bytes(&bytes).unwrap();

        assert_eq!(parsed.magic, MAGIC);
        assert_eq!(parsed.version, FORMAT_VERSION);
        assert_eq!(parsed.index_type, IndexType::IvfPq);
        assert_eq!(parsed.checksum, 0x12345678);
    }

    #[test]
    fn test_header_with_flags() {
        let mut flags = FormatFlags::default();
        flags.set(FormatFlags::HAS_RERANK_VECTORS);
        flags.set(FormatFlags::HAS_METADATA);

        let header = FileHeader::with_flags(IndexType::Hnsw, flags, 0xDEADBEEF);
        let bytes = header.to_bytes();
        let parsed = FileHeader::from_bytes(&bytes).unwrap();

        assert!(parsed.flags.has(FormatFlags::HAS_RERANK_VECTORS));
        assert!(parsed.flags.has(FormatFlags::HAS_METADATA));
        assert!(!parsed.flags.has(FormatFlags::LZ4_COMPRESSED));
    }

    #[test]
    fn test_invalid_magic() {
        let mut bytes = [0u8; FileHeader::SIZE];
        bytes[0..8].copy_from_slice(b"INVALID\0");

        let result = FileHeader::from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_type_mismatch() {
        let header = FileHeader::new(IndexType::IvfPq, 0);
        let result = header.verify(IndexType::Hnsw);
        assert!(result.is_err());
    }

    #[test]
    fn test_index_type_from_u32() {
        assert_eq!(IndexType::from_u32(1), Some(IndexType::IvfPq));
        assert_eq!(IndexType::from_u32(2), Some(IndexType::Hnsw));
        assert_eq!(IndexType::from_u32(99), None);
    }
}
