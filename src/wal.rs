//! Write-Ahead Log (WAL) for durable mutation tracking.
//!
//! The WAL records every mutation (insert, delete) before it is applied to the
//! in-memory index. On startup, any mutations not yet reflected in a saved
//! snapshot are replayed in sequence, ensuring no data loss across restarts.
//!
//! # File Format
//!
//! ```text
//! [Entry]*
//!
//! Entry:
//!   [sequence_number: u64 LE]
//!   [operation_type:  u8   ]
//!   [payload_length:  u64 LE]
//!   [payload:         bytes  ]
//!   [crc32:           u32 LE]  (over sequence + op_type + payload_len + payload)
//! ```
//!
//! # Example
//!
//! ```no_run
//! use forge_db::wal::{WriteAheadLog, WalOperation};
//!
//! let mut wal = WriteAheadLog::open("./wal").unwrap();
//!
//! // Record an insert
//! let op = WalOperation::Insert { id: 1, vector: vec![0.1, 0.2, 0.3], metadata: None };
//! wal.append(&op).unwrap();
//!
//! // Checkpoint: caller has persisted the index snapshot
//! wal.checkpoint(0).unwrap();
//! ```

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{ForgeDbError, Result};

/// WAL entry header size: seq(8) + op_type(1) + payload_len(8) + crc32(4) = 21 bytes.
const HEADER_SIZE: usize = 21;

/// Magic prefix for WAL files (reserved for future header validation).
#[allow(dead_code)]
const WAL_MAGIC: [u8; 4] = *b"FWAL";

/// WAL file version (reserved for future format migrations).
#[allow(dead_code)]
const WAL_VERSION: u8 = 1;

/// Operations that can be logged to the WAL.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WalOperation {
    /// Insert or upsert a vector.
    Insert {
        /// Vector ID.
        id: u64,
        /// Vector data.
        vector: Vec<f32>,
        /// Optional metadata as JSON.
        metadata: Option<String>,
    },

    /// Delete a vector by ID (tombstone).
    Delete {
        /// Vector ID to delete.
        id: u64,
    },

    /// Mark a checkpoint — everything before this is safe to truncate
    /// after a successful snapshot save.
    Checkpoint {
        /// Sequence number of the last applied operation in the snapshot.
        last_applied_seq: u64,
    },

    /// Compact the index (remove tombstoned vectors).
    Compact,
}

impl WalOperation {
    fn type_byte(&self) -> u8 {
        match self {
            WalOperation::Insert { .. } => 1,
            WalOperation::Delete { .. } => 2,
            WalOperation::Checkpoint { .. } => 3,
            WalOperation::Compact => 4,
        }
    }
}

/// A single WAL entry with metadata.
#[derive(Debug, Clone)]
pub struct WalEntry {
    /// Monotonically increasing sequence number.
    pub sequence: u64,
    /// The operation recorded.
    pub operation: WalOperation,
}

/// Write-Ahead Log writer and reader.
///
/// The WAL appends entries to a log file and supports replay on startup.
/// Use `checkpoint()` to truncate entries that have been durably persisted
/// in an index snapshot.
pub struct WriteAheadLog {
    /// Directory containing WAL segment files.
    dir: PathBuf,
    /// Current active log file path.
    current_path: PathBuf,
    /// Buffered writer for the active file.
    writer: BufWriter<File>,
    /// Next sequence number to assign.
    next_seq: u64,
    /// Sequence number of the last checkpoint.
    last_checkpoint_seq: u64,
}

impl WriteAheadLog {
    /// Open (or create) a WAL in the given directory.
    ///
    /// If existing WAL files are found, they are kept for replay.
    /// Call [`replay`](Self::replay) to recover pending entries.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;

        let current_path = dir.join("wal.log");
        let next_seq = Self::recover_next_seq(&current_path)?;

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&current_path)?;

        Ok(Self {
            dir,
            current_path,
            writer: BufWriter::new(file),
            next_seq,
            last_checkpoint_seq: 0,
        })
    }

    /// Append an operation to the WAL.
    ///
    /// The operation is serialized, checksummed, and flushed to disk
    /// before this function returns.
    pub fn append(&mut self, operation: &WalOperation) -> Result<u64> {
        let seq = self.next_seq;
        self.next_seq += 1;

        let payload = Self::serialize_operation(operation)?;
        let entry_bytes = Self::encode_entry(seq, operation.type_byte(), &payload)?;

        self.writer.write_all(&entry_bytes)?;
        self.writer.flush()?;

        Ok(seq)
    }

    /// Flush any buffered writes to the OS (does not guarantee fsync).
    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }

    /// Fsync the WAL file to ensure durability.
    pub fn sync(&mut self) -> Result<()> {
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        Ok(())
    }

    /// Checkpoint: record that all entries up to `last_applied_seq` are
    /// safely persisted in a snapshot. After a successful checkpoint, the
    /// WAL can be truncated on next open.
    pub fn checkpoint(&mut self, last_applied_seq: u64) -> Result<()> {
        self.last_checkpoint_seq = last_applied_seq;
        let op = WalOperation::Checkpoint { last_applied_seq };
        self.append(&op)?;
        self.sync()?;
        Ok(())
    }

    /// Truncate the WAL, removing all entries that have been checkpointed.
    ///
    /// This reads all un-checkpointed entries, rewrites them to a new file,
    /// and atomically replaces the old WAL.
    pub fn truncate_to_checkpoint(&mut self) -> Result<()> {
        let checkpoint_seq = self.last_checkpoint_seq;
        let entries = self.replay_all()?;
        let pending: Vec<_> = entries
            .into_iter()
            .filter(|e| e.sequence > checkpoint_seq)
            .collect();

        let tmp_path = self.dir.join("wal.log.tmp");
        {
            let tmp_file = File::create(&tmp_path)?;
            let mut writer = BufWriter::new(tmp_file);
            for entry in &pending {
                let payload = Self::serialize_operation(&entry.operation)?;
                let bytes = Self::encode_entry(entry.sequence, entry.operation.type_byte(), &payload)?;
                writer.write_all(&bytes)?;
            }
            writer.flush()?;
            writer.get_ref().sync_all()?;
        }

        fs::rename(&tmp_path, &self.current_path)?;

        // Reopen the writer on the new file
        let file = OpenOptions::new()
            .append(true)
            .open(&self.current_path)?;
        self.writer = BufWriter::new(file);

        Ok(())
    }

    /// Replay all entries from the WAL file from the beginning.
    ///
    /// Returns entries in sequence order. Use this on startup to recover
    /// operations not yet reflected in a saved snapshot.
    pub fn replay_all(&self) -> Result<Vec<WalEntry>> {
        Self::read_entries(&self.current_path)
    }

    /// Replay only entries after a given sequence number.
    ///
    /// Use this when you have loaded an index snapshot and only need to
    /// apply mutations that occurred after the snapshot was taken.
    pub fn replay_after(&self, after_seq: u64) -> Result<Vec<WalEntry>> {
        Ok(self
            .replay_all()?
            .into_iter()
            .filter(|e| e.sequence > after_seq)
            .collect())
    }

    /// Return the sequence number of the next entry that will be written.
    pub fn next_sequence(&self) -> u64 {
        self.next_seq
    }

    /// Return the sequence number of the last checkpointed entry.
    pub fn last_checkpoint_seq(&self) -> u64 {
        self.last_checkpoint_seq
    }

    /// Return the number of un-checkpointed entries (approximate).
    pub fn pending_count(&self) -> Result<usize> {
        Ok(self
            .replay_all()?
            .into_iter()
            .filter(|e| e.sequence > self.last_checkpoint_seq)
            .count())
    }

    // ── Private helpers ────────────────────────────────────────────────────────

    fn serialize_operation(op: &WalOperation) -> Result<Vec<u8>> {
        bincode::serialize(op).map_err(|e| ForgeDbError::serialization_error(e.to_string()))
    }

    fn deserialize_operation(bytes: &[u8]) -> Result<WalOperation> {
        bincode::deserialize(bytes).map_err(|e| ForgeDbError::serialization_error(e.to_string()))
    }

    /// Encode a single WAL entry to bytes.
    ///
    /// Layout: [seq: u64][type: u8][payload_len: u64][payload][crc32: u32]
    fn encode_entry(seq: u64, type_byte: u8, payload: &[u8]) -> Result<Vec<u8>> {
        let mut buf = Vec::with_capacity(HEADER_SIZE + payload.len());

        buf.extend_from_slice(&seq.to_le_bytes());
        buf.push(type_byte);
        buf.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        buf.extend_from_slice(payload);

        // CRC32 over everything so far
        let crc = crc32fast::hash(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());

        Ok(buf)
    }

    /// Read and parse all WAL entries from a file path.
    /// Silently stops at the first truncated or corrupt entry.
    fn read_entries(path: &Path) -> Result<Vec<WalEntry>> {
        if !path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut entries = Vec::new();

        loop {
            // Read: seq(8) + type(1) + payload_len(8)
            let mut prefix = [0u8; 17];
            match reader.read_exact(&mut prefix) {
                Ok(_) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(ForgeDbError::Io(e)),
            }

            let seq = u64::from_le_bytes(prefix[0..8].try_into().unwrap());
            let type_byte = prefix[8];
            let payload_len = u64::from_le_bytes(prefix[9..17].try_into().unwrap()) as usize;

            // Read payload
            let mut payload = vec![0u8; payload_len];
            if let Err(e) = reader.read_exact(&mut payload) {
                if e.kind() == io::ErrorKind::UnexpectedEof {
                    break; // truncated entry, stop
                }
                return Err(ForgeDbError::Io(e));
            }

            // Read CRC32
            let mut crc_bytes = [0u8; 4];
            if reader.read_exact(&mut crc_bytes).is_err() {
                break; // truncated
            }
            let stored_crc = u32::from_le_bytes(crc_bytes);

            // Verify checksum over prefix + payload
            let mut to_check = prefix.to_vec();
            to_check.extend_from_slice(&payload);
            let computed_crc = crc32fast::hash(&to_check);

            if computed_crc != stored_crc {
                tracing::warn!(
                    seq,
                    "WAL entry checksum mismatch — stopping replay at this point"
                );
                break;
            }

            // Deserialize
            let _ = type_byte; // type_byte is implicit in the serialized payload
            let operation = match Self::deserialize_operation(&payload) {
                Ok(op) => op,
                Err(e) => {
                    tracing::warn!(seq, error = %e, "WAL entry deserialization failed, stopping replay");
                    break;
                }
            };

            entries.push(WalEntry { sequence: seq, operation });
        }

        Ok(entries)
    }

    /// Scan the WAL file to determine the next sequence number.
    fn recover_next_seq(path: &Path) -> Result<u64> {
        let entries = Self::read_entries(path)?;
        Ok(entries.last().map(|e| e.sequence + 1).unwrap_or(0))
    }
}

impl Drop for WriteAheadLog {
    fn drop(&mut self) {
        let _ = self.writer.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn open_wal(tmp: &TempDir) -> WriteAheadLog {
        WriteAheadLog::open(tmp.path()).expect("open WAL")
    }

    #[test]
    fn test_wal_append_and_replay() {
        let tmp = TempDir::new().unwrap();
        let mut wal = open_wal(&tmp);

        let seq0 = wal
            .append(&WalOperation::Insert {
                id: 1,
                vector: vec![0.1, 0.2, 0.3],
                metadata: None,
            })
            .unwrap();

        let seq1 = wal
            .append(&WalOperation::Insert {
                id: 2,
                vector: vec![0.4, 0.5, 0.6],
                metadata: Some(r#"{"category":"test"}"#.to_string()),
            })
            .unwrap();

        let seq2 = wal.append(&WalOperation::Delete { id: 1 }).unwrap();

        assert_eq!(seq0, 0);
        assert_eq!(seq1, 1);
        assert_eq!(seq2, 2);

        let entries = wal.replay_all().unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].sequence, 0);
        assert_eq!(entries[2].sequence, 2);

        if let WalOperation::Delete { id } = &entries[2].operation {
            assert_eq!(*id, 1);
        } else {
            panic!("expected Delete");
        }
    }

    #[test]
    fn test_wal_replay_after_reopen() {
        let tmp = TempDir::new().unwrap();
        {
            let mut wal = open_wal(&tmp);
            wal.append(&WalOperation::Insert { id: 42, vector: vec![1.0], metadata: None })
                .unwrap();
            wal.append(&WalOperation::Insert { id: 43, vector: vec![2.0], metadata: None })
                .unwrap();
        }

        // Reopen — should continue from seq 2
        let mut wal2 = open_wal(&tmp);
        assert_eq!(wal2.next_sequence(), 2);

        wal2.append(&WalOperation::Delete { id: 42 }).unwrap();
        let entries = wal2.replay_all().unwrap();
        assert_eq!(entries.len(), 3);
    }

    #[test]
    fn test_wal_checkpoint_and_replay_after() {
        let tmp = TempDir::new().unwrap();
        let mut wal = open_wal(&tmp);

        wal.append(&WalOperation::Insert { id: 1, vector: vec![0.1], metadata: None }).unwrap();
        wal.append(&WalOperation::Insert { id: 2, vector: vec![0.2], metadata: None }).unwrap();
        wal.checkpoint(1).unwrap(); // everything through seq 1 is in snapshot

        wal.append(&WalOperation::Insert { id: 3, vector: vec![0.3], metadata: None }).unwrap();

        let pending = wal.replay_after(1).unwrap();
        // seq 2 = checkpoint entry, seq 3 = Insert{id=3}
        // replay_after(1) gives entries with seq > 1
        assert!(pending.iter().any(|e| {
            matches!(&e.operation, WalOperation::Insert { id, .. } if *id == 3)
        }));
    }

    #[test]
    fn test_wal_truncate_to_checkpoint() {
        let tmp = TempDir::new().unwrap();
        let mut wal = open_wal(&tmp);

        for i in 0..5u64 {
            wal.append(&WalOperation::Insert { id: i, vector: vec![i as f32], metadata: None })
                .unwrap();
        }

        // Checkpoint everything through seq 3
        wal.checkpoint(3).unwrap();
        wal.truncate_to_checkpoint().unwrap();

        // After truncation only entries after seq 3 remain (seq 4 insert + seq 5 checkpoint)
        let entries = wal.replay_all().unwrap();
        assert!(entries.iter().all(|e| e.sequence > 3));
    }

    #[test]
    fn test_wal_empty_replay() {
        let tmp = TempDir::new().unwrap();
        let wal = open_wal(&tmp);
        let entries = wal.replay_all().unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_wal_insert_with_metadata() {
        let tmp = TempDir::new().unwrap();
        let mut wal = open_wal(&tmp);
        let meta = r#"{"tag":"production","score":0.95}"#.to_string();
        wal.append(&WalOperation::Insert {
            id: 99,
            vector: vec![1.0, 2.0],
            metadata: Some(meta.clone()),
        })
        .unwrap();

        let entries = wal.replay_all().unwrap();
        if let WalOperation::Insert { id, metadata, .. } = &entries[0].operation {
            assert_eq!(*id, 99);
            assert_eq!(metadata.as_deref(), Some(meta.as_str()));
        } else {
            panic!("expected Insert");
        }
    }
}
