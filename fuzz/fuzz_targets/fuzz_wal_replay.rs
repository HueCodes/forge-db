#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Write;

fuzz_target!(|data: &[u8]| {
    // Write arbitrary bytes as a WAL file and attempt to replay them.
    // The WAL should handle any corrupted input gracefully by returning
    // Ok with partial entries or stopping at the first corrupt entry —
    // it must never panic.

    let tmp = tempfile::tempdir().expect("create temp dir");
    let wal_path = tmp.path().join("wal.log");

    // Write fuzzed bytes as the WAL log file
    {
        let mut f = std::fs::File::create(&wal_path).expect("create wal file");
        f.write_all(data).expect("write fuzzed data");
        f.sync_all().expect("sync");
    }

    // Open the WAL directory — this triggers recover_next_seq which reads entries
    let wal_result = forge_db::WriteAheadLog::open(tmp.path());

    if let Ok(wal) = wal_result {
        // Replay all entries — should never panic on corrupt data
        let _ = wal.replay_all();
    }
});
