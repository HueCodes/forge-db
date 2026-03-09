#![no_main]

use forge_db::Persistable;
use libfuzzer_sys::fuzz_target;
use std::io::Write;

fuzz_target!(|data: &[u8]| {
    // Write arbitrary bytes as a .fdb file and attempt to load a BruteForceIndex.
    // The loader must return an Err for malformed data — it must never panic.

    let tmp = tempfile::tempdir().expect("create temp dir");
    let fdb_path = tmp.path().join("fuzzed.fdb");

    // Write fuzzed bytes as the index file
    {
        let mut f = std::fs::File::create(&fdb_path).expect("create fdb file");
        f.write_all(data).expect("write fuzzed data");
        f.sync_all().expect("sync");
    }

    // Attempt to load — should return Err, not panic
    let _ = forge_db::BruteForceIndex::load(&fdb_path);
});
