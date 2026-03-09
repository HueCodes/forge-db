#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Feed arbitrary bytes to FileHeader::from_bytes.
    // It must return Err for invalid data — it must never panic.

    let _ = forge_db::persistence::FileHeader::from_bytes(data);
});
