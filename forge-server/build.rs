fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&["proto/forge.proto"], &["proto"])?;

    println!("cargo:rerun-if-changed=proto/forge.proto");
    Ok(())
}
