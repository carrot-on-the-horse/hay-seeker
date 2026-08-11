use std::path::PathBuf;

use ort::session::Session;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: onnx_contract /absolute/path/to/model.onnx")?;
    let session = Session::builder()?.commit_from_file(&path)?;

    println!("model: {}", path.display());
    for input in session.inputs() {
        println!("input: {} {:?}", input.name(), input.dtype());
    }
    for output in session.outputs() {
        println!("output: {} {:?}", output.name(), output.dtype());
    }
    Ok(())
}
