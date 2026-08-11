use std::fs;
use std::io::{self, Read};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use cast_core::{ByteSizer, ChunkConfig, LineSizer, Sizer, UnicodeWordSizer};
use cast_tokenizers::OpenAiBpeSizer;
use cast_tree_sitter::TreeSitterChunker;
use clap::{Parser, ValueEnum};

#[derive(Debug, Parser)]
#[command(name = "cast", version, about = "AST-aware semantic source chunker")]
struct Arguments {
    /// UTF-8 source path, or - for stdin.
    path: PathBuf,

    /// Explicit language. Common aliases such as js, py, golang, c++, and c# are accepted.
    #[arg(long)]
    language: Option<String>,

    /// Maximum chunk size in the selected sizing unit.
    #[arg(long, default_value = "1500")]
    max_size: NonZeroUsize,

    /// Unit used to enforce max-size.
    #[arg(long, value_enum, default_value_t = SizerName::OpenAi)]
    sizer: SizerName,

    /// Hard byte ceiling per chunk, regardless of sizing unit.
    #[arg(long, default_value = "25000")]
    max_chunk_bytes: NonZeroUsize,

    /// Emit pretty JSON instead of compact JSON.
    #[arg(long)]
    pretty: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum SizerName {
    OpenAi,
    Bytes,
    Words,
    Lines,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("cast: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = Arguments::parse();
    let source = read_source(&arguments.path)?;
    let sizer: Arc<dyn Sizer> = match arguments.sizer {
        SizerName::OpenAi => Arc::new(OpenAiBpeSizer),
        SizerName::Bytes => Arc::new(ByteSizer),
        SizerName::Words => Arc::new(UnicodeWordSizer),
        SizerName::Lines => Arc::new(LineSizer),
    };
    let config = ChunkConfig {
        max_size: arguments.max_size,
        max_chunk_bytes: Some(arguments.max_chunk_bytes),
        ..ChunkConfig::default()
    };
    let mut chunker = TreeSitterChunker::new(sizer);
    let output = if let Some(language) = arguments.language.as_deref() {
        chunker.chunk(&source, language, &config)?
    } else if arguments.path.as_os_str() == "-" {
        return Err("--language is required when reading stdin".into());
    } else {
        chunker.chunk_path(&source, &arguments.path, &config)?
    };

    if arguments.pretty {
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("{}", serde_json::to_string(&output)?);
    }
    Ok(())
}

fn read_source(path: &PathBuf) -> Result<String, Box<dyn std::error::Error>> {
    if path.as_os_str() == "-" {
        let mut source = String::new();
        io::stdin().read_to_string(&mut source)?;
        return Ok(source);
    }

    Ok(fs::read_to_string(path)?)
}
