use std::env;
use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::Instant;

use cast_core::LanguageId;
use cast_embeddings::{CloudflareVertexGemini2, CloudflareVertexGemini2Config};
use cast_index::{DocumentId, Embedder, EmbeddingInput, EmbeddingVector, NormalizedPath};
use hay_search::{Chunker, ChunkerV1, CorpusDocument, FixedWindowConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();
    let mut arguments = env::args().skip(1);
    let path = arguments.next().map_or_else(default_source, PathBuf::from);
    let query = {
        let value = arguments.collect::<Vec<_>>().join(" ");
        if value.is_empty() {
            "where are API routes registered?".to_owned()
        } else {
            value
        }
    };

    let source = fs::read_to_string(&path)?;
    let relative = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("repository smoke path must end in a valid UTF-8 file name")?;
    let document = CorpusDocument {
        doc_id: DocumentId::new("gemini-repo-smoke")?,
        path: NormalizedPath::new(relative)?,
        language: LanguageId::from(language_for_path(&path)?),
        text: source,
    };

    let max_chunk_size = env::var("COTH_HAY_SEEKER_GEMINI_SMOKE_CHUNK_TOKENS")
        .ok()
        .map_or(Ok(1_500), |value| value.parse())?;
    let mut chunker = ChunkerV1::new(
        NonZeroUsize::new(max_chunk_size)
            .ok_or("COTH_HAY_SEEKER_GEMINI_SMOKE_CHUNK_TOKENS must be non-zero")?,
        FixedWindowConfig::default(),
    )?;
    let chunks = chunker.chunk(&document)?;
    if chunks.is_empty() {
        return Err("CAST produced no chunks".into());
    }

    let embedder = embedder_from_env()?;
    let inputs = chunks
        .iter()
        .map(|chunk| EmbeddingInput {
            document_id: &chunk.chunk_id,
            text: &chunk.text,
        })
        .collect::<Vec<_>>();
    let started = Instant::now();
    let vectors = embedder.embed_batch(&inputs).await?;
    let query_vector = embedder.embed_query(&query).await?;
    let elapsed = started.elapsed();
    if vectors.len() != chunks.len() {
        return Err("embedding output count did not match CAST chunk count".into());
    }

    let mut ranked = chunks
        .iter()
        .zip(&vectors)
        .map(|(chunk, vector)| (cosine(&query_vector, vector), chunk))
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.ordinal.cmp(&right.1.ordinal))
    });

    println!(
        "file={} bytes={} chunks={} vectors={} dimensions={} elapsed_ms={} query={query:?}",
        path.display(),
        document.text.len(),
        chunks.len(),
        vectors.len(),
        query_vector.values.len(),
        elapsed.as_millis(),
    );
    for (rank, (score, chunk)) in ranked.into_iter().take(5).enumerate() {
        println!(
            "rank={} score={score:.6} chunk={} line={} bytes={}..{} snippet={:?}",
            rank + 1,
            chunk.ordinal,
            chunk.core_range.start.line,
            chunk.core_range.start_byte,
            chunk.core_range.end_byte,
            snippet(&chunk.text),
        );
    }
    Ok(())
}

fn embedder_from_env() -> Result<CloudflareVertexGemini2, Box<dyn std::error::Error>> {
    let token = env::var("COTH_HAY_SEEKER_CF_AIG_TOKEN")?;
    if token.trim().is_empty() {
        return Err("COTH_HAY_SEEKER_CF_AIG_TOKEN is empty".into());
    }
    let endpoint = env::var("COTH_HAY_SEEKER_GEMINI_GATEWAY_URL")?;
    let mut config = CloudflareVertexGemini2Config::new(endpoint, token);
    if let Ok(dimensions) = env::var("COTH_HAY_SEEKER_GEMINI_EMBEDDING_DIMENSIONS") {
        config = config.with_dimensions(dimensions.parse()?);
    }
    if let Ok(concurrency) = env::var("COTH_HAY_SEEKER_GEMINI_EMBEDDING_CONCURRENCY") {
        config = config.with_max_concurrency(concurrency.parse()?);
    }
    Ok(CloudflareVertexGemini2::new(config)?)
}

fn default_source() -> PathBuf {
    PathBuf::from(".bench-repos/ollama/server/routes.go")
}

fn language_for_path(path: &Path) -> Result<&'static str, Box<dyn std::error::Error>> {
    let language = match path.extension().and_then(|extension| extension.to_str()) {
        Some("go") => "go",
        Some("py") => "python",
        Some("php") => "php",
        Some("rs") => "rust",
        Some("js" | "mjs" | "cjs") => "javascript",
        Some("ts") => "typescript",
        Some("tsx") => "tsx",
        Some("java") => "java",
        Some("rb") => "ruby",
        Some("c") => "c",
        Some("cc" | "cpp" | "cxx") => "cpp",
        Some("cs") => "c_sharp",
        Some("sh" | "bash") => "bash",
        _ => return Err("unsupported repository smoke-test extension".into()),
    };
    Ok(language)
}

fn cosine(left: &EmbeddingVector, right: &EmbeddingVector) -> f64 {
    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    for (left, right) in left.values.iter().zip(&right.values) {
        let left = f64::from(*left);
        let right = f64::from(*right);
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }
    dot / (left_norm.sqrt() * right_norm.sqrt())
}

fn snippet(text: &str) -> String {
    let flattened = text.split_whitespace().collect::<Vec<_>>().join(" ");
    flattened.chars().take(120).collect()
}
