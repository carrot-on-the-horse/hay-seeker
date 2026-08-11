use std::time::Instant;

use cast_embeddings::{LocalStaticConfig, LocalStaticEmbedder};
use cast_index::{DocumentId, Embedder, EmbeddingInput};
use futures::executor::block_on;
use serde::Serialize;

#[derive(Serialize)]
struct SmokeResult {
    model: String,
    dimensions: usize,
    startup_ms: f64,
    first_query_ms: f64,
    warm_query_ms: f64,
    document_batch_size: usize,
    document_batch_ms: f64,
    documents_per_second: f64,
    long_document_bytes: usize,
    long_document_ms: f64,
    query_norm: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();
    let bundle_dir = std::env::var("HAY_LOCAL_STATIC_MODEL_DIR")?;
    let startup = Instant::now();
    let embedder = LocalStaticEmbedder::new(LocalStaticConfig::new(bundle_dir))?;
    let startup_ms = milliseconds(startup.elapsed());

    let first_query = Instant::now();
    let query =
        block_on(embedder.embed_query("where is repository checkpoint compatibility validated?"))?;
    let first_query_ms = milliseconds(first_query.elapsed());

    let texts = [
        "Validate the stored index manifest before retrieval.",
        "Apply changed files and deleted document IDs atomically.",
        "Split source files at syntax-tree boundaries.",
        "Retry transient embedding failures with a bounded policy.",
        "Swap the Elasticsearch alias after a complete build.",
        "Persist content hashes for incremental repository indexing.",
        "Fuse lexical and dense candidate ranks deterministically.",
        "Reject a tokenizer or model revision mismatch.",
    ];
    let ids = (0..texts.len())
        .map(|ordinal| DocumentId::new(format!("synthetic-{ordinal}")))
        .collect::<Result<Vec<_>, _>>()?;
    let inputs = ids
        .iter()
        .zip(texts)
        .map(|(document_id, text)| EmbeddingInput { document_id, text })
        .collect::<Vec<_>>();
    let document_batch = Instant::now();
    let documents = block_on(embedder.embed_batch(&inputs))?;
    let document_batch_ms = milliseconds(document_batch.elapsed());

    let long_document = synthetic_long_document();
    let long_document_id = DocumentId::new("synthetic-long-document")?;
    let long_input = [EmbeddingInput {
        document_id: &long_document_id,
        text: &long_document,
    }];
    let long_started = Instant::now();
    let long_output = block_on(embedder.embed_batch(&long_input))?;
    let long_document_ms = milliseconds(long_started.elapsed());

    let warm_query = Instant::now();
    let warm =
        block_on(embedder.embed_query("where is repository checkpoint compatibility validated?"))?;
    let warm_query_ms = milliseconds(warm_query.elapsed());
    if warm.values.len() != query.values.len()
        || documents.len() != texts.len()
        || long_output.len() != 1
    {
        return Err("local static embedder returned an inconsistent output count".into());
    }
    let query_norm = query
        .values
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    let documents_per_second = if document_batch_ms == 0.0 {
        f64::INFINITY
    } else {
        f64::from(u32::try_from(documents.len())?) * 1_000.0 / document_batch_ms
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&SmokeResult {
            model: embedder.identity().model.clone(),
            dimensions: embedder.identity().dimensions,
            startup_ms,
            first_query_ms,
            warm_query_ms,
            document_batch_size: documents.len(),
            document_batch_ms,
            documents_per_second,
            long_document_bytes: long_document.len(),
            long_document_ms,
            query_norm,
        })?
    );
    Ok(())
}

fn synthetic_long_document() -> String {
    use std::fmt::Write as _;

    (0..64).fold(String::new(), |mut document, ordinal| {
        let _ = writeln!(
            document,
            "fn validate_manifest_{ordinal}() {{ assert_runtime_contract(index, tokenizer, chunker); }}"
        );
        document
    })
}

fn milliseconds(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}
