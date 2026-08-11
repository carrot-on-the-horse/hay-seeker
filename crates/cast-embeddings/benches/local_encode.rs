use std::hint::black_box;
use std::time::Duration;

use cast_embeddings::{LocalOnnxConfig, LocalOnnxEmbedder};
use cast_index::{DocumentId, Embedder, EmbeddingInput};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use futures::executor::block_on;

fn benchmark_local_encode(criterion: &mut Criterion) {
    dotenvy::dotenv().ok();
    let Ok(bundle_dir) = std::env::var("HAY_LOCAL_MODEL_DIR") else {
        eprintln!("skipping local_encode: HAY_LOCAL_MODEL_DIR is not set");
        return;
    };
    let embedder = LocalOnnxEmbedder::new(LocalOnnxConfig::new(bundle_dir))
        .expect("HAY_LOCAL_MODEL_DIR must contain a valid pinned bundle");
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
        .collect::<Result<Vec<_>, _>>()
        .expect("synthetic IDs are valid");
    let inputs = ids
        .iter()
        .zip(texts)
        .map(|(document_id, text)| EmbeddingInput { document_id, text })
        .collect::<Vec<_>>();
    let long_document = synthetic_long_document();
    let long_document_id =
        DocumentId::new("synthetic-long-document").expect("synthetic ID is valid");
    let long_input = [EmbeddingInput {
        document_id: &long_document_id,
        text: &long_document,
    }];

    let mut group = criterion.benchmark_group("local_onnx_encode");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    for batch_size in [1_usize, 8] {
        group.throughput(Throughput::Elements(
            u64::try_from(batch_size).unwrap_or(u64::MAX),
        ));
        group.bench_with_input(
            BenchmarkId::new("documents", batch_size),
            &batch_size,
            |bencher, batch_size| {
                bencher.iter(|| {
                    black_box(
                        block_on(embedder.embed_batch(&inputs[..*batch_size]))
                            .expect("local document encode succeeds"),
                    )
                });
            },
        );
    }
    group.throughput(Throughput::Elements(1));
    group.bench_function("query", |bencher| {
        bencher.iter(|| {
            black_box(
                block_on(
                    embedder.embed_query("where is repository checkpoint compatibility validated?"),
                )
                .expect("local query encode succeeds"),
            )
        });
    });
    group.throughput(Throughput::Elements(1));
    group.bench_function("long_document", |bencher| {
        bencher.iter(|| {
            black_box(
                block_on(embedder.embed_batch(&long_input))
                    .expect("local long-document encode succeeds"),
            )
        });
    });
    group.finish();
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

criterion_group!(benches, benchmark_local_encode);
criterion_main!(benches);
