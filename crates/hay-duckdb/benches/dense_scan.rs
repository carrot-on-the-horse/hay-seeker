use std::hint::black_box;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Duration;

use cast_core::LanguageId;
use cast_index::{
    BoxFuture, ContentHash, Embedder, EmbeddingIdentity, EmbeddingInput, EmbeddingVector,
    HashAlgorithm, IndexError, IndexErrorKind, NormalizedPath,
};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use futures::executor::block_on;
use hay_duckdb::DuckDbIndex;
use hay_search::{
    DocId, FdeParams, IndexManifest, Quantization, Query, Retriever, SearchDocument, SearchOpts,
};

#[path = "../src/vector.rs"]
#[allow(dead_code)]
mod vector;

use vector::{cosine_int8, encode_int8, prepare_query};

const DIMENSIONS: usize = 256;

struct FixedQueryEmbedder {
    identity: EmbeddingIdentity,
    query: Vec<f32>,
}

impl Embedder for FixedQueryEmbedder {
    fn identity(&self) -> &EmbeddingIdentity {
        &self.identity
    }

    fn embed_batch<'a>(
        &'a self,
        _inputs: &'a [EmbeddingInput<'a>],
    ) -> BoxFuture<'a, Result<Vec<EmbeddingVector>, IndexError>> {
        Box::pin(async {
            Err(IndexError::new(
                IndexErrorKind::Invariant,
                "benchmark_documents_are_preembedded",
                "benchmark documents must already contain vectors",
            ))
        })
    }

    fn embed_query<'a>(
        &'a self,
        _text: &'a str,
    ) -> BoxFuture<'a, Result<EmbeddingVector, IndexError>> {
        Box::pin(async move {
            Ok(EmbeddingVector {
                identity: self.identity.clone(),
                values: self.query.clone(),
            })
        })
    }
}

fn synthetic_vector(seed: usize) -> Vec<f32> {
    let mut state = u64::try_from(seed).unwrap_or(u64::MAX) ^ 0x9e37_79b9_7f4a_7c15;
    let mut vector = Vec::with_capacity(DIMENSIONS);
    for _ in 0..DIMENSIONS {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let sample = u16::try_from(state & 0xffff).unwrap_or_default();
        vector.push(f32::from(sample) / f32::from(u16::MAX) * 2.0 - 1.0);
    }
    vector
}

fn benchmark_dense_scan(criterion: &mut Criterion) {
    let query = synthetic_vector(0);
    let query = prepare_query(&query).expect("synthetic query is finite");
    let mut group = criterion.benchmark_group("duckdb_int8_exact_scan");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));

    for document_count in [30_000_usize, 250_000] {
        let vectors = (1..=document_count)
            .map(|seed| encode_int8(&synthetic_vector(seed)).expect("synthetic vector is finite"))
            .collect::<Vec<_>>();
        let elements = u64::try_from(document_count.saturating_mul(DIMENSIONS)).unwrap_or(u64::MAX);
        group.throughput(Throughput::Elements(elements));
        group.bench_with_input(
            BenchmarkId::from_parameter(document_count),
            &vectors,
            |bencher, vectors| {
                bencher.iter(|| {
                    let mut maximum = f32::NEG_INFINITY;
                    for vector in vectors {
                        let score = cosine_int8(&query, vector)
                            .expect("benchmark vectors use the current codec");
                        maximum = maximum.max(score);
                    }
                    black_box(maximum)
                });
            },
        );
    }
    group.finish();
}

fn benchmark_duckdb_dense_query(criterion: &mut Criterion) {
    const DOCUMENT_COUNT: usize = 30_000;
    let identity = EmbeddingIdentity {
        provider: "benchmark".into(),
        model: "synthetic-256".into(),
        dimensions: DIMENSIONS,
        profile: "symmetric-v1".into(),
    };
    let manifest = IndexManifest {
        model_id: identity.model.clone(),
        model_revision: "benchmark-revision".into(),
        embedding_profile: identity.profile.clone(),
        embed_dim: DIMENSIONS,
        mrl_dim: DIMENSIONS,
        quantization: Quantization::Int8PerVectorScaleOffset,
        tokenizer_hash: ContentHash::new(HashAlgorithm::Sha256, "a".repeat(64))
            .expect("benchmark digest is valid"),
        chunker_version: "benchmark-chunker".into(),
        fde_params: FdeParams::Disabled,
        schema_version: 1,
    };
    let embedder: Arc<dyn Embedder> = Arc::new(FixedQueryEmbedder {
        identity,
        query: synthetic_vector(0),
    });
    let index =
        DuckDbIndex::open_in_memory(manifest, Some(embedder)).expect("benchmark index opens");
    let documents = (1..=DOCUMENT_COUNT)
        .map(|ordinal| SearchDocument {
            doc_id: DocId::new(format!("dense-{ordinal:06}")).expect("benchmark ID is valid"),
            path: NormalizedPath::new(format!("src/dense_{ordinal:06}.rs"))
                .expect("benchmark path is valid"),
            language: LanguageId::new("rust"),
            text: "synthetic dense benchmark payload".into(),
            embedding: Some(synthetic_vector(ordinal)),
        })
        .collect::<Vec<_>>();
    block_on(index.replace_all(&documents)).expect("benchmark index builds");
    let query = Query::new("unmatched semantic query").expect("benchmark query is valid");
    let options = SearchOpts {
        top_k: NonZeroUsize::new(10).expect("nonzero"),
        candidate_limit: NonZeroUsize::new(50).expect("nonzero"),
        enable_late_interaction: false,
    };

    let mut group = criterion.benchmark_group("duckdb_int8_full_query");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(
        u64::try_from(DOCUMENT_COUNT.saturating_mul(DIMENSIONS)).unwrap_or(u64::MAX),
    ));
    group.bench_function(BenchmarkId::from_parameter(DOCUMENT_COUNT), |bencher| {
        bencher.iter(|| {
            black_box(block_on(index.search(&query, &options)).expect("benchmark search succeeds"))
        });
    });
    group.finish();
}

criterion_group!(benches, benchmark_dense_scan, benchmark_duckdb_dense_query);
criterion_main!(benches);
