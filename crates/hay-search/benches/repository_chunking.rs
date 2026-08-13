use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};

use cast_core::{ChunkStrategy, LanguageId};
use cast_index::{DocumentId, NormalizedPath};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use hay_search::{Chunker, ChunkerV1, CorpusDocument};

const SOURCES: [(&str, &str, &str); 4] = [
    ("wordpress", "src/wp-includes/class-wp-query.php", "php"),
    ("django", "django/db/models/query.py", "python"),
    ("kubernetes", "pkg/kubelet/kubelet.go", "go"),
    ("ollama", "server/routes.go", "go"),
];

fn benchmark_repository_chunking(criterion: &mut Criterion) {
    let root = repository_root();
    let mut group = criterion.benchmark_group("repository_chunking");

    for (repository, relative_path, language) in SOURCES {
        let source_path = root.join(repository).join(relative_path);
        let source = read_source(&source_path);
        let document = benchmark_document(repository, relative_path, language, source);
        let mut verification_chunker = ChunkerV1::default();
        let verification = verification_chunker
            .chunk(&document)
            .unwrap_or_else(|error| panic!("cannot AST-chunk {}: {error}", source_path.display()));
        assert!(
            verification
                .iter()
                .all(|chunk| chunk.strategy == ChunkStrategy::Ast),
            "{} did not use a compiled Tree-sitter grammar",
            source_path.display()
        );
        group.throughput(Throughput::Bytes(
            u64::try_from(document.text.len()).unwrap_or(u64::MAX),
        ));
        let mut chunker = ChunkerV1::default();

        group.bench_with_input(
            BenchmarkId::new(repository, relative_path),
            &document,
            |bencher, input| {
                bencher.iter(|| {
                    let result = chunker.chunk(black_box(input));
                    match result {
                        Ok(chunks) => black_box(chunks),
                        Err(error) => panic!("chunk benchmark failed: {error}"),
                    }
                });
            },
        );
    }
    group.finish();
}

fn repository_root() -> PathBuf {
    if let Some(root) = std::env::var_os("COTH_HAY_SEEKER_BENCH_REPOS") {
        return PathBuf::from(root);
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(".bench-repos")
}

fn read_source(path: &Path) -> String {
    match fs::read_to_string(path) {
        Ok(source) => source,
        Err(error) => panic!(
            "cannot read {}: {error}; run scripts/fetch-bench-repos.sh first",
            path.display()
        ),
    }
}

fn benchmark_document(
    repository: &str,
    relative_path: &str,
    language: &str,
    text: String,
) -> CorpusDocument {
    let doc_id = match DocumentId::new(format!("{repository}:{relative_path}")) {
        Ok(doc_id) => doc_id,
        Err(error) => panic!("invalid benchmark document ID: {error}"),
    };
    let path = match NormalizedPath::new(format!("{repository}/{relative_path}")) {
        Ok(path) => path,
        Err(error) => panic!("invalid benchmark path: {error}"),
    };
    CorpusDocument {
        doc_id,
        path,
        language: LanguageId::new(language),
        text,
    }
}

criterion_group!(benches, benchmark_repository_chunking);
criterion_main!(benches);
