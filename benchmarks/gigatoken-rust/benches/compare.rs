use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use gigatoken_rs::load_tokenizer::tiktoken::load_tiktoken;
use gigatoken_rs::pretokenize::PretokenizerType;

const GIGATOKEN_REVISION: &str = "fac0114b37120ec8a76362e9ee8e1c742aaafaef";
const O200K_SHA256: &str = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d";
const SOURCES: [(&str, &str); 4] = [
    ("wordpress", "src/wp-includes/class-wp-query.php"),
    ("django", "django/db/models/query.py"),
    ("kubernetes", "pkg/kubelet/kubelet.go"),
    ("ollama", "server/routes.go"),
];

fn benchmark_tokenizers(criterion: &mut Criterion) {
    let workspace = workspace_root();
    let repository_root = repository_root(&workspace);
    let ranks = workspace.join(".bench-assets/o200k_base.tiktoken");
    assert!(
        ranks.is_file(),
        "missing {}; run scripts/bench-tokenizers.sh",
        ranks.display()
    );

    let mut group = criterion.benchmark_group("o200k_tokenizers");
    for (repository, relative_path) in SOURCES {
        let path = repository_root.join(repository).join(relative_path);
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
        let expected = tiktoken_rs::o200k_base_singleton().encode_ordinary(&source);
        let gigatoken = load_tiktoken(&ranks, PretokenizerType::O200k, Vec::new())
            .unwrap_or_else(|error| panic!("cannot load GigaToken ranks: {error}"));
        let mut warm_gigatoken = gigatoken.fork();
        let mut actual = Vec::new();
        warm_gigatoken.encode_with_added_tokens_flat(source.as_bytes(), &mut actual);
        assert_eq!(actual, expected, "token mismatch for {}", path.display());

        group.throughput(Throughput::Bytes(
            u64::try_from(source.len()).unwrap_or(u64::MAX),
        ));
        group.bench_with_input(
            BenchmarkId::new("tiktoken_rs", repository),
            source.as_str(),
            |bencher, input| {
                bencher.iter(|| {
                    black_box(tiktoken_rs::o200k_base_singleton().count_ordinary(black_box(input)))
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("gigatoken_warm", repository),
            source.as_str(),
            |bencher, input| {
                bencher.iter(|| {
                    let mut count = 0;
                    warm_gigatoken.encode_with_added_tokens(
                        black_box(input.as_bytes()),
                        |tokens| {
                            count += tokens.len();
                        },
                    );
                    black_box(count)
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("gigatoken_cold_cache", repository),
            source.as_str(),
            |bencher, input| {
                bencher.iter_batched(
                    || gigatoken.fork(),
                    |mut cold_gigatoken| {
                        let mut count = 0;
                        cold_gigatoken.encode_with_added_tokens(
                            black_box(input.as_bytes()),
                            |tokens| {
                                count += tokens.len();
                            },
                        );
                        black_box(count)
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();

    eprintln!("GigaToken revision: {GIGATOKEN_REVISION}");
    eprintln!("o200k_base SHA-256: {O200K_SHA256}");
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root must exist")
}

fn repository_root(workspace: &Path) -> PathBuf {
    std::env::var_os("HAY_BENCH_REPOS")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace.join(".bench-repos"))
}

criterion_group!(benches, benchmark_tokenizers);
criterion_main!(benches);
