use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};

use cast_core::{Sizer, UnicodeWordSizer};
use cast_tokenizers::OpenAiBpeSizer;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

const SOURCES: [(&str, &str); 4] = [
    ("wordpress", "src/wp-includes/class-wp-query.php"),
    ("django", "django/db/models/query.py"),
    ("kubernetes", "pkg/kubelet/kubelet.go"),
    ("ollama", "server/routes.go"),
];

fn benchmark_sizers(criterion: &mut Criterion) {
    let root = repository_root();
    let mut group = criterion.benchmark_group("source_sizing");
    let sizers: [(&str, &dyn Sizer); 2] = [
        ("unicode_words", &UnicodeWordSizer),
        ("openai_o200k", &OpenAiBpeSizer),
    ];

    for (repository, relative_path) in SOURCES {
        let source_path = root.join(repository).join(relative_path);
        let source = read_source(&source_path);
        group.throughput(Throughput::Bytes(
            u64::try_from(source.len()).unwrap_or(u64::MAX),
        ));

        for (name, sizer) in sizers {
            group.bench_with_input(
                BenchmarkId::new(name, repository),
                source.as_str(),
                |bencher, input| {
                    bencher.iter(|| {
                        black_box(
                            sizer
                                .measure(black_box(input))
                                .unwrap_or_else(|error| panic!("sizer failed: {error}")),
                        )
                    });
                },
            );
        }
    }
    group.finish();
}

fn repository_root() -> PathBuf {
    if let Some(root) = std::env::var_os("HAY_BENCH_REPOS") {
        return PathBuf::from(root);
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(".bench-repos")
}

fn read_source(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|error| {
        panic!(
            "cannot read {}: {error}; run scripts/fetch-bench-repos.sh first",
            path.display()
        )
    })
}

criterion_group!(benches, benchmark_sizers);
criterion_main!(benches);
