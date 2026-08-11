#![cfg(feature = "popular-languages")]

use std::env;
use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cast_core::{ChunkConfig, NodeKindMode, ParsePolicy};
use cast_tokenizers::OpenAiBpeSizer;
use cast_tree_sitter::TreeSitterChunker;

const UPDATE_ENV: &str = "UPDATE_CAST_GOLDENS";

#[test]
fn go_routes_matches_golden() {
    assert_golden("go_routes.go");
}

#[test]
fn recovered_python_matches_golden() {
    assert_golden("recovered.py");
}

#[test]
fn minified_generic_fallback_matches_golden() {
    assert_golden("minified.json");
}

fn assert_golden(name: &str) {
    let fixture = fixture_path(name);
    let source = fs::read_to_string(&fixture)
        .unwrap_or_else(|error| panic!("read fixture {}: {error}", fixture.display()));
    let config = ChunkConfig {
        max_size: NonZeroUsize::new(40).unwrap_or(NonZeroUsize::MIN),
        max_chunk_bytes: NonZeroUsize::new(240),
        parse_policy: ParsePolicy::GenericFallback,
        include_node_kinds: NodeKindMode::AllNamed,
        ..ChunkConfig::default()
    };
    let mut chunker = TreeSitterChunker::new(Arc::new(OpenAiBpeSizer));
    let output = chunker
        .chunk_path(&source, &fixture, &config)
        .unwrap_or_else(|error| panic!("chunk fixture {}: {error}", fixture.display()));
    let actual = format!(
        "{}\n",
        serde_json::to_string_pretty(&output).expect("serialize golden output")
    );
    let golden = golden_path(name);

    if env::var_os(UPDATE_ENV).as_deref() == Some(std::ffi::OsStr::new("1")) {
        assert!(
            env::var_os("CI").is_none(),
            "refusing to update CAST goldens in CI"
        );
        fs::write(&golden, &actual)
            .unwrap_or_else(|error| panic!("write golden {}: {error}", golden.display()));
    }

    let expected = fs::read_to_string(&golden).unwrap_or_else(|error| {
        panic!(
            "read golden {}: {error}; review with `{UPDATE_ENV}=1 cargo test -p cast-tree-sitter --test golden`",
            golden.display()
        )
    });
    assert_eq!(
        actual, expected,
        "CAST output changed for {name}; inspect the diff, then regenerate intentionally with `{UPDATE_ENV}=1 cargo test -p cast-tree-sitter --test golden`"
    );
}

fn fixture_path(name: &str) -> PathBuf {
    tests_root().join("fixtures").join(name)
}

fn golden_path(name: &str) -> PathBuf {
    tests_root()
        .join("goldens")
        .join(format!("{name}.approved.json"))
}

fn tests_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests")
}
