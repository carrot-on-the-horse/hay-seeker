//! End-to-end behavior of a run that was given no paths.
//!
//! Every test drives the real executable with pipes for standard input and
//! standard error, which is exactly what an automated caller looks like. That
//! makes the CI guarantee testable: a `hay` that prompted here would hang this
//! suite instead of passing it.
//!
//! Downloads are refused and the model cache is redirected, so `--embeddings
//! none` keeps these tests offline and free of the 33 MB bundle.

use std::fs;
use std::path::Path;
use std::process::{Command, Output};

const ISOLATED_ENV_VARS: &[&str] = &[
    "COTH_HAY_SEEKER_AUTO_INDEX",
    "COTH_HAY_SEEKER_BACKEND",
    "COTH_HAY_SEEKER_CORPUS",
    "COTH_HAY_SEEKER_DATABASE",
    "COTH_HAY_SEEKER_EMBEDDINGS",
    "COTH_HAY_SEEKER_QUERY",
    "COTH_HAY_SEEKER_REPOSITORY",
    "HAY_LOCAL_STATIC_MODEL_DIR",
];

/// Builds an isolated, offline `hay` invocation rooted at `current_dir`.
fn hay(current_dir: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_hay"));
    command.current_dir(current_dir);
    for name in ISOLATED_ENV_VARS {
        command.env_remove(name);
    }
    command.env("COTH_HAY_SEEKER_DOWNLOAD_MODELS", "false");
    command.env("COTH_HAY_SEEKER_MODEL_CACHE_DIR", current_dir.join("cache"));
    command.env("COTH_HAY_SEEKER_EMBEDDINGS", "none");
    command
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "hay failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        stderr(output)
    );
}

/// Writes a repository whose only source file is findable by one query.
fn write_source(root: &Path) {
    let source = root.join("src");
    fs::create_dir_all(&source).unwrap();
    fs::write(
        source.join("lib.rs"),
        "pub fn haystack_needle() -> u32 { 42 }\n",
    )
    .unwrap();
}

/// Creates a Git repository, or reports that this host has no Git.
fn git_init(root: &Path) -> bool {
    match Command::new("git").arg("-C").arg(root).arg("init").output() {
        Ok(output) => output.status.success(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => panic!("run git init: {error}"),
    }
}

fn index_path(root: &Path) -> std::path::PathBuf {
    root.join(".hay-seeker").join("index.duckdb")
}

/// An automated caller is told what to run instead of being asked anything.
#[test]
fn searching_without_an_index_fails_closed_when_nobody_can_answer() {
    let directory = tempfile::tempdir().unwrap();
    write_source(directory.path());

    let output = hay(directory.path())
        .args(["search", "needle"])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let error = stderr(&output);
    assert!(
        error.contains("no index at"),
        "the failure names the missing index: {error}"
    );
    assert!(
        error.contains("hay index") && error.contains("COTH_HAY_SEEKER_AUTO_INDEX=always"),
        "the failure states both remedies: {error}"
    );
    assert!(
        !index_path(directory.path()).exists(),
        "a refused search must not leave an empty index behind"
    );
}

#[test]
fn never_refuses_to_index_even_without_asking() {
    let directory = tempfile::tempdir().unwrap();
    write_source(directory.path());

    let output = hay(directory.path())
        .args(["search", "needle", "--auto-index", "never"])
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert!(stderr(&output).contains("no index at"));
    assert!(!index_path(directory.path()).exists());
}

/// The setting an automated caller opts in with: index, then answer the query.
#[test]
fn always_indexes_the_current_directory_and_then_searches_it() {
    let directory = tempfile::tempdir().unwrap();
    write_source(directory.path());

    let output = hay(directory.path())
        .args(["search", "haystack_needle", "--auto-index", "always"])
        .output()
        .unwrap();

    assert_success(&output);
    assert!(index_path(directory.path()).is_file());
    let report = stderr(&output);
    assert!(
        report.contains("indexed") && report.contains("chunks"),
        "the build reports to standard error: {report}"
    );

    let results: serde_json::Value = serde_json::from_slice(&output.stdout)
        .expect("standard output carries exactly one JSON document");
    assert_eq!(results["backend"], "duckdb");
    let hits = results["results"].as_array().unwrap();
    assert!(
        hits.iter()
            .any(|hit| hit["path"].as_str() == Some("src/lib.rs")),
        "the freshly built index answers the query: {results}"
    );
}

/// `CI` withdraws the offer even where a terminal would have allowed it.
#[test]
fn continuous_integration_never_turns_into_a_prompt() {
    let directory = tempfile::tempdir().unwrap();
    write_source(directory.path());

    let output = hay(directory.path())
        .args(["search", "needle"])
        .env("CI", "true")
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert!(stderr(&output).contains("no index at"));
}

/// An index path someone configured is never filled in by guesswork.
#[test]
fn an_explicitly_configured_index_is_not_built_implicitly() {
    let directory = tempfile::tempdir().unwrap();
    write_source(directory.path());
    let elsewhere = directory.path().join("managed").join("index.duckdb");

    let output = hay(directory.path())
        .args(["search", "needle", "--database"])
        .arg(&elsewhere)
        .output()
        .unwrap();

    assert!(!output.status.success());
    let error = stderr(&output);
    assert!(
        error.contains("hay index --database"),
        "the remedy keeps the configured path: {error}"
    );
    assert!(!elsewhere.exists());
}

#[test]
fn indexing_with_no_arguments_indexes_the_current_directory() {
    let directory = tempfile::tempdir().unwrap();
    write_source(directory.path());

    let output = hay(directory.path()).arg("index").output().unwrap();

    assert_success(&output);
    assert!(index_path(directory.path()).is_file());
    let report = stderr(&output);
    assert!(
        report.contains(&directory.path().display().to_string()),
        "the run names the directory it chose: {report}"
    );
}

/// One repository, one index: a subdirectory must not start a second one.
#[test]
fn a_subdirectory_uses_the_repository_index() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().canonicalize().unwrap();
    if !git_init(&root) {
        return;
    }
    write_source(&root);
    let nested = root.join("src");

    let indexed = hay(&nested).arg("index").output().unwrap();
    assert_success(&indexed);
    assert!(
        index_path(&root).is_file(),
        "the index belongs to the repository root, not to src/"
    );
    assert!(!index_path(&nested).exists());

    let found = hay(&nested)
        .args(["search", "haystack_needle"])
        .output()
        .unwrap();

    assert_success(&found);
    let results: serde_json::Value = serde_json::from_slice(&found.stdout).unwrap();
    assert!(
        !results["results"].as_array().unwrap().is_empty(),
        "search from src/ reads the repository index: {results}"
    );
}

/// The index directory keeps itself out of the operator's `git status`.
#[test]
fn a_provisioned_index_directory_ignores_itself() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().canonicalize().unwrap();
    if !git_init(&root) {
        return;
    }
    write_source(&root);

    assert_success(&hay(&root).arg("index").output().unwrap());

    assert_eq!(
        fs::read_to_string(root.join(".hay-seeker/.gitignore")).unwrap(),
        "*\n"
    );
    let status = Command::new("git")
        .arg("-C")
        .arg(&root)
        .args(["status", "--porcelain"])
        .output()
        .unwrap();
    let status = String::from_utf8_lossy(&status.stdout);
    assert!(
        !status.contains(".hay-seeker"),
        "the index must not show up as an untracked change: {status}"
    );
}
