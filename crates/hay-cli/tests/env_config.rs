use std::fs;
use std::path::Path;
use std::process::{Command, Output};

/// Every variable the executables read, cleared so a developer's own
/// configuration can never change a test result.
const ISOLATED_ENV_VARS: &[&str] = &[
    "COTH_HAY_SEEKER_AUTO_INDEX",
    "COTH_HAY_SEEKER_BACKEND",
    "COTH_HAY_SEEKER_BENCH_REPOS",
    "COTH_HAY_SEEKER_CANDIDATE_LIMIT",
    "COTH_HAY_SEEKER_CF_AIG_TOKEN",
    "COTH_HAY_SEEKER_CHECKPOINT",
    "COTH_HAY_SEEKER_CLOUDFLARE_ACCOUNT_ID",
    "COTH_HAY_SEEKER_CLOUDFLARE_AI_MAX_ATTEMPTS",
    "COTH_HAY_SEEKER_CLOUDFLARE_AI_TOKEN",
    "COTH_HAY_SEEKER_CLOUDFLARE_WORKERS_AI_MODEL_REVISION",
    "COTH_HAY_SEEKER_CORPUS",
    "COTH_HAY_SEEKER_DATABASE",
    "COTH_HAY_SEEKER_DOWNLOAD_MODELS",
    "COTH_HAY_SEEKER_ELASTICSEARCH_API_KEY",
    "COTH_HAY_SEEKER_ELASTICSEARCH_BEARER_TOKEN",
    "COTH_HAY_SEEKER_ELASTICSEARCH_ENDPOINT",
    "COTH_HAY_SEEKER_ELASTICSEARCH_INDEX",
    "COTH_HAY_SEEKER_EMBEDDINGS",
    "COTH_HAY_SEEKER_GEMINI_EMBEDDING_CONCURRENCY",
    "COTH_HAY_SEEKER_GEMINI_EMBEDDING_DIMENSIONS",
    "COTH_HAY_SEEKER_GEMINI_EMBEDDING_MAX_ATTEMPTS",
    "COTH_HAY_SEEKER_GEMINI_GATEWAY_URL",
    "COTH_HAY_SEEKER_GEMINI_MODEL_REVISION",
    "COTH_HAY_SEEKER_GEMINI_SMOKE_CHUNK_TOKENS",
    "COTH_HAY_SEEKER_LOCAL_MODEL_DIR",
    "COTH_HAY_SEEKER_LOCAL_RESEARCH_ELASTICSEARCH_DIMENSIONS",
    "COTH_HAY_SEEKER_LOCAL_STATIC_MODEL_DIR",
    "COTH_HAY_SEEKER_LOCAL_STORED_DIMENSIONS",
    "COTH_HAY_SEEKER_MODEL_BASE_URL",
    "COTH_HAY_SEEKER_MODEL_CACHE_DIR",
    "COTH_HAY_SEEKER_OPENAI_API_KEY",
    "COTH_HAY_SEEKER_OPENAI_EMBEDDING_DIMENSIONS",
    "COTH_HAY_SEEKER_OPENAI_EMBEDDING_MAX_ATTEMPTS",
    "COTH_HAY_SEEKER_OPENAI_EMBEDDING_MODEL",
    "COTH_HAY_SEEKER_OPENAI_GATEWAY_URL",
    "COTH_HAY_SEEKER_OPENAI_MODEL_REVISION",
    "COTH_HAY_SEEKER_OPENAI_STATIC_ADDRESS",
    "COTH_HAY_SEEKER_PROGRESS_INTERVAL_SECONDS",
    "COTH_HAY_SEEKER_QUERY",
    "COTH_HAY_SEEKER_REPOSITORY",
    "COTH_HAY_SEEKER_STALL_TIMEOUT_SECONDS",
    "COTH_HAY_SEEKER_TOP_K",
    "COTH_HAY_SEEKER_VOYAGE_EMBEDDING_DIMENSIONS",
    "COTH_HAY_SEEKER_VOYAGE_EMBEDDING_MAX_ATTEMPTS",
    "COTH_HAY_SEEKER_VOYAGE_EMBEDDING_MODEL",
    "COTH_HAY_SEEKER_VOYAGE_MODEL_REVISION",
    "COTH_HAY_SEEKER_VOYAGE_TOKEN",
];

/// Builds an isolated `hay` invocation that cannot reach the network.
///
/// The default provider is `local-static`, so an un-pinned test would provision
/// a model from upstream. Every test here is about argument and `.env`
/// plumbing, so downloads are refused and the cache is redirected into the
/// test's own directory; a developer's warm cache cannot change the result
/// either.
fn hay_command(current_dir: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_hay"));
    command.current_dir(current_dir);
    for name in ISOLATED_ENV_VARS {
        command.env_remove(name);
    }
    command.env("COTH_HAY_SEEKER_DOWNLOAD_MODELS", "false");
    command.env(
        "COTH_HAY_SEEKER_MODEL_CACHE_DIR",
        current_dir.join("model-cache"),
    );
    command
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "hay failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn write_repository(root: &Path) {
    let source = root.join("repository/src");
    fs::create_dir_all(&source).unwrap();
    fs::write(
        source.join("lib.rs"),
        "pub fn configured_from_dotenv() -> bool { true }\n",
    )
    .unwrap();
}

#[test]
fn dotenv_can_supply_required_index_arguments() {
    let directory = tempfile::tempdir().unwrap();
    write_repository(directory.path());
    let database = directory.path().join("from-env.duckdb");
    fs::write(
        directory.path().join(".env"),
        format!(
            "COTH_HAY_SEEKER_DATABASE={}\nCOTH_HAY_SEEKER_REPOSITORY={}\nCOTH_HAY_SEEKER_EMBEDDINGS=none\n",
            database.display(),
            directory.path().join("repository").display()
        ),
    )
    .unwrap();

    let output = hay_command(directory.path()).arg("index").output().unwrap();

    assert_success(&output);
    assert!(database.is_file());
}

#[test]
fn command_line_value_overrides_dotenv_value() {
    let directory = tempfile::tempdir().unwrap();
    write_repository(directory.path());
    let env_database = directory.path().join("from-env.duckdb");
    let cli_database = directory.path().join("from-cli.duckdb");
    fs::write(
        directory.path().join(".env"),
        format!(
            "COTH_HAY_SEEKER_DATABASE={}\nCOTH_HAY_SEEKER_REPOSITORY={}\nCOTH_HAY_SEEKER_EMBEDDINGS=none\n",
            env_database.display(),
            directory.path().join("repository").display()
        ),
    )
    .unwrap();

    let output = hay_command(directory.path())
        .args(["index", "--database"])
        .arg(&cli_database)
        .output()
        .unwrap();

    assert_success(&output);
    assert!(cli_database.is_file());
    assert!(!env_database.exists());
}

#[test]
fn help_names_environment_variables_without_revealing_values() {
    let directory = tempfile::tempdir().unwrap();
    let secret_path = directory.path().join("private-repository-name");
    let output = hay_command(directory.path())
        .args(["index", "--help"])
        .env("COTH_HAY_SEEKER_REPOSITORY", &secret_path)
        .output()
        .unwrap();

    assert_success(&output);
    let help = String::from_utf8(output.stdout).unwrap();
    assert!(help.contains("env: COTH_HAY_SEEKER_REPOSITORY"));
    assert!(!help.contains("private-repository-name"));
}

#[test]
fn malformed_dotenv_fails_instead_of_using_partial_configuration() {
    let directory = tempfile::tempdir().unwrap();
    fs::write(
        directory.path().join(".env"),
        "COTH_HAY_SEEKER_DATABASE='unterminated\n",
    )
    .unwrap();

    let output = hay_command(directory.path())
        .arg("--help")
        .output()
        .unwrap();

    assert!(!output.status.success());
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains("load .env configuration"));
    assert!(error.contains("Error parsing line"));
}

#[test]
fn dotenv_selects_the_embedding_provider() {
    let directory = tempfile::tempdir().unwrap();
    write_repository(directory.path());
    fs::write(
        directory.path().join(".env"),
        format!(
            concat!(
                "COTH_HAY_SEEKER_DATABASE={}\n",
                "COTH_HAY_SEEKER_REPOSITORY={}\n",
                "COTH_HAY_SEEKER_EMBEDDINGS=local-static\n"
            ),
            directory.path().join("index.duckdb").display(),
            directory.path().join("repository").display()
        ),
    )
    .unwrap();

    let output = hay_command(directory.path()).arg("index").output().unwrap();

    assert!(!output.status.success());
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(
        error.contains("minishlab/potion-code-16M-v2"),
        "the failure names the model it could not provision: {error}"
    );
    assert!(
        error.contains("COTH_HAY_SEEKER_DOWNLOAD_MODELS=true")
            && error.contains("COTH_HAY_SEEKER_LOCAL_STATIC_MODEL_DIR"),
        "the failure states both remedies: {error}"
    );
}

/// The zero-setup default is what makes `hay index` work without credentials,
/// so a silent revert to lexical-only must fail this suite.
#[test]
fn the_default_embedding_provider_is_the_static_code_model() {
    let directory = tempfile::tempdir().unwrap();
    write_repository(directory.path());

    let output = hay_command(directory.path())
        .args(["index", "--repository"])
        .arg(directory.path().join("repository"))
        .args(["--database"])
        .arg(directory.path().join("index.duckdb"))
        .output()
        .unwrap();

    assert!(
        !output.status.success(),
        "with downloads refused and no staged bundle, the default cannot index"
    );
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(
        error.contains("potion-code-16M-v2"),
        "the default provider is the static code model: {error}"
    );
}

/// A staged bundle must win over provisioning so air-gapped installs keep
/// working exactly as before.
#[test]
fn a_staged_bundle_directory_is_used_without_downloading() {
    let directory = tempfile::tempdir().unwrap();
    write_repository(directory.path());
    let staged = directory.path().join("staged-bundle");
    fs::create_dir_all(&staged).unwrap();

    let output = hay_command(directory.path())
        .args(["index", "--repository"])
        .arg(directory.path().join("repository"))
        .args(["--database"])
        .arg(directory.path().join("index.duckdb"))
        .env("COTH_HAY_SEEKER_LOCAL_STATIC_MODEL_DIR", &staged)
        .output()
        .unwrap();

    assert!(!output.status.success(), "the staged bundle is empty");
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(
        error.contains("static embedding bundle") && error.contains("staged-bundle"),
        "the staged directory is opened directly instead of being provisioned: {error}"
    );
}
