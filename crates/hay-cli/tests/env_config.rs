use std::fs;
use std::path::Path;
use std::process::{Command, Output};

const ISOLATED_ENV_VARS: &[&str] = &[
    "COTH_HAY_SEEKER_BACKEND",
    "COTH_HAY_SEEKER_CANDIDATE_LIMIT",
    "COTH_HAY_SEEKER_CF_AIG_TOKEN",
    "COTH_HAY_SEEKER_CHECKPOINT",
    "COTH_HAY_SEEKER_CORPUS",
    "COTH_HAY_SEEKER_DATABASE",
    "COTH_HAY_SEEKER_ELASTICSEARCH_ENDPOINT",
    "COTH_HAY_SEEKER_ELASTICSEARCH_INDEX",
    "COTH_HAY_SEEKER_EMBEDDINGS",
    "COTH_HAY_SEEKER_PROGRESS_INTERVAL_SECONDS",
    "COTH_HAY_SEEKER_OPENAI_API_KEY",
    "COTH_HAY_SEEKER_QUERY",
    "COTH_HAY_SEEKER_REPOSITORY",
    "COTH_HAY_SEEKER_STALL_TIMEOUT_SECONDS",
    "COTH_HAY_SEEKER_TOP_K",
    "HAY_LOCAL_STATIC_MODEL_DIR",
];

fn hay_command(current_dir: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_hay"));
    command.current_dir(current_dir);
    for name in ISOLATED_ENV_VARS {
        command.env_remove(name);
    }
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
            "COTH_HAY_SEEKER_DATABASE={}\nCOTH_HAY_SEEKER_REPOSITORY={}\n",
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
            "COTH_HAY_SEEKER_DATABASE={}\nCOTH_HAY_SEEKER_REPOSITORY={}\n",
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
    assert!(error.contains("HAY_LOCAL_STATIC_MODEL_DIR is required"));
}
