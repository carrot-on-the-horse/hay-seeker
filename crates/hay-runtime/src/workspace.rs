//! Locates the repository an executable indexes and searches by default.
//!
//! Zero-setup runs take no path arguments, so every surface has to agree on
//! what "here" means. A repository is one index: the enclosing Git working
//! tree, resolved from the current directory, holds it at
//! `<root>/.hay-seeker/index.duckdb`. That makes `hay search` from a
//! subdirectory find the same index `hay index` wrote from the root, instead of
//! scattering one index per directory the operator happened to stand in.
//!
//! Git is a convenience here, not a requirement. A directory outside any
//! repository — or a host without Git installed — resolves to the directory
//! itself and is walked directly.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};

/// Directory holding a repository-local index and its checkpoint.
pub const INDEX_DIRECTORY: &str = ".hay-seeker";
/// File name of the repository-local `DuckDB` index.
pub const INDEX_FILE: &str = "index.duckdb";

/// The directory an executable indexes and searches when given no path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Workspace {
    root: PathBuf,
    git: bool,
}

impl Workspace {
    /// Resolves the workspace containing `directory`.
    ///
    /// Prefers the enclosing Git working tree so any subdirectory of a
    /// repository resolves to the same root, and falls back to `directory`
    /// itself when there is no repository to find.
    ///
    /// # Errors
    ///
    /// Returns an error when Git reports a root that cannot be canonicalized or
    /// is not UTF-8.
    pub fn resolve(directory: &Path) -> Result<Self> {
        match git_root(directory)? {
            Some(root) => Ok(Self { root, git: true }),
            None => Ok(Self {
                root: directory.to_path_buf(),
                git: false,
            }),
        }
    }

    /// Resolves the workspace containing the current directory.
    ///
    /// # Errors
    ///
    /// Returns an error when the current directory is unavailable or when Git
    /// reports an unusable root.
    pub fn from_current_dir() -> Result<Self> {
        let current = std::env::current_dir().context("read the current directory")?;
        Self::resolve(&current)
    }

    /// Directory that is indexed and that owns the index.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Whether the root is a Git working tree.
    ///
    /// Git working trees are enumerated with `git ls-files`, which honors the
    /// repository's ignore rules; anything else is walked directly.
    #[must_use]
    pub const fn is_git_repository(&self) -> bool {
        self.git
    }

    /// Path of this workspace's index, used when none is configured.
    #[must_use]
    pub fn default_database(&self) -> PathBuf {
        self.root.join(INDEX_DIRECTORY).join(INDEX_FILE)
    }
}

/// Returns the Git working tree containing `directory`.
///
/// A directory outside any repository, an unreadable repository, and a host
/// with no `git` executable all resolve to `None` so callers can fall back to
/// walking the directory itself.
///
/// # Errors
///
/// Returns an error when Git cannot be run for a reason other than being
/// absent, or when it reports a root that is not UTF-8 or cannot be
/// canonicalized.
pub fn git_root(directory: &Path) -> Result<Option<PathBuf>> {
    let output = match Command::new("git")
        .arg("-C")
        .arg(directory)
        .args(["rev-parse", "--show-toplevel"])
        .output()
    {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("detect Git repository for {}", directory.display()));
        }
    };
    if !output.status.success() {
        return Ok(None);
    }
    let value = String::from_utf8(output.stdout).context("Git root is not UTF-8")?;
    let root = PathBuf::from(value.trim());
    Ok(Some(root.canonicalize().with_context(|| {
        format!("canonicalize Git root {}", root.display())
    })?))
}

/// Creates the directory that will hold `database`.
///
/// A `.hay-seeker` directory this call creates is given a `.gitignore` that
/// excludes its own contents, so provisioning an index never adds untracked
/// files to the operator's repository. An existing directory is left alone,
/// including one whose `.gitignore` was deliberately removed.
///
/// # Errors
///
/// Returns an error when the directory or its ignore file cannot be written.
pub fn prepare_index_directory(database: &Path) -> Result<()> {
    let parent = database
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if parent.is_dir() {
        return Ok(());
    }
    std::fs::create_dir_all(parent)
        .with_context(|| format!("create index directory {}", parent.display()))?;
    if parent.file_name() == Some(std::ffi::OsStr::new(INDEX_DIRECTORY)) {
        let ignore = parent.join(".gitignore");
        std::fs::write(&ignore, "*\n")
            .with_context(|| format!("write {}", ignore.display()))
            .or_else(|error| if ignore.exists() { Ok(()) } else { Err(error) })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::process::Command;

    use super::{INDEX_DIRECTORY, Workspace, prepare_index_directory};

    /// Initializes a repository, or reports that this host has no Git.
    fn git_init(root: &Path) -> bool {
        match Command::new("git").arg("-C").arg(root).arg("init").output() {
            Ok(output) => output.status.success(),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
            Err(error) => panic!("run git init: {error}"),
        }
    }

    #[test]
    fn a_subdirectory_resolves_to_the_repository_root() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        if !git_init(&root) {
            return;
        }
        let nested = root.join("crates/deep");
        std::fs::create_dir_all(&nested).unwrap();

        let workspace = Workspace::resolve(&nested).unwrap();

        assert!(workspace.is_git_repository());
        assert_eq!(workspace.root(), root);
        assert_eq!(
            workspace.default_database(),
            root.join(INDEX_DIRECTORY).join("index.duckdb"),
            "one repository has one index, wherever the operator stands"
        );
    }

    #[test]
    fn a_directory_outside_git_is_its_own_workspace() {
        let directory = tempfile::tempdir().unwrap();
        let workspace = Workspace::resolve(directory.path()).unwrap();
        assert!(!workspace.is_git_repository());
        assert_eq!(workspace.root(), directory.path());
        assert_eq!(
            workspace.default_database(),
            directory.path().join(INDEX_DIRECTORY).join("index.duckdb")
        );
    }

    #[test]
    fn a_created_index_directory_ignores_its_own_contents() {
        let directory = tempfile::tempdir().unwrap();
        let database = directory.path().join(INDEX_DIRECTORY).join("index.duckdb");

        prepare_index_directory(&database).unwrap();

        let ignore = directory.path().join(INDEX_DIRECTORY).join(".gitignore");
        assert_eq!(std::fs::read_to_string(ignore).unwrap(), "*\n");
    }

    #[test]
    fn an_explicit_index_path_does_not_gain_an_ignore_file() {
        let directory = tempfile::tempdir().unwrap();
        let database = directory.path().join("indexes").join("index.duckdb");

        prepare_index_directory(&database).unwrap();

        assert!(directory.path().join("indexes").is_dir());
        assert!(!directory.path().join("indexes/.gitignore").exists());
    }
}
