//! Checksum-enforced provisioning for pinned local model bundles.
//!
//! Provisioning never decides what is acceptable. A [`LocalModelEntry`] already
//! pins the revision, the artifact list, each artifact's byte length, and each
//! artifact's SHA-256, so this module only moves bytes into place and refuses
//! anything that differs. A hostile mirror, a truncated transfer, and a
//! corrupted cache all fail the same way, which is why the transport is allowed
//! to be plain HTTP against a user-chosen base URL.
//!
//! The bytes written are identical to a hand-provisioned bundle, including the
//! manifest, so a downloaded model produces the same index identity as one
//! staged by hand.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use reqwest::{Client, Url};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::model_catalog::{LocalModelArtifact, LocalModelEntry};

/// Upstream host used when no mirror is configured.
pub const DEFAULT_MODEL_BASE_URL: &str = "https://huggingface.co";

const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const READ_TIMEOUT: Duration = Duration::from_secs(60);
const PROGRESS_INTERVAL_BYTES: u64 = 4 * 1024 * 1024;

/// Failure while provisioning a pinned model bundle.
#[derive(Debug, Error)]
pub enum ModelFetchError {
    /// The bundle is incomplete and downloads are turned off.
    #[error(
        "{model} is not provisioned in {} and model downloads are disabled",
        directory.display()
    )]
    DownloadDisabled {
        /// Upstream model identifier.
        model: &'static str,
        /// Bundle directory that was checked.
        directory: PathBuf,
    },
    /// The configured base URL is not a usable HTTP or HTTPS base.
    #[error("invalid model base URL: {0}")]
    InvalidBaseUrl(String),
    /// A cache file or directory could not be read or written.
    #[error("failed to prepare model cache at {}: {message}", directory.display())]
    Cache {
        /// Directory being prepared.
        directory: PathBuf,
        /// Underlying error text.
        message: String,
    },
    /// The transport failed before the artifact was complete.
    #[error("failed to download {path} for {model}: {message}")]
    Transport {
        /// Upstream model identifier.
        model: &'static str,
        /// Artifact being fetched.
        path: &'static str,
        /// Underlying error text.
        message: String,
    },
    /// The origin answered with a non-success status.
    #[error("download of {path} for {model} returned HTTP {status}")]
    Status {
        /// Upstream model identifier.
        model: &'static str,
        /// Artifact being fetched.
        path: &'static str,
        /// Status code returned by the origin.
        status: u16,
    },
    /// The transferred length did not match the pinned length.
    #[error("{path} for {model} is {actual} bytes, expected {expected}")]
    Length {
        /// Upstream model identifier.
        model: &'static str,
        /// Artifact being fetched.
        path: &'static str,
        /// Bytes actually received.
        actual: u64,
        /// Bytes the catalog pins.
        expected: u64,
    },
    /// The transferred bytes did not match the pinned digest.
    #[error("{path} for {model} failed checksum verification: expected {expected}, got {actual}")]
    Checksum {
        /// Upstream model identifier.
        model: &'static str,
        /// Artifact being fetched.
        path: &'static str,
        /// Digest the catalog pins.
        expected: &'static str,
        /// Digest computed from the received bytes.
        actual: String,
    },
    /// The HTTP client could not be constructed.
    #[error("could not construct the model download client")]
    HttpClient,
}

/// Progress signal emitted while a bundle is provisioned.
#[derive(Clone, Copy, Debug)]
pub enum ModelFetchEvent {
    /// Every artifact was already present at its pinned length.
    Cached {
        /// Model that was found complete.
        model: &'static str,
    },
    /// A cold or partial provisioning run started.
    Started {
        /// Model being provisioned.
        model: &'static str,
        /// Upstream license identifier.
        license: &'static str,
        /// Bytes this run will transfer.
        total_bytes: u64,
    },
    /// Bytes arrived for one artifact.
    Progress {
        /// Artifact being fetched.
        path: &'static str,
        /// Bytes received so far.
        downloaded: u64,
        /// Bytes the catalog pins for this artifact.
        total_bytes: u64,
    },
    /// One artifact was verified and committed.
    Verified {
        /// Artifact that passed its checksum.
        path: &'static str,
    },
    /// Every artifact is present and verified.
    Finished {
        /// Model that is now provisioned.
        model: &'static str,
    },
}

/// Receives [`ModelFetchEvent`] values during provisioning.
pub type ModelFetchReporter<'a> = &'a (dyn Fn(ModelFetchEvent) + Send + Sync);

/// Where bundles are cached and whether provisioning may use the network.
#[derive(Clone, Debug)]
pub struct ModelFetchConfig {
    cache_dir: PathBuf,
    base_url: String,
    allow_download: bool,
}

impl ModelFetchConfig {
    /// Caches under `cache_dir` and downloads from the default upstream host.
    #[must_use]
    pub fn new(cache_dir: impl Into<PathBuf>) -> Self {
        Self {
            cache_dir: cache_dir.into(),
            base_url: DEFAULT_MODEL_BASE_URL.to_owned(),
            allow_download: true,
        }
    }

    /// Redirects downloads to a mirror or proxy.
    ///
    /// Artifacts stay pinned by length and digest, so a mirror can serve the
    /// bytes faster or from inside a private network but cannot substitute a
    /// different model.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Turns network provisioning on or off.
    #[must_use]
    pub const fn with_allow_download(mut self, allow_download: bool) -> Self {
        self.allow_download = allow_download;
        self
    }

    /// Directory a given entry's artifacts are provisioned into.
    ///
    /// The revision is part of the path, so a future catalog bump provisions
    /// beside the old bundle instead of overwriting a working one.
    #[must_use]
    pub fn bundle_dir(&self, entry: &LocalModelEntry) -> PathBuf {
        self.cache_dir.join(entry.key).join(entry.revision)
    }
}

/// Returns a verified bundle directory, downloading artifacts when permitted.
///
/// Presence is checked by pinned byte length rather than by digest so a warm
/// start does not re-hash the model table; the loading provider verifies every
/// digest before use, which is the authoritative check.
///
/// # Errors
///
/// Returns [`ModelFetchError`] when the bundle is incomplete and downloads are
/// disabled, when the cache cannot be written, or when any transferred artifact
/// fails its pinned length or digest.
pub async fn ensure_bundle(
    entry: &'static LocalModelEntry,
    config: &ModelFetchConfig,
    reporter: Option<ModelFetchReporter<'_>>,
) -> Result<PathBuf, ModelFetchError> {
    let directory = config.bundle_dir(entry);
    let outstanding = outstanding_artifacts(&directory, entry);
    let manifest_current = manifest_is_current(&directory, entry);
    if outstanding.is_empty() && manifest_current {
        report(
            reporter,
            ModelFetchEvent::Cached {
                model: entry.model_id,
            },
        );
        return Ok(directory);
    }
    if !config.allow_download {
        return Err(ModelFetchError::DownloadDisabled {
            model: entry.model_id,
            directory,
        });
    }

    fs::create_dir_all(&directory).map_err(|error| ModelFetchError::Cache {
        directory: directory.clone(),
        message: error.to_string(),
    })?;
    let total_bytes = outstanding
        .iter()
        .map(|artifact| artifact.size_bytes)
        .sum::<u64>();
    report(
        reporter,
        ModelFetchEvent::Started {
            model: entry.model_id,
            license: entry.license,
            total_bytes,
        },
    );

    let client = Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .read_timeout(READ_TIMEOUT)
        .build()
        .map_err(|_| ModelFetchError::HttpClient)?;
    for artifact in outstanding {
        download_artifact(&client, entry, artifact, &directory, config, reporter).await?;
        report(
            reporter,
            ModelFetchEvent::Verified {
                path: artifact.path,
            },
        );
    }
    if !manifest_current {
        write_atomic(
            &directory,
            entry.manifest_file,
            entry.manifest_bytes.as_bytes(),
        )?;
    }
    report(
        reporter,
        ModelFetchEvent::Finished {
            model: entry.model_id,
        },
    );
    Ok(directory)
}

fn outstanding_artifacts(
    directory: &Path,
    entry: &'static LocalModelEntry,
) -> Vec<&'static LocalModelArtifact> {
    entry
        .artifacts
        .iter()
        .filter(|artifact| {
            fs::metadata(directory.join(artifact.path)).map_or(true, |metadata| {
                !metadata.is_file() || metadata.len() != artifact.size_bytes
            })
        })
        .collect()
}

fn manifest_is_current(directory: &Path, entry: &LocalModelEntry) -> bool {
    fs::read(directory.join(entry.manifest_file))
        .is_ok_and(|bytes| bytes == entry.manifest_bytes.as_bytes())
}

async fn download_artifact(
    client: &Client,
    entry: &'static LocalModelEntry,
    artifact: &'static LocalModelArtifact,
    directory: &Path,
    config: &ModelFetchConfig,
    reporter: Option<ModelFetchReporter<'_>>,
) -> Result<(), ModelFetchError> {
    let url = artifact_url(&config.base_url, entry, artifact)?;
    let mut response =
        client
            .get(url)
            .send()
            .await
            .map_err(|error| ModelFetchError::Transport {
                model: entry.model_id,
                path: artifact.path,
                message: error.to_string(),
            })?;
    let status = response.status();
    if !status.is_success() {
        return Err(ModelFetchError::Status {
            model: entry.model_id,
            path: artifact.path,
            status: status.as_u16(),
        });
    }
    if let Some(length) = response.content_length()
        && length != artifact.size_bytes
    {
        return Err(ModelFetchError::Length {
            model: entry.model_id,
            path: artifact.path,
            actual: length,
            expected: artifact.size_bytes,
        });
    }

    let temporary = directory.join(temporary_name(artifact.path));
    let verified = stream_verified(
        &mut response,
        entry,
        artifact,
        directory,
        &temporary,
        reporter,
    )
    .await;
    if verified.is_err() {
        remove_quietly(&temporary);
    }
    verified?;
    replace_atomic(&temporary, &directory.join(artifact.path)).map_err(|message| {
        remove_quietly(&temporary);
        ModelFetchError::Cache {
            directory: directory.to_path_buf(),
            message,
        }
    })
}

/// Streams one artifact into `temporary`, enforcing its pinned length and digest.
///
/// Verification happens before the caller commits the file, so a partial or
/// substituted transfer never appears under the artifact's real name.
async fn stream_verified(
    response: &mut reqwest::Response,
    entry: &'static LocalModelEntry,
    artifact: &'static LocalModelArtifact,
    directory: &Path,
    temporary: &Path,
    reporter: Option<ModelFetchReporter<'_>>,
) -> Result<(), ModelFetchError> {
    let cache_error = |message: String| ModelFetchError::Cache {
        directory: directory.to_path_buf(),
        message,
    };
    let length_error = |actual: u64| ModelFetchError::Length {
        model: entry.model_id,
        path: artifact.path,
        actual,
        expected: artifact.size_bytes,
    };
    let mut file = fs::File::create(temporary).map_err(|error| cache_error(error.to_string()))?;
    let mut hasher = Sha256::new();
    let mut written = 0_u64;
    let mut announced = 0_u64;
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| ModelFetchError::Transport {
            model: entry.model_id,
            path: artifact.path,
            message: error.to_string(),
        })?
    {
        written = written.saturating_add(chunk.len() as u64);
        if written > artifact.size_bytes {
            return Err(length_error(written));
        }
        hasher.update(&chunk);
        file.write_all(&chunk)
            .map_err(|error| cache_error(error.to_string()))?;
        if written - announced >= PROGRESS_INTERVAL_BYTES {
            announced = written;
            report(
                reporter,
                ModelFetchEvent::Progress {
                    path: artifact.path,
                    downloaded: written,
                    total_bytes: artifact.size_bytes,
                },
            );
        }
    }
    file.sync_all()
        .map_err(|error| cache_error(error.to_string()))?;
    drop(file);
    if written != artifact.size_bytes {
        return Err(length_error(written));
    }
    let actual = format!("{:x}", hasher.finalize());
    if actual != artifact.sha256 {
        return Err(ModelFetchError::Checksum {
            model: entry.model_id,
            path: artifact.path,
            expected: artifact.sha256,
            actual,
        });
    }
    Ok(())
}

fn artifact_url(
    base_url: &str,
    entry: &LocalModelEntry,
    artifact: &LocalModelArtifact,
) -> Result<Url, ModelFetchError> {
    let mut url =
        Url::parse(base_url).map_err(|_| ModelFetchError::InvalidBaseUrl(base_url.to_owned()))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(ModelFetchError::InvalidBaseUrl(base_url.to_owned()));
    }
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|()| ModelFetchError::InvalidBaseUrl(base_url.to_owned()))?;
        segments.pop_if_empty();
        for segment in entry.model_id.split('/') {
            segments.push(segment);
        }
        segments.push("resolve");
        segments.push(entry.revision);
        segments.push(artifact.path);
    }
    Ok(url)
}

fn write_atomic(directory: &Path, name: &str, bytes: &[u8]) -> Result<(), ModelFetchError> {
    let temporary = directory.join(temporary_name(name));
    let cache_error = |message: String| ModelFetchError::Cache {
        directory: directory.to_path_buf(),
        message,
    };
    fs::write(&temporary, bytes).map_err(|error| cache_error(error.to_string()))?;
    replace_atomic(&temporary, &directory.join(name)).map_err(|message| {
        remove_quietly(&temporary);
        cache_error(message)
    })
}

/// Commits a verified temporary file over its final name.
///
/// Two processes may provision the same bundle concurrently. Both write
/// uniquely named temporaries and both verify before committing, so whichever
/// rename lands last still leaves byte-identical content in place.
fn replace_atomic(from: &Path, to: &Path) -> Result<(), String> {
    match fs::rename(from, to) {
        Ok(()) => Ok(()),
        Err(error) if to.exists() => {
            fs::remove_file(to).map_err(|error| error.to_string())?;
            fs::rename(from, to).map_err(|_| error.to_string())
        }
        Err(error) => Err(error.to_string()),
    }
}

fn temporary_name(name: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.subsec_nanos())
        .unwrap_or_default();
    format!(".{name}.partial-{}-{nanos}", std::process::id())
}

fn remove_quietly(path: &Path) {
    drop(fs::remove_file(path));
}

fn report(reporter: Option<ModelFetchReporter<'_>>, event: ModelFetchEvent) {
    if let Some(reporter) = reporter {
        reporter(event);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_MODEL_BASE_URL, ModelFetchConfig, ModelFetchError, artifact_url,
        manifest_is_current, outstanding_artifacts, replace_atomic,
    };
    use crate::model_catalog::POTION_CODE_16M_V2;

    #[test]
    fn artifact_urls_resolve_against_upstream_and_mirrors() {
        let artifact = &POTION_CODE_16M_V2.artifacts[0];
        let upstream = artifact_url(DEFAULT_MODEL_BASE_URL, &POTION_CODE_16M_V2, artifact).unwrap();
        assert_eq!(
            upstream.as_str(),
            "https://huggingface.co/minishlab/potion-code-16M-v2/resolve/e9d2a44ca6a05ac6685f3b23709ea57eb7352d5b/model.safetensors"
        );
        let mirror = artifact_url(
            "http://mirror.internal:8080/hf/",
            &POTION_CODE_16M_V2,
            artifact,
        )
        .unwrap();
        assert_eq!(
            mirror.as_str(),
            "http://mirror.internal:8080/hf/minishlab/potion-code-16M-v2/resolve/e9d2a44ca6a05ac6685f3b23709ea57eb7352d5b/model.safetensors"
        );
    }

    #[test]
    fn non_http_base_urls_are_rejected() {
        let artifact = &POTION_CODE_16M_V2.artifacts[0];
        assert!(matches!(
            artifact_url("file:///models", &POTION_CODE_16M_V2, artifact),
            Err(ModelFetchError::InvalidBaseUrl(_))
        ));
        assert!(matches!(
            artifact_url("not a url", &POTION_CODE_16M_V2, artifact),
            Err(ModelFetchError::InvalidBaseUrl(_))
        ));
    }

    #[test]
    fn bundle_directory_is_scoped_by_key_and_revision() {
        let config = ModelFetchConfig::new("/cache");
        assert!(
            config
                .bundle_dir(&POTION_CODE_16M_V2)
                .ends_with("potion-code-16m-v2/e9d2a44ca6a05ac6685f3b23709ea57eb7352d5b")
        );
    }

    #[test]
    fn wrong_length_files_are_treated_as_outstanding() {
        let directory = tempfile::tempdir().unwrap();
        assert_eq!(
            outstanding_artifacts(directory.path(), &POTION_CODE_16M_V2).len(),
            POTION_CODE_16M_V2.artifacts.len()
        );
        std::fs::write(directory.path().join("config.json"), b"{}").unwrap();
        assert_eq!(
            outstanding_artifacts(directory.path(), &POTION_CODE_16M_V2).len(),
            POTION_CODE_16M_V2.artifacts.len(),
            "a truncated artifact must not count as provisioned"
        );
    }

    #[test]
    fn manifest_must_match_the_catalog_byte_for_byte() {
        let directory = tempfile::tempdir().unwrap();
        assert!(!manifest_is_current(directory.path(), &POTION_CODE_16M_V2));
        std::fs::write(
            directory.path().join(POTION_CODE_16M_V2.manifest_file),
            POTION_CODE_16M_V2.manifest_bytes,
        )
        .unwrap();
        assert!(manifest_is_current(directory.path(), &POTION_CODE_16M_V2));
        std::fs::write(
            directory.path().join(POTION_CODE_16M_V2.manifest_file),
            format!("{} ", POTION_CODE_16M_V2.manifest_bytes),
        )
        .unwrap();
        assert!(
            !manifest_is_current(directory.path(), &POTION_CODE_16M_V2),
            "a rewritten manifest changes the index identity and must be replaced"
        );
    }

    #[test]
    fn commit_overwrites_an_existing_file() {
        let directory = tempfile::tempdir().unwrap();
        let from = directory.path().join("staged");
        let to = directory.path().join("final");
        std::fs::write(&to, b"old").unwrap();
        std::fs::write(&from, b"new").unwrap();
        replace_atomic(&from, &to).unwrap();
        assert_eq!(std::fs::read(&to).unwrap(), b"new");
        assert!(!from.exists());
    }
}
