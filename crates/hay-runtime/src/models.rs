//! Resolves local model bundles before a runtime is constructed.
//!
//! Loading stays offline and checksum-verified. This module only decides which
//! directory the `local-static` provider opens, and provisions that directory
//! from the pinned catalog when it is missing. A caller-provisioned bundle
//! always wins, so an air-gapped deployment keeps its existing behavior and
//! never consults the network.

use std::env;
use std::path::PathBuf;

use anyhow::{Context as _, Result, bail};
use cast_embeddings::model_catalog::default_static_model;
use cast_embeddings::{ModelFetchConfig, ModelFetchError, ModelFetchEvent, ModelFetchReporter};

use crate::EmbeddingProvider;

/// Turns automatic model provisioning on or off.
pub const DOWNLOAD_MODELS_ENV: &str = "COTH_HAY_SEEKER_DOWNLOAD_MODELS";
/// Overrides the directory provisioned bundles are cached in.
pub const MODEL_CACHE_DIR_ENV: &str = "COTH_HAY_SEEKER_MODEL_CACHE_DIR";
/// Overrides the host artifacts are downloaded from.
pub const MODEL_BASE_URL_ENV: &str = "COTH_HAY_SEEKER_MODEL_BASE_URL";
/// Points `local-static` at a bundle staged by hand.
pub const LOCAL_STATIC_DIR_ENV: &str = "HAY_LOCAL_STATIC_MODEL_DIR";

const CACHE_NAMESPACE: &str = "hay-seeker";
const MODEL_SUBDIRECTORY: &str = "models";

/// Bundle directories a runtime should open for local providers.
#[derive(Clone, Debug, Default)]
pub struct ResolvedModels {
    /// Verified bundle directory for the `local-static` provider.
    pub local_static_dir: Option<PathBuf>,
}

/// Resolves and, when permitted, provisions every bundle `provider` needs.
///
/// Providers that do not read local artifacts resolve to an empty result, so
/// this is safe to call unconditionally before building a runtime.
///
/// # Errors
///
/// Returns an error when a bundle is missing and downloads are disabled, when
/// the cache directory cannot be located or written, or when a downloaded
/// artifact fails its pinned length or digest.
pub async fn ensure_models(
    provider: EmbeddingProvider,
    allow_download: bool,
    reporter: Option<ModelFetchReporter<'_>>,
) -> Result<ResolvedModels> {
    let mut resolved = ResolvedModels::default();
    if provider != EmbeddingProvider::LocalStatic {
        return Ok(resolved);
    }
    if let Some(staged) = configured_path(LOCAL_STATIC_DIR_ENV) {
        resolved.local_static_dir = Some(staged);
        return Ok(resolved);
    }

    let entry = default_static_model();
    let config = ModelFetchConfig::new(model_cache_dir()?)
        .with_base_url(
            configured_value(MODEL_BASE_URL_ENV)
                .unwrap_or_else(|| cast_embeddings::DEFAULT_MODEL_BASE_URL.to_owned()),
        )
        .with_allow_download(allow_download);
    let directory = cast_embeddings::ensure_bundle(entry, &config, reporter)
        .await
        .map_err(|error| provisioning_error(&error))?;
    resolved.local_static_dir = Some(directory);
    Ok(resolved)
}

/// Writes provisioning progress to standard error.
///
/// Standard output carries search results and the MCP protocol, so progress
/// never goes there.
pub fn report_to_stderr(event: ModelFetchEvent) {
    match event {
        ModelFetchEvent::Cached { .. } | ModelFetchEvent::Verified { .. } => {}
        ModelFetchEvent::Started {
            model,
            license,
            total_bytes,
        } => {
            eprintln!(
                "hay: downloading {model} ({license}, {}) once; \
                 disable with {DOWNLOAD_MODELS_ENV}=false",
                human_bytes(total_bytes)
            );
        }
        ModelFetchEvent::Progress {
            path,
            downloaded,
            total_bytes,
        } => {
            eprintln!(
                "hay: {path} {} / {}",
                human_bytes(downloaded),
                human_bytes(total_bytes)
            );
        }
        ModelFetchEvent::Finished { model } => {
            eprintln!("hay: {model} is provisioned and checksum-verified");
        }
    }
}

/// Directory provisioned bundles are cached in.
///
/// # Errors
///
/// Returns an error when no per-user cache location can be determined.
pub fn model_cache_dir() -> Result<PathBuf> {
    if let Some(explicit) = configured_path(MODEL_CACHE_DIR_ENV) {
        return Ok(explicit);
    }
    Ok(platform_cache_root()?
        .join(CACHE_NAMESPACE)
        .join(MODEL_SUBDIRECTORY))
}

fn platform_cache_root() -> Result<PathBuf> {
    if cfg!(target_os = "windows") {
        if let Some(local) = configured_path("LOCALAPPDATA") {
            return Ok(local);
        }
        if let Some(profile) = configured_path("USERPROFILE") {
            return Ok(profile.join("AppData").join("Local"));
        }
    } else if cfg!(target_os = "macos") {
        if let Some(home) = configured_path("HOME") {
            return Ok(home.join("Library").join("Caches"));
        }
    } else {
        if let Some(cache) = configured_path("XDG_CACHE_HOME") {
            return Ok(cache);
        }
        if let Some(home) = configured_path("HOME") {
            return Ok(home.join(".cache"));
        }
    }
    bail!(
        "could not determine a model cache directory; set {MODEL_CACHE_DIR_ENV} to an absolute path"
    )
}

fn provisioning_error(error: &ModelFetchError) -> anyhow::Error {
    let remedy = match error {
        ModelFetchError::DownloadDisabled { .. } => format!(
            "set {DOWNLOAD_MODELS_ENV}=true to provision it automatically, \
             or stage the bundle by hand and point {LOCAL_STATIC_DIR_ENV} at it"
        ),
        ModelFetchError::Checksum { .. } | ModelFetchError::Length { .. } => format!(
            "the bytes served did not match the pinned build; \
             retry, or set {MODEL_BASE_URL_ENV} to a trusted mirror"
        ),
        ModelFetchError::Cache { directory, .. } => format!(
            "check permissions on {}, or set {MODEL_CACHE_DIR_ENV} to a writable path",
            directory.display()
        ),
        _ => format!(
            "retry, or set {DOWNLOAD_MODELS_ENV}=false and stage the bundle by hand \
             at {LOCAL_STATIC_DIR_ENV}"
        ),
    };
    anyhow::anyhow!("{error}; {remedy}")
}

fn configured_value(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn configured_path(name: &str) -> Option<PathBuf> {
    configured_value(name).map(PathBuf::from)
}

fn human_bytes(bytes: u64) -> String {
    #[expect(
        clippy::cast_precision_loss,
        reason = "display only; artifact sizes are far below the f64 integer limit"
    )]
    let mebibytes = bytes as f64 / (1024.0 * 1024.0);
    if mebibytes < 1.0 {
        return format!("{bytes} B");
    }
    format!("{mebibytes:.1} MiB")
}

/// Reads the pinned static bundle directory from an explicit setting only.
///
/// # Errors
///
/// Returns an error when neither a resolved bundle nor an explicit directory is
/// available.
pub(crate) fn static_bundle_dir(resolved: &ResolvedModels) -> Result<PathBuf> {
    if let Some(directory) = resolved.local_static_dir.clone() {
        return Ok(directory);
    }
    configured_path(LOCAL_STATIC_DIR_ENV).with_context(|| {
        format!(
            "no local static model bundle is available: allow automatic provisioning with \
             {DOWNLOAD_MODELS_ENV}=true, or point {LOCAL_STATIC_DIR_ENV} at a staged bundle"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::{ResolvedModels, human_bytes, static_bundle_dir};

    #[test]
    fn a_resolved_bundle_is_preferred_over_the_environment() {
        let resolved = ResolvedModels {
            local_static_dir: Some("/provisioned".into()),
        };
        assert_eq!(
            static_bundle_dir(&resolved).unwrap(),
            std::path::PathBuf::from("/provisioned")
        );
    }

    #[test]
    fn sizes_are_reported_in_readable_units() {
        assert_eq!(human_bytes(59), "59 B");
        assert_eq!(human_bytes(33_514_471), "32.0 MiB");
    }
}
