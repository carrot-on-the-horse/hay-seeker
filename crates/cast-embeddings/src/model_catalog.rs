//! Pinned catalog of locally runnable embedding models.
//!
//! Every entry names an immutable upstream revision, the exact artifacts that
//! revision publishes, and the lowercase SHA-256 of each artifact. Nothing in
//! this catalog is discovered at runtime: an entry is the only description of a
//! model Hay will accept, so a provisioning path can fetch bytes from an
//! untrusted mirror and still reject anything that is not the pinned build.
//!
//! Adding a model for a different hardware profile is a data change here plus
//! the provider contract that consumes it, not new download code.

/// Runtime layout an entry's artifacts are provisioned for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LocalModelKind {
    /// A static token-embedding table loaded by `local-static`.
    Static,
}

/// One pinned artifact inside a catalog entry.
#[derive(Clone, Copy, Debug)]
pub struct LocalModelArtifact {
    /// Bundle-relative file name.
    pub path: &'static str,
    /// Lowercase SHA-256 of the published bytes.
    pub sha256: &'static str,
    /// Published byte length, used for cheap presence checks and size caps.
    pub size_bytes: u64,
}

/// A locally runnable model pinned to one upstream revision.
#[derive(Clone, Copy, Debug)]
pub struct LocalModelEntry {
    /// Stable directory-safe catalog key.
    pub key: &'static str,
    /// Provider layout the artifacts are provisioned for.
    pub kind: LocalModelKind,
    /// Upstream repository identifier.
    pub model_id: &'static str,
    /// Immutable upstream revision.
    pub revision: &'static str,
    /// Native trained output width.
    pub dimensions: usize,
    /// Exact inference profile recorded in the index identity.
    pub embedding_profile: &'static str,
    /// Bundle manifest file name written next to the artifacts.
    pub manifest_file: &'static str,
    /// Canonical manifest bytes.
    ///
    /// These bytes are hashed into the persisted index identity, so they are
    /// pinned verbatim rather than re-serialized. A provisioned bundle is
    /// byte-identical to a hand-built one and produces the same index.
    pub manifest_bytes: &'static str,
    /// Artifacts the bundle requires, in download order.
    pub artifacts: &'static [LocalModelArtifact],
    /// Upstream license identifier, surfaced when a model is provisioned.
    pub license: &'static str,
}

impl LocalModelEntry {
    /// Total bytes a cold provisioning run downloads.
    #[must_use]
    pub fn download_bytes(&self) -> u64 {
        self.artifacts
            .iter()
            .map(|artifact| artifact.size_bytes)
            .sum()
    }

    /// Looks up a pinned artifact by its bundle-relative path.
    #[must_use]
    pub fn artifact(&self, path: &str) -> Option<&LocalModelArtifact> {
        self.artifacts.iter().find(|artifact| artifact.path == path)
    }
}

/// Canonical `static-bundle.json` for [`POTION_CODE_16M_V2`].
///
/// Kept byte-identical to `models/potion-code-16m-v2/static-bundle.example.json`
/// so a downloaded bundle and a hand-provisioned one share an index identity.
const POTION_CODE_16M_V2_MANIFEST: &str = r#"{
  "schema_version": 1,
  "model_id": "minishlab/potion-code-16M-v2",
  "model_revision": "e9d2a44ca6a05ac6685f3b23709ea57eb7352d5b",
  "dimensions": 256,
  "embedding_profile": "potion-code-16m-v2-nospecial-drop-unk-max16384-mean-l2-v1",
  "unknown_token_id": 1,
  "max_tokens": 16384,
  "artifacts": [
    {
      "path": "model.safetensors",
      "sha256": "75cf7a6c2171b230ad19b1e7d8e0b1aee86da5a02af8e7cacedd9921d227623c"
    },
    {
      "path": "tokenizer.json",
      "sha256": "107bbdcbad4bff1d299b7a4c3a2fb17c52890688b7dd0e4c9deab79d3c4f3d45"
    },
    {
      "path": "config.json",
      "sha256": "148e5691a6fcc553437156859701fba017a1ba5d340b170f17e0f3668fb861a7"
    }
  ]
}
"#;

/// Code-trained 256-dimensional static table used by `local-static`.
///
/// MIT throughout its lineage: the published checkpoint is MIT, it was
/// distilled from the MIT-licensed `nomic-ai/CodeRankEmbed`, which is built on
/// Apache-2.0 `Snowflake/snowflake-arctic-embed-m-long`. Redistribution and
/// automated provisioning are permitted with attribution.
pub const POTION_CODE_16M_V2: LocalModelEntry = LocalModelEntry {
    key: "potion-code-16m-v2",
    kind: LocalModelKind::Static,
    model_id: "minishlab/potion-code-16M-v2",
    revision: "e9d2a44ca6a05ac6685f3b23709ea57eb7352d5b",
    dimensions: 256,
    embedding_profile: crate::POTION_CODE_16M_V2_PROFILE,
    manifest_file: "static-bundle.json",
    manifest_bytes: POTION_CODE_16M_V2_MANIFEST,
    artifacts: &[
        LocalModelArtifact {
            path: "model.safetensors",
            sha256: "75cf7a6c2171b230ad19b1e7d8e0b1aee86da5a02af8e7cacedd9921d227623c",
            size_bytes: 32_490_072,
        },
        LocalModelArtifact {
            path: "tokenizer.json",
            sha256: "107bbdcbad4bff1d299b7a4c3a2fb17c52890688b7dd0e4c9deab79d3c4f3d45",
            size_bytes: 1_024_340,
        },
        LocalModelArtifact {
            path: "config.json",
            sha256: "148e5691a6fcc553437156859701fba017a1ba5d340b170f17e0f3668fb861a7",
            size_bytes: 59,
        },
    ],
    license: "MIT",
};

/// Every model Hay can provision automatically.
pub const CATALOG: &[LocalModelEntry] = &[POTION_CODE_16M_V2];

/// Entry backing the `local-static` provider when no bundle directory is set.
#[must_use]
pub const fn default_static_model() -> &'static LocalModelEntry {
    &POTION_CODE_16M_V2
}

/// Looks up a catalog entry by its stable key.
#[must_use]
pub fn find(key: &str) -> Option<&'static LocalModelEntry> {
    CATALOG.iter().find(|entry| entry.key == key)
}

#[cfg(test)]
mod tests {
    use super::{CATALOG, POTION_CODE_16M_V2, default_static_model, find};
    use sha2::{Digest, Sha256};

    #[test]
    fn manifest_pins_every_catalog_artifact() {
        for entry in CATALOG {
            for artifact in entry.artifacts {
                assert!(
                    entry.manifest_bytes.contains(artifact.sha256),
                    "{} manifest omits the pinned digest for {}",
                    entry.key,
                    artifact.path
                );
                assert!(
                    entry.manifest_bytes.contains(artifact.path),
                    "{} manifest omits {}",
                    entry.key,
                    artifact.path
                );
            }
            assert!(
                entry.manifest_bytes.contains(entry.revision),
                "{} manifest omits its pinned revision",
                entry.key
            );
        }
    }

    #[test]
    fn catalog_digests_are_lowercase_sha256() {
        for entry in CATALOG {
            for artifact in entry.artifacts {
                assert_eq!(artifact.sha256.len(), 64, "{} digest length", artifact.path);
                assert!(
                    artifact
                        .sha256
                        .bytes()
                        .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
                    "{} digest must be lowercase hex",
                    artifact.path
                );
            }
        }
    }

    #[test]
    fn catalog_keys_are_unique_and_directory_safe() {
        for (index, entry) in CATALOG.iter().enumerate() {
            assert!(
                entry
                    .key
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-'),
                "{} is not a directory-safe key",
                entry.key
            );
            assert!(
                CATALOG[..index].iter().all(|other| other.key != entry.key),
                "duplicate catalog key {}",
                entry.key
            );
        }
    }

    /// The manifest hash is part of the persisted index identity, so a change
    /// to these bytes silently invalidates every existing static index.
    #[test]
    fn potion_manifest_bytes_are_frozen() {
        let digest = format!(
            "{:x}",
            Sha256::digest(POTION_CODE_16M_V2.manifest_bytes.as_bytes())
        );
        assert_eq!(
            digest, "c8e2bbcb8518eb6ed1b48faa6b7c6ec9ef591364303515ec29cf9f7c1223bb94",
            "static bundle manifest bytes changed; this rewrites the index identity"
        );
    }

    #[test]
    fn lookup_resolves_the_default_static_model() {
        assert_eq!(
            find("potion-code-16m-v2").map(|entry| entry.key),
            Some("potion-code-16m-v2")
        );
        assert!(find("missing").is_none());
        assert_eq!(default_static_model().dimensions, 256);
        assert_eq!(default_static_model().download_bytes(), 33_514_471);
    }
}
