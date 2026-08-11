use cast_core::LanguageId;
use serde::{Deserialize, Serialize};

use crate::{
    BranchName, ContentHash, ContractError, IndexMode, NormalizedPath, RepositoryId, RevisionId,
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Immutable repository, branch, and revision selected for one run.
pub struct RepositorySnapshot {
    /// Globally scoped repository identity.
    pub repository: RepositoryId,
    /// Branch being indexed.
    pub branch: BranchName,
    /// Resolved revision read by the run.
    pub revision: RevisionId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Request used to open a pull-based repository inventory.
pub struct FileInventoryRequest {
    /// Repository state to inventory.
    pub snapshot: RepositorySnapshot,
    /// Requested full or incremental inventory mode.
    pub mode: IndexMode,
    /// Previous completed revision for incremental discovery.
    pub since: Option<RevisionId>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Change state assigned to a discovered repository path.
pub enum FileStatus {
    /// Path is new since the prior checkpoint.
    Added,
    /// Existing path has changed content or metadata.
    Modified,
    /// Path existed at the prior checkpoint and is now absent.
    Deleted,
    /// Path and content are unchanged.
    Unchanged,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Metadata discovered before loading complete file content.
pub struct FileDescriptor {
    /// Normalized repository-relative path.
    pub path: NormalizedPath,
    /// Change state relative to the requested base revision.
    pub status: FileStatus,
    /// Source size in bytes.
    pub byte_len: u64,
    /// Optional language hint from path or repository metadata.
    pub language_hint: Option<LanguageId>,
    /// Optional precomputed complete-content hash.
    pub content_hash: Option<ContentHash>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// UTF-8 file content paired with its discovery descriptor.
pub struct SourceFile {
    /// Metadata supplied by repository discovery.
    pub descriptor: FileDescriptor,
    /// Complete UTF-8 source content.
    pub content: String,
}

impl SourceFile {
    /// Validates descriptor/content agreement.
    ///
    /// # Errors
    ///
    /// Returns [`ContractError`] for deleted files with content or a byte-size
    /// mismatch.
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.descriptor.status == FileStatus::Deleted {
            return Err(ContractError::SourceInvariant(
                "deleted files must be represented by a delete operation, not SourceFile".into(),
            ));
        }
        if self.descriptor.byte_len != self.content.len() as u64 {
            return Err(ContractError::SourceInvariant(format!(
                "descriptor reports {} bytes but content contains {}",
                self.descriptor.byte_len,
                self.content.len()
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "reason")]
/// Stable reason a discovered file was excluded from indexing.
pub enum SkipReason {
    /// A path component is hidden by repository policy.
    HiddenPath,
    /// Ignore rules excluded the path.
    Ignored,
    /// No supported language or generic policy matched.
    UnsupportedLanguage,
    /// The binary probe detected non-text content.
    Binary,
    /// Source bytes are not valid UTF-8.
    InvalidUtf8,
    /// A configuration or data-like file exceeded its lower size limit.
    ConfigTooLarge,
    /// A source-code file exceeded the configured size limit.
    SourceTooLarge,
    /// Content heuristics classified the file as generated or data-like.
    DataLike,
    /// Loading the file failed recoverably.
    ReadFailed {
        /// Stable source-adapter failure code.
        code: String,
    },
    /// A caller-defined policy excluded the file.
    Policy {
        /// Stable caller-defined policy code.
        code: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "decision")]
/// Result of applying indexing eligibility policy to one descriptor.
pub enum Eligibility {
    /// The file should be loaded and indexed.
    Include {
        /// Accepted descriptor, optionally enriched by policy evaluation.
        descriptor: FileDescriptor,
    },
    /// The file should not be loaded or indexed.
    Skip {
        /// Normalized excluded path.
        path: NormalizedPath,
        /// Stable exclusion reason.
        reason: SkipReason,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_file_size_must_match_utf8_bytes() {
        let source = SourceFile {
            descriptor: FileDescriptor {
                path: NormalizedPath::new("src/lib.rs").unwrap(),
                status: FileStatus::Added,
                byte_len: 1,
                language_hint: Some(LanguageId::from("rust")),
                content_hash: None,
            },
            content: "🙂".into(),
        };

        assert!(source.validate().is_err());
    }
}
