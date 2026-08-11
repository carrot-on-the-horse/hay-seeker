use std::num::{NonZeroU64, NonZeroUsize};

use serde::{Deserialize, Serialize};

use crate::{
    BranchName, ContentHash, ContractError, DocumentId, IndexDocument, IndexError,
    IndexFingerprint, NormalizedPath, RepositoryId, RevisionId, UnixMillis,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Strategy used to discover and reconcile repository files.
pub enum IndexMode {
    /// Select full or incremental indexing from the stored checkpoint.
    #[default]
    Auto,
    /// Re-read all eligible files and replace prior branch state.
    Full,
    /// Process changes since the compatible stored checkpoint.
    Incremental,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Runtime-independent limits and concurrency controls for indexing.
pub struct IndexConfig {
    /// Requested reconciliation strategy.
    pub mode: IndexMode,
    /// Maximum concurrent file-processing workers.
    pub workers: NonZeroUsize,
    /// Maximum queued files awaiting processing.
    pub queue_capacity: NonZeroUsize,
    /// Target number of documents per storage write batch.
    pub write_batch_documents: NonZeroUsize,
    /// Target number of documents sent to an embedder per request.
    pub embedding_batch_documents: NonZeroUsize,
    /// Maximum bytes accepted for ordinary source-code files.
    pub code_file_bytes: NonZeroU64,
    /// Maximum bytes accepted for configuration and data-like files.
    pub config_file_bytes: NonZeroU64,
    /// Number of leading bytes inspected for binary detection.
    pub binary_probe_bytes: NonZeroUsize,
    /// Maximum processing time for one file.
    pub per_file_timeout_ms: NonZeroU64,
    /// Maximum time without pipeline progress before aborting.
    pub stall_timeout_ms: NonZeroU64,
}

impl Default for IndexConfig {
    fn default() -> Self {
        let workers = std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);
        Self {
            mode: IndexMode::Auto,
            workers,
            queue_capacity: NonZeroUsize::new(workers.get().saturating_mul(2))
                .unwrap_or(NonZeroUsize::MIN),
            write_batch_documents: NonZeroUsize::new(100).unwrap_or(NonZeroUsize::MIN),
            embedding_batch_documents: NonZeroUsize::new(100).unwrap_or(NonZeroUsize::MIN),
            code_file_bytes: NonZeroU64::new(5 * 1024 * 1024).unwrap_or(NonZeroU64::MIN),
            config_file_bytes: NonZeroU64::new(50 * 1024).unwrap_or(NonZeroU64::MIN),
            binary_probe_bytes: NonZeroUsize::new(8 * 1024).unwrap_or(NonZeroUsize::MIN),
            per_file_timeout_ms: NonZeroU64::new(60_000).unwrap_or(NonZeroU64::MIN),
            stall_timeout_ms: NonZeroU64::new(10 * 60_000).unwrap_or(NonZeroU64::MIN),
        }
    }
}

impl IndexConfig {
    /// Validates cross-field bounds.
    ///
    /// # Errors
    ///
    /// Returns [`ContractError`] when queue capacity is smaller than the worker
    /// count or the config-file limit exceeds the code-file limit.
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.queue_capacity.get() < self.workers.get() {
            return Err(ContractError::Invalid {
                field: "queue_capacity",
                value: self.queue_capacity.to_string(),
            });
        }
        if self.config_file_bytes > self.code_file_bytes {
            return Err(ContractError::Invalid {
                field: "config_file_bytes",
                value: self.config_file_bytes.to_string(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Observable phase of an indexing run.
pub enum IndexPhase {
    /// Discover repository files and revisions.
    Discover,
    /// Apply ignore, binary, language, and size policies.
    Filter,
    /// Read eligible source content.
    Read,
    /// Parse and split source content.
    Chunk,
    /// Create optional dense vectors.
    Embed,
    /// Apply file-atomic backend writes.
    Store,
    /// Make pending writes durable.
    Flush,
    /// Persist a completed-run checkpoint.
    Checkpoint,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Terminal status of an indexing run.
pub enum IndexStatus {
    /// All discovered files were processed without policy skips.
    Completed,
    /// The run succeeded but one or more files were intentionally skipped.
    CompletedWithSkips,
    /// Cooperative cancellation stopped the run.
    Cancelled,
    /// No progress occurred within the stall deadline.
    Stalled,
    /// An unrecoverable source, chunking, embedding, or storage error occurred.
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "event")]
/// Progress event emitted by the indexing orchestrator.
pub enum IndexEvent {
    /// The pipeline entered a new phase.
    PhaseStarted {
        /// Newly active phase.
        phase: IndexPhase,
    },
    /// Processing started for one file.
    FileStarted {
        /// Repository-relative source path.
        path: NormalizedPath,
    },
    /// A discovered file was intentionally skipped.
    FileSkipped {
        /// Repository-relative skipped path.
        path: NormalizedPath,
        /// Stable policy or failure code.
        reason_code: String,
    },
    /// Processing completed for one file.
    FileCompleted {
        /// Repository-relative completed path.
        path: NormalizedPath,
        /// Number of produced chunk documents.
        chunks: usize,
    },
    /// A recoverable indexing failure occurred.
    Warning {
        /// Structured warning details.
        error: IndexError,
    },
    /// Aggregate counters changed.
    Progress {
        /// Files discovered so far.
        discovered: u64,
        /// Eligible files fully processed so far.
        processed: u64,
        /// Files intentionally skipped so far.
        skipped: u64,
        /// Files deleted from branch state so far.
        deleted: u64,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Aggregate counters and timings returned by an indexing run.
pub struct IndexSummary {
    /// Terminal run status.
    pub status: IndexStatus,
    /// Effective reconciliation mode.
    pub mode: IndexMode,
    /// Repositories processed by the run.
    pub repositories: u64,
    /// Files discovered before filtering.
    pub discovered: u64,
    /// Files accepted by indexing policy.
    pub eligible: u64,
    /// Eligible files successfully processed.
    pub processed: u64,
    /// Files rejected or skipped by policy.
    pub skipped: u64,
    /// File branch associations removed from the index.
    pub deleted: u64,
    /// Chunk documents successfully written.
    pub chunks_written: u64,
    /// Source bytes successfully processed.
    pub source_bytes: u64,
    /// Time spent discovering files.
    pub discover_ms: u64,
    /// Time spent parsing and chunking files.
    pub chunk_ms: u64,
    /// Time spent generating embeddings.
    pub embed_ms: u64,
    /// Time spent applying storage writes.
    pub store_ms: u64,
    /// End-to-end elapsed run time.
    pub total_ms: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Durable marker for a fully successful repository revision sync.
pub struct SyncCheckpoint {
    /// Repository synchronized by the run.
    pub repository: RepositoryId,
    /// Branch synchronized by the run.
    pub branch: BranchName,
    /// Source revision fully reflected in storage.
    pub revision: RevisionId,
    /// Chunking and embedding identity used by the run.
    pub fingerprint: IndexFingerprint,
    /// Wall-clock completion time.
    pub completed_at: UnixMillis,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Identity used to find reusable documents for unchanged file content.
pub struct FileLookup {
    /// Repository owning the source file.
    pub repository: RepositoryId,
    /// Normalized repository-relative path.
    pub path: NormalizedPath,
    /// Hash of the complete current source content.
    pub content_hash: ContentHash,
    /// Required chunking and embedding identity.
    pub fingerprint: IndexFingerprint,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
/// Optional semantics implemented by an index storage adapter.
pub struct StoreCapabilities {
    /// Identical content can share documents between branches.
    pub cross_branch_deduplication: bool,
    /// Dense vector persistence and retrieval is supported.
    pub vectors: bool,
    /// All operations for one file can be committed atomically.
    pub transactional_file_batches: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "operation")]
/// Backend-neutral mutation in a file-atomic write batch.
pub enum WriteOperation {
    /// Insert or replace one chunk document.
    Upsert {
        /// Complete validated index document.
        document: Box<IndexDocument>,
    },
    /// Remove one file from a branch without deleting shared branch documents.
    DeleteFile {
        /// Repository owning the source file.
        repository: RepositoryId,
        /// Normalized repository-relative path.
        path: NormalizedPath,
        /// Branch from which the file disappeared.
        branch: BranchName,
    },
    /// Attach an existing set of content-identical documents to another branch.
    AttachBranch {
        /// Repository owning the source file.
        repository: RepositoryId,
        /// Normalized repository-relative path.
        path: NormalizedPath,
        /// Hash used to verify the reused source content.
        content_hash: ContentHash,
        /// Branch receiving the existing documents.
        branch: BranchName,
        /// Deterministic documents to attach.
        document_ids: Vec<DocumentId>,
    },
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
/// Ordered mutations for one repository file.
pub struct WriteBatch {
    /// A batch must contain operations for only one source file. This is the
    /// unit that stores should make atomic when supported.
    pub operations: Vec<WriteOperation>,
}

impl WriteBatch {
    /// Validates the file-atomic batch contract and contained documents.
    ///
    /// # Errors
    ///
    /// Returns [`ContractError`] when the batch is empty, spans multiple files,
    /// contains an invalid document, or attaches a branch without document ids.
    pub fn validate(&self) -> Result<(), ContractError> {
        let Some(first) = self.operations.first() else {
            return Err(ContractError::DocumentInvariant(
                "write batch must not be empty".into(),
            ));
        };
        let (repository, path) = operation_target(first);

        for operation in &self.operations {
            let (candidate_repository, candidate_path) = operation_target(operation);
            if candidate_repository != repository || candidate_path != path {
                return Err(ContractError::DocumentInvariant(
                    "write batch operations must target one repository file".into(),
                ));
            }
            match operation {
                WriteOperation::Upsert { document } => document.validate()?,
                WriteOperation::AttachBranch { document_ids, .. } if document_ids.is_empty() => {
                    return Err(ContractError::DocumentInvariant(
                        "attach-branch operation requires document ids".into(),
                    ));
                }
                WriteOperation::DeleteFile { .. } | WriteOperation::AttachBranch { .. } => {}
            }
        }
        Ok(())
    }
}

fn operation_target(operation: &WriteOperation) -> (&RepositoryId, &NormalizedPath) {
    match operation {
        WriteOperation::Upsert { document } => (&document.repository, &document.path),
        WriteOperation::DeleteFile {
            repository, path, ..
        }
        | WriteOperation::AttachBranch {
            repository, path, ..
        } => (repository, path),
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Storage acknowledgement for one applied write batch.
pub struct WriteReceipt {
    /// Number of accepted operations from the submitted batch.
    pub accepted_operations: usize,
    /// Whether accepted operations were durable when the call returned.
    pub durable: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repository(name: &str) -> RepositoryId {
        RepositoryId::new("github", "example/team", name).unwrap()
    }

    #[test]
    fn defaults_preserve_proven_indexer_limits() {
        let config = IndexConfig::default();

        assert!(config.queue_capacity >= config.workers);
        assert_eq!(config.code_file_bytes.get(), 5 * 1024 * 1024);
        assert_eq!(config.config_file_bytes.get(), 50 * 1024);
        assert_eq!(config.binary_probe_bytes.get(), 8 * 1024);
        assert_eq!(config.per_file_timeout_ms.get(), 60_000);
        assert_eq!(config.stall_timeout_ms.get(), 10 * 60_000);
        config.validate().unwrap();
    }

    #[test]
    fn write_batch_must_target_one_file() {
        let branch = BranchName::new("main").unwrap();
        let batch = WriteBatch {
            operations: vec![
                WriteOperation::DeleteFile {
                    repository: repository("one"),
                    path: NormalizedPath::new("src/lib.rs").unwrap(),
                    branch: branch.clone(),
                },
                WriteOperation::DeleteFile {
                    repository: repository("two"),
                    path: NormalizedPath::new("src/lib.rs").unwrap(),
                    branch,
                },
            ],
        };

        assert!(batch.validate().is_err());
    }
}
