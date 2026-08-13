//! Runtime-agnostic contracts for repository-scale source indexing.
//!
//! This crate owns domain types and object-safe interfaces. It intentionally
//! does not select an async runtime, Git implementation, embedding provider, or
//! storage backend.
//!
//! ```
//! use cast_index::{NormalizedPath, RepositoryId};
//!
//! let repository = RepositoryId::new("github", "acme/search", "engine")?;
//! let path = NormalizedPath::new(r"src\lib.rs")?;
//! assert_eq!(repository.to_string(), "github:acme/search/engine");
//! assert_eq!(path.as_str(), "src/lib.rs");
//! # Ok::<(), cast_index::ContractError>(())
//! ```

#![deny(missing_docs)]

mod document;
mod error;
mod identity;
mod pipeline;
mod source;
mod traits;

pub use document::{
    DocumentIdentityInput, EmbeddingIdentity, EmbeddingVector, IndexDocument, IndexFingerprint,
    SizerIdentity,
};
pub use error::{ContractError, IndexError, IndexErrorKind, RetryAdvice};
pub use identity::{
    BranchName, ContentHash, DocumentId, HashAlgorithm, NormalizedPath, RepositoryId, RevisionId,
    UnixMillis,
};
pub use pipeline::{
    FileLookup, IndexConfig, IndexEvent, IndexMode, IndexPhase, IndexStatus, IndexSummary,
    StoreCapabilities, SyncCheckpoint, WriteBatch, WriteOperation, WriteReceipt,
};
pub use source::{
    Eligibility, FileDescriptor, FileInventoryRequest, FileStatus, RepositorySnapshot, SkipReason,
    SourceFile,
};
pub use traits::{
    BoxFuture, Cancellation, ChunkEngine, ChunkEngineFactory, DocumentIdFactory, Embedder,
    EmbeddingInput, FileCursor, IndexObserver, IndexStore, RepositorySource, RerankIdentity,
    RerankRequest, RerankScores, Reranker,
};

/// Schema version for [`IndexDocument`].
pub const INDEX_DOCUMENT_SCHEMA_VERSION: u32 = 1;
