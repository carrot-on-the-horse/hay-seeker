use std::future::Future;
use std::pin::Pin;

use cast_core::ChunkOutput;

use crate::{
    ContractError, DocumentId, DocumentIdentityInput, EmbeddingIdentity, EmbeddingVector,
    FileDescriptor, FileInventoryRequest, FileLookup, IndexError, IndexEvent, RepositorySnapshot,
    SourceFile, StoreCapabilities, SyncCheckpoint, WriteBatch, WriteReceipt,
};

/// Runtime-neutral boxed future used by object-safe adapter traits.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Pull-based inventory cursor. Pulling one descriptor at a time allows the
/// orchestrator to apply bounded backpressure.
pub trait FileCursor: Send {
    /// Pulls the next descriptor, or `None` at the end of the inventory.
    fn next(&mut self) -> BoxFuture<'_, Result<Option<FileDescriptor>, IndexError>>;
}

/// Adapter that inventories and reads immutable repository snapshots.
pub trait RepositorySource: Send + Sync {
    /// Opens a bounded, pull-based file inventory.
    fn open_inventory<'a>(
        &'a self,
        request: &'a FileInventoryRequest,
    ) -> BoxFuture<'a, Result<Box<dyn FileCursor>, IndexError>>;

    /// Reads the complete UTF-8 content for a discovered file descriptor.
    fn read_file<'a>(
        &'a self,
        snapshot: &'a RepositorySnapshot,
        descriptor: &'a FileDescriptor,
    ) -> BoxFuture<'a, Result<SourceFile, IndexError>>;
}

/// Worker-local parser/chunker. Implementations may be mutable and are not
/// required to be `Sync`.
pub trait ChunkEngine: Send {
    /// # Errors
    ///
    /// Returns [`IndexError`] when language resolution, parsing, sizing, or
    /// chunk invariants fail.
    fn chunk(&mut self, file: &SourceFile) -> Result<ChunkOutput, IndexError>;
}

/// Factory for mutable worker-local chunk engines.
pub trait ChunkEngineFactory: Send + Sync {
    /// # Errors
    ///
    /// Returns [`IndexError`] when a worker-local engine cannot be initialized.
    fn create(&self) -> Result<Box<dyn ChunkEngine>, IndexError>;
}

/// Deterministic identity policy. Implementations must include every field in
/// [`DocumentIdentityInput`], preventing reuse across incompatible chunking
/// fingerprints.
pub trait DocumentIdFactory: Send + Sync {
    /// # Errors
    ///
    /// Returns [`IndexError`] when identity inputs cannot be canonicalized or
    /// hashed.
    fn create(&self, input: DocumentIdentityInput<'_>) -> Result<DocumentId, IndexError>;
}

#[derive(Clone, Copy, Debug)]
/// One ordered document input sent to an embedding provider.
pub struct EmbeddingInput<'a> {
    /// Deterministic document identity used for request correlation.
    pub document_id: &'a DocumentId,
    /// Complete text to embed.
    pub text: &'a str,
}

/// Model-provider adapter for ordered document and query embeddings.
pub trait Embedder: Send + Sync {
    /// Returns the exact vector and input-profile identity produced by this adapter.
    fn identity(&self) -> &EmbeddingIdentity;

    /// Output order must match input order exactly.
    fn embed_batch<'a>(
        &'a self,
        inputs: &'a [EmbeddingInput<'a>],
    ) -> BoxFuture<'a, Result<Vec<EmbeddingVector>, IndexError>>;

    /// Embeds one search query using the provider's retrieval-query profile.
    ///
    /// Asymmetric models must apply their query-specific instruction here
    /// rather than reusing the document embedding path.
    fn embed_query<'a>(
        &'a self,
        text: &'a str,
    ) -> BoxFuture<'a, Result<EmbeddingVector, IndexError>>;
}

/// Storage adapter for reusable chunk documents and completed checkpoints.
pub trait IndexStore: Send + Sync {
    /// Reports optional semantics implemented by the backend.
    fn capabilities(&self) -> StoreCapabilities;

    /// Finds reusable documents matching file content and fingerprint.
    fn lookup_file<'a>(
        &'a self,
        lookup: &'a FileLookup,
    ) -> BoxFuture<'a, Result<Option<Vec<DocumentId>>, IndexError>>;

    /// Atomically applies one file's mutations when the backend supports it.
    fn apply_batch<'a>(
        &'a self,
        batch: &'a WriteBatch,
    ) -> BoxFuture<'a, Result<WriteReceipt, IndexError>>;

    /// Loads the latest compatible checkpoint for a repository branch.
    fn load_checkpoint<'a>(
        &'a self,
        snapshot: &'a RepositorySnapshot,
    ) -> BoxFuture<'a, Result<Option<SyncCheckpoint>, IndexError>>;

    /// Makes all previously accepted writes durable.
    fn flush(&self) -> BoxFuture<'_, Result<(), IndexError>>;

    /// Must only be called after every file batch has succeeded and `flush`
    /// confirms durable writes.
    fn save_checkpoint<'a>(
        &'a self,
        checkpoint: &'a SyncCheckpoint,
    ) -> BoxFuture<'a, Result<(), IndexError>>;
}

/// Progress observers must return quickly and must not perform blocking I/O on
/// worker threads.
pub trait IndexObserver: Send + Sync {
    /// Receives one non-blocking progress event.
    fn on_event(&self, event: &IndexEvent);
}

/// Cooperative cancellation signal shared with an indexing run.
/// Identity of a reranking model, recorded with any ranking it produced.
///
/// A reranker changes order, never stored vectors, so it is deliberately not
/// part of the index manifest: turning one on or off is not a reindex. It is
/// still pinned and reported, because two runs whose ordering came from
/// different rerankers are not comparable evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RerankIdentity {
    /// Provider that executed the scoring, such as `cloudflare-workers-ai`.
    pub provider: String,
    /// Exact model identifier.
    pub model: String,
    /// Deployment revision the operator approved.
    pub revision: String,
}

impl RerankIdentity {
    /// Rejects a blank provider, model, or revision.
    ///
    /// # Errors
    ///
    /// Returns [`ContractError::Empty`] naming the first blank field, so an
    /// unattributable ranking cannot be reported as pinned.
    pub fn validate(&self) -> Result<(), ContractError> {
        for (field, value) in [
            ("rerank provider", &self.provider),
            ("rerank model", &self.model),
            ("rerank revision", &self.revision),
        ] {
            if value.trim().is_empty() {
                return Err(ContractError::Empty { field });
            }
        }
        Ok(())
    }
}

/// One query paired with the retrieved passages to be scored against it.
#[derive(Clone, Copy, Debug)]
pub struct RerankRequest<'a> {
    /// The searcher's query text, unmodified.
    pub query: &'a str,
    /// Retrieved passage texts, in the order the retriever returned them.
    pub passages: &'a [&'a str],
}

/// Relevance scores for one [`RerankRequest`].
///
/// Scores are provider-scaled and comparable only within a single response, so
/// a caller sorts by them and never thresholds against a fixed value.
#[derive(Clone, Debug)]
pub struct RerankScores {
    /// Model that produced these scores.
    pub identity: RerankIdentity,
    /// One score per input passage, in the order the passages were given.
    pub scores: Vec<f32>,
}

/// Reorders retrieved passages by scoring each against the query directly.
///
/// This is the cascade's last stage. A retriever ranks a passage without ever
/// seeing it beside the query; a cross-encoder scores the pair, which is more
/// accurate and far too slow to run over a corpus, so it only ever sees the
/// candidates a retriever already selected.
pub trait Reranker: Send + Sync {
    /// Returns the pinned identity of the scoring model.
    fn identity(&self) -> &RerankIdentity;

    /// Scores every passage against the query.
    ///
    /// Output order must match input order exactly, and the returned length must
    /// equal `request.passages.len()`; a caller relies on position to map a
    /// score back to the candidate it belongs to. Implementations must not
    /// reorder or truncate on the caller's behalf.
    fn rerank<'a>(
        &'a self,
        request: RerankRequest<'a>,
    ) -> BoxFuture<'a, Result<RerankScores, IndexError>>;
}

/// Cooperative cancellation signal shared with an indexing run.
pub trait Cancellation: Send + Sync {
    /// Returns whether the caller has requested cancellation.
    fn is_cancelled(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapter_traits_are_object_safe() {
        #[expect(
            clippy::too_many_arguments,
            reason = "one parameter per adapter trait; the point is object safety"
        )]
        fn accept(
            _source: Option<&dyn RepositorySource>,
            _factory: Option<&dyn ChunkEngineFactory>,
            _ids: Option<&dyn DocumentIdFactory>,
            _embedder: Option<&dyn Embedder>,
            _reranker: Option<&dyn Reranker>,
            _store: Option<&dyn IndexStore>,
            _observer: Option<&dyn IndexObserver>,
            _cancellation: Option<&dyn Cancellation>,
        ) {
        }

        accept(None, None, None, None, None, None, None, None);
    }
}
