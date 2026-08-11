use std::future::Future;
use std::pin::Pin;

use cast_core::ChunkOutput;

use crate::{
    DocumentId, DocumentIdentityInput, EmbeddingIdentity, EmbeddingVector, FileDescriptor,
    FileInventoryRequest, FileLookup, IndexError, IndexEvent, RepositorySnapshot, SourceFile,
    StoreCapabilities, SyncCheckpoint, WriteBatch, WriteReceipt,
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
pub trait Cancellation: Send + Sync {
    /// Returns whether the caller has requested cancellation.
    fn is_cancelled(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapter_traits_are_object_safe() {
        fn accept(
            _source: Option<&dyn RepositorySource>,
            _factory: Option<&dyn ChunkEngineFactory>,
            _ids: Option<&dyn DocumentIdFactory>,
            _embedder: Option<&dyn Embedder>,
            _store: Option<&dyn IndexStore>,
            _observer: Option<&dyn IndexObserver>,
            _cancellation: Option<&dyn Cancellation>,
        ) {
        }

        accept(None, None, None, None, None, None, None);
    }
}
