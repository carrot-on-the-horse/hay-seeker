use cast_core::LanguageId;
use cast_index::{DocumentId, NormalizedPath};

use crate::SearchError;

#[derive(Clone, Debug, Eq, PartialEq)]
/// Source document supplied to chunking and indexing.
pub struct CorpusDocument {
    /// Stable identity independent of processing order.
    pub doc_id: DocumentId,
    /// Normalized repository-relative path.
    pub path: NormalizedPath,
    /// Explicit or resolved language identifier.
    pub language: LanguageId,
    /// Exact UTF-8 source contents.
    pub text: String,
}

/// Streaming source of evaluation or indexing documents.
pub trait Corpus: Send + Sync {
    /// Opens a deterministic document iterator.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::Corpus`] when the source cannot be opened.
    fn documents(
        &self,
    ) -> Result<Box<dyn Iterator<Item = Result<CorpusDocument, SearchError>> + '_>, SearchError>;
}
