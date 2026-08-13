#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Provider-neutral embedding adapters for CAST indexing.
//!
//! Constructors validate credentials and model settings without sending a
//! request; embedding calls remain behind the common [`cast_index::Embedder`]
//! contract.
//!
//! ```
//! use cast_embeddings::{OpenAiEmbeddings, OpenAiEmbeddingsConfig};
//! use cast_index::Embedder;
//!
//! let config = OpenAiEmbeddingsConfig::through_cloudflare(
//!     "https://gateway.ai.cloudflare.com/v1/account/gateway/openai/embeddings",
//!     "gateway-run-token",
//! )
//! .with_api_key("openai-key");
//! let embedder = OpenAiEmbeddings::new(config)?;
//! assert_eq!(embedder.identity().provider, "openai");
//! # Ok::<(), cast_embeddings::RemoteEmbeddingConfigError>(())
//! ```

mod cloudflare_vertex_gemini;
mod cloudflare_workers_ai;
mod http;
mod local_onnx;
mod local_static;
pub mod model_catalog;
mod model_fetch;
mod openai;
mod retry;
mod voyage;

pub use cloudflare_vertex_gemini::{
    CloudflareVertexGemini2, CloudflareVertexGemini2Config, GeminiConfigError, GeminiQueryTask,
};
pub use cloudflare_workers_ai::{
    CLOUDFLARE_EMBEDDINGGEMMA_MODEL, COTH_HAY_SEEKER_CLOUDFLARE_WORKERS_AI_MODEL,
    CloudflareWorkersAiEmbeddings, CloudflareWorkersAiEmbeddingsConfig, WorkersAiModel,
};
pub use http::RemoteEmbeddingConfigError;
pub use local_onnx::{
    LocalExecutionProvider, LocalOnnxConfig, LocalOnnxEmbedder, LocalOnnxError,
    STATIC_RETRIEVAL_MRL_EN_V1_PROFILE,
};
pub use local_static::{
    LocalStaticConfig, LocalStaticEmbedder, LocalStaticError, POTION_CODE_16M_V2_PROFILE,
};
pub use model_catalog::{LocalModelArtifact, LocalModelEntry, LocalModelKind};
pub use model_fetch::{
    DEFAULT_MODEL_BASE_URL, ModelFetchConfig, ModelFetchError, ModelFetchEvent, ModelFetchReporter,
    ensure_bundle,
};
pub use openai::{
    OPENAI_DEFAULT_MODEL, OPENAI_EMBEDDINGS_ENDPOINT, OpenAiEmbeddings, OpenAiEmbeddingsConfig,
};
pub use retry::{RetryPolicy, RetryPolicyError, RetryingEmbedder};
pub use voyage::{
    VOYAGE_DEFAULT_MODEL, VOYAGE_EMBEDDINGS_ENDPOINT, VoyageEmbeddings, VoyageEmbeddingsConfig,
};
