use std::fmt;
use std::time::Duration;

use cast_index::{
    BoxFuture, Embedder, EmbeddingIdentity, EmbeddingInput, EmbeddingVector, IndexError,
};

use crate::http::{
    Dialect, HttpEmbeddingClient, HttpEmbeddingConfig, InputKind, RemoteEmbeddingConfigError,
    RequestLimits, validate_endpoint,
};

/// Official `Voyage` text-embeddings endpoint.
pub const VOYAGE_EMBEDDINGS_ENDPOINT: &str = "https://api.voyageai.com/v1/embeddings";
/// Code-specialized `Voyage` model selected by default.
pub const VOYAGE_DEFAULT_MODEL: &str = "voyage-code-3";

const DEFAULT_DIMENSIONS: usize = 1_024;
const RETRIEVAL_PROFILE: &str = "voyage-retrieval-input-type-float-no-truncation-v1";

/// Configuration for `Voyage` text embeddings.
pub struct VoyageEmbeddingsConfig {
    endpoint: String,
    api_key: String,
    model: String,
    dimensions: usize,
    timeout: Duration,
}

impl VoyageEmbeddingsConfig {
    /// Uses `voyage-code-3`, 1,024 dimensions, and a 30-second timeout.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            endpoint: VOYAGE_EMBEDDINGS_ENDPOINT.into(),
            api_key: api_key.into(),
            model: VOYAGE_DEFAULT_MODEL.into(),
            dimensions: DEFAULT_DIMENSIONS,
            timeout: Duration::from_secs(30),
        }
    }

    /// Overrides the endpoint for contract tests.
    #[must_use]
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Selects a `Voyage` embedding model.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Selects one of `Voyage`'s supported MRL dimensions.
    #[must_use]
    pub const fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = dimensions;
        self
    }

    /// Overrides the complete HTTP request timeout.
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

impl fmt::Debug for VoyageEmbeddingsConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VoyageEmbeddingsConfig")
            .field("endpoint", &self.endpoint)
            .field("api_key", &"[REDACTED]")
            .field("model", &self.model)
            .field("dimensions", &self.dimensions)
            .field("timeout", &self.timeout)
            .finish()
    }
}

/// `Voyage` adapter with asymmetric query/document input types.
#[derive(Debug)]
pub struct VoyageEmbeddings {
    inner: HttpEmbeddingClient,
}

impl VoyageEmbeddings {
    /// Builds a reusable, thread-safe `Voyage` adapter.
    ///
    /// # Errors
    ///
    /// Returns [`RemoteEmbeddingConfigError`] for invalid credentials, model,
    /// dimensions, endpoint, timeout, or HTTP client configuration.
    pub fn new(config: VoyageEmbeddingsConfig) -> Result<Self, RemoteEmbeddingConfigError> {
        validate_endpoint(&config.endpoint, "api.voyageai.com", "/v1/embeddings")?;
        if config.model.trim().is_empty() {
            return Err(RemoteEmbeddingConfigError::InvalidModel);
        }
        if !matches!(config.dimensions, 256 | 512 | 1_024 | 2_048) {
            return Err(RemoteEmbeddingConfigError::InvalidDimensions);
        }
        let inner = HttpEmbeddingClient::new(HttpEmbeddingConfig {
            endpoint: config.endpoint,
            bearer: Some(config.api_key),
            gateway_bearer: None,
            identity: EmbeddingIdentity {
                provider: "voyage".into(),
                model: config.model,
                dimensions: config.dimensions,
                profile: RETRIEVAL_PROFILE.into(),
            },
            dialect: Dialect::Voyage,
            // Voyage rejects a batch over 120k tokens outright rather than
            // embedding a prefix. Measured against this corpus, Go source runs
            // about 1.41 characters per token, so the bound is expressed in
            // characters at a pessimistic one-to-one: 100k characters cannot
            // exceed 100k tokens whatever the input looks like.
            limits: RequestLimits {
                max_texts: 64,
                max_chars: 100_000,
            },
            timeout: config.timeout,
            error_prefix: "voyage",
            provider_name: "Voyage",
        })?;
        Ok(Self { inner })
    }
}

impl Embedder for VoyageEmbeddings {
    fn identity(&self) -> &EmbeddingIdentity {
        self.inner.identity()
    }

    fn embed_batch<'a>(
        &'a self,
        inputs: &'a [EmbeddingInput<'a>],
    ) -> BoxFuture<'a, Result<Vec<EmbeddingVector>, IndexError>> {
        Box::pin(async move {
            let texts = inputs.iter().map(|input| input.text).collect::<Vec<_>>();
            self.inner.embed(InputKind::Document, &texts).await
        })
    }

    fn embed_query<'a>(
        &'a self,
        text: &'a str,
    ) -> BoxFuture<'a, Result<EmbeddingVector, IndexError>> {
        Box::pin(async move {
            self.inner
                .embed(InputKind::Query, &[text])
                .await?
                .into_iter()
                .next()
                .ok_or_else(|| {
                    IndexError::new(
                        cast_index::IndexErrorKind::Embedding,
                        "voyage_missing_embedding",
                        "Voyage returned no query embedding",
                    )
                })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_pin_asymmetric_input_types_and_disable_truncation() {
        let adapter = VoyageEmbeddings::new(VoyageEmbeddingsConfig::new("test-key")).unwrap();
        let document = adapter
            .inner
            .request_body(InputKind::Document, &["fn main() {}"]);
        let query = adapter
            .inner
            .request_body(InputKind::Query, &["entry point"]);
        assert_eq!(document["input_type"], "document");
        assert_eq!(query["input_type"], "query");
        assert_eq!(document["truncation"], false);
        assert_eq!(document["output_dtype"], "float");
        assert_eq!(adapter.identity().model, VOYAGE_DEFAULT_MODEL);
    }
}
