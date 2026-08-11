use std::fmt;
use std::time::Duration;

use cast_index::{
    BoxFuture, Embedder, EmbeddingIdentity, EmbeddingInput, EmbeddingVector, IndexError,
};

use crate::http::{
    Dialect, HttpEmbeddingClient, HttpEmbeddingConfig, InputKind, RemoteEmbeddingConfigError,
    validate_endpoint,
};

/// Official `OpenAI` embeddings endpoint.
pub const OPENAI_EMBEDDINGS_ENDPOINT: &str = "https://api.openai.com/v1/embeddings";
/// Default `OpenAI` model selected by the provider adapter.
pub const OPENAI_DEFAULT_MODEL: &str = "text-embedding-3-small";

const DEFAULT_DIMENSIONS: usize = 768;
const MAX_DIMENSIONS: usize = 3_072;
const RETRIEVAL_PROFILE: &str = "openai-embeddings-symmetric-float-v1";

/// Configuration for `OpenAI`'s embeddings API.
pub struct OpenAiEmbeddingsConfig {
    endpoint: String,
    api_key: String,
    model: String,
    dimensions: usize,
    timeout: Duration,
}

impl OpenAiEmbeddingsConfig {
    /// Uses `text-embedding-3-small`, 768 dimensions, and a 30-second timeout.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            endpoint: OPENAI_EMBEDDINGS_ENDPOINT.into(),
            api_key: api_key.into(),
            model: OPENAI_DEFAULT_MODEL.into(),
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

    /// Selects an `OpenAI` embedding model.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Selects the output dimension sent to `text-embedding-3` models.
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

impl fmt::Debug for OpenAiEmbeddingsConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiEmbeddingsConfig")
            .field("endpoint", &self.endpoint)
            .field("api_key", &"[REDACTED]")
            .field("model", &self.model)
            .field("dimensions", &self.dimensions)
            .field("timeout", &self.timeout)
            .finish()
    }
}

/// `OpenAI` embeddings adapter implementing the common CAST contract.
#[derive(Debug)]
pub struct OpenAiEmbeddings {
    inner: HttpEmbeddingClient,
}

impl OpenAiEmbeddings {
    /// Builds a reusable, thread-safe `OpenAI` adapter.
    ///
    /// # Errors
    ///
    /// Returns [`RemoteEmbeddingConfigError`] for invalid credentials, model,
    /// dimensions, endpoint, timeout, or HTTP client configuration.
    pub fn new(config: OpenAiEmbeddingsConfig) -> Result<Self, RemoteEmbeddingConfigError> {
        validate_endpoint(&config.endpoint, "api.openai.com", "/v1/embeddings")?;
        if config.model.trim().is_empty() {
            return Err(RemoteEmbeddingConfigError::InvalidModel);
        }
        if config.dimensions == 0 || config.dimensions > MAX_DIMENSIONS {
            return Err(RemoteEmbeddingConfigError::InvalidDimensions);
        }
        let inner = HttpEmbeddingClient::new(HttpEmbeddingConfig {
            endpoint: config.endpoint,
            bearer: config.api_key,
            identity: EmbeddingIdentity {
                provider: "openai".into(),
                model: config.model,
                dimensions: config.dimensions,
                profile: RETRIEVAL_PROFILE.into(),
            },
            dialect: Dialect::OpenAi,
            timeout: config.timeout,
            error_prefix: "openai",
            provider_name: "OpenAI",
        })?;
        Ok(Self { inner })
    }
}

impl Embedder for OpenAiEmbeddings {
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
                        "openai_missing_embedding",
                        "OpenAI returned no query embedding",
                    )
                })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_and_identity_pin_symmetric_float_contract() {
        let adapter = OpenAiEmbeddings::new(OpenAiEmbeddingsConfig::new("test-key")).unwrap();
        let body = adapter
            .inner
            .request_body(InputKind::Query, &["where is the parser?"]);
        assert_eq!(body["model"], OPENAI_DEFAULT_MODEL);
        assert_eq!(body["dimensions"], DEFAULT_DIMENSIONS);
        assert_eq!(body["encoding_format"], "float");
        assert_eq!(adapter.identity().profile, RETRIEVAL_PROFILE);
    }

    #[test]
    fn configuration_rejects_secret_leaks_and_invalid_dimensions() {
        assert_eq!(
            OpenAiEmbeddings::new(OpenAiEmbeddingsConfig::new(""))
                .unwrap_err()
                .to_string(),
            "invalid embedding bearer token"
        );
        assert!(
            OpenAiEmbeddings::new(
                OpenAiEmbeddingsConfig::new("secret").with_dimensions(MAX_DIMENSIONS + 1)
            )
            .is_err()
        );
        assert!(
            !format!("{:?}", OpenAiEmbeddingsConfig::new("super-secret")).contains("super-secret")
        );
    }
}
