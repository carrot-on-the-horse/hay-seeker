use std::fmt;
use std::time::Duration;

use cast_index::{
    BoxFuture, Embedder, EmbeddingIdentity, EmbeddingInput, EmbeddingVector, IndexError,
};

use crate::http::{
    Dialect, HttpEmbeddingClient, HttpEmbeddingConfig, InputKind, RemoteEmbeddingConfigError,
    validate_endpoint,
};

/// Current `Cloudflare`-hosted embedding model used by the adapter.
pub const CLOUDFLARE_WORKERS_AI_MODEL: &str = "@cf/qwen/qwen3-embedding-0.6b";

const DIMENSIONS: usize = 1_024;
const RETRIEVAL_PROFILE: &str = "cloudflare-qwen3-code-query-document-v1";

/// Configuration for native `Cloudflare Workers AI` embeddings.
pub struct CloudflareWorkersAiEmbeddingsConfig {
    account_id: String,
    endpoint: Option<String>,
    api_token: String,
    timeout: Duration,
}

impl CloudflareWorkersAiEmbeddingsConfig {
    /// Uses `Qwen3 Embedding 0.6B` and a 30-second request timeout.
    #[must_use]
    pub fn new(account_id: impl Into<String>, api_token: impl Into<String>) -> Self {
        Self {
            account_id: account_id.into(),
            endpoint: None,
            api_token: api_token.into(),
            timeout: Duration::from_secs(30),
        }
    }

    /// Overrides the endpoint for contract tests.
    #[must_use]
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    /// Overrides the complete HTTP request timeout.
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

impl fmt::Debug for CloudflareWorkersAiEmbeddingsConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CloudflareWorkersAiEmbeddingsConfig")
            .field("account_id", &self.account_id)
            .field("endpoint", &self.endpoint)
            .field("api_token", &"[REDACTED]")
            .field("timeout", &self.timeout)
            .finish()
    }
}

/// `Cloudflare Workers AI` adapter using `Qwen3`'s native query/document modes.
#[derive(Debug)]
pub struct CloudflareWorkersAiEmbeddings {
    inner: HttpEmbeddingClient,
}

impl CloudflareWorkersAiEmbeddings {
    /// Builds a reusable, thread-safe Workers AI adapter.
    ///
    /// # Errors
    ///
    /// Returns [`RemoteEmbeddingConfigError`] for invalid account, token,
    /// endpoint, timeout, or HTTP client configuration.
    pub fn new(
        config: CloudflareWorkersAiEmbeddingsConfig,
    ) -> Result<Self, RemoteEmbeddingConfigError> {
        if config.account_id.trim().is_empty()
            || !config
                .account_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
        {
            return Err(RemoteEmbeddingConfigError::InvalidAccountId);
        }
        let path = format!(
            "/client/v4/accounts/{}/ai/run/{CLOUDFLARE_WORKERS_AI_MODEL}",
            config.account_id
        );
        let endpoint = config
            .endpoint
            .unwrap_or_else(|| format!("https://api.cloudflare.com{path}"));
        validate_endpoint(&endpoint, "api.cloudflare.com", &path)?;
        let inner = HttpEmbeddingClient::new(HttpEmbeddingConfig {
            endpoint,
            bearer: Some(config.api_token),
            gateway_bearer: None,
            identity: EmbeddingIdentity {
                provider: "cloudflare-workers-ai".into(),
                model: CLOUDFLARE_WORKERS_AI_MODEL.into(),
                dimensions: DIMENSIONS,
                profile: RETRIEVAL_PROFILE.into(),
            },
            dialect: Dialect::CloudflareQwen,
            timeout: config.timeout,
            error_prefix: "cloudflare_workers_ai",
            provider_name: "Cloudflare Workers AI",
        })?;
        Ok(Self { inner })
    }
}

impl Embedder for CloudflareWorkersAiEmbeddings {
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
                        "cloudflare_workers_ai_missing_embedding",
                        "Cloudflare Workers AI returned no query embedding",
                    )
                })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_requests_distinguish_code_queries_and_documents() {
        let adapter = CloudflareWorkersAiEmbeddings::new(CloudflareWorkersAiEmbeddingsConfig::new(
            "account", "token",
        ))
        .unwrap();
        let documents = adapter
            .inner
            .request_body(InputKind::Document, &["fn main() {}"]);
        let query = adapter
            .inner
            .request_body(InputKind::Query, &["entry point"]);
        assert_eq!(documents["documents"][0], "fn main() {}");
        assert_eq!(query["queries"][0], "entry point");
        assert!(
            query["instruction"]
                .as_str()
                .unwrap()
                .contains("code search")
        );
        assert_eq!(adapter.identity().dimensions, DIMENSIONS);
    }

    #[test]
    fn account_id_cannot_become_endpoint_path_input() {
        assert!(
            CloudflareWorkersAiEmbeddings::new(CloudflareWorkersAiEmbeddingsConfig::new(
                "../other-account",
                "token"
            ))
            .is_err()
        );
    }
}
