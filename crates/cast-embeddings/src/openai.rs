use std::fmt;
use std::time::Duration;

use cast_index::{
    BoxFuture, Embedder, EmbeddingIdentity, EmbeddingInput, EmbeddingVector, IndexError,
};

use crate::http::{
    Dialect, HttpEmbeddingClient, HttpEmbeddingConfig, InputKind, RemoteEmbeddingConfigError,
    validate_endpoint,
};

/// Default `OpenAI` model selected by the provider adapter.
pub const OPENAI_DEFAULT_MODEL: &str = "text-embedding-3-small";
/// Official direct `OpenAI` embeddings endpoint.
pub const OPENAI_EMBEDDINGS_ENDPOINT: &str = "https://api.openai.com/v1/embeddings";

const DEFAULT_DIMENSIONS: usize = 768;
const MAX_DIMENSIONS: usize = 3_072;
const RETRIEVAL_PROFILE: &str = "openai-embeddings-symmetric-float-v1";

/// Configuration for `OpenAI`'s embeddings API.
pub struct OpenAiEmbeddingsConfig {
    endpoint: String,
    api_key: Option<String>,
    gateway_bearer: Option<String>,
    model: String,
    dimensions: usize,
    timeout: Duration,
}

impl OpenAiEmbeddingsConfig {
    /// Calls the official `OpenAI` embeddings endpoint directly.
    #[must_use]
    pub fn direct(api_key: impl Into<String>) -> Self {
        Self {
            endpoint: OPENAI_EMBEDDINGS_ENDPOINT.into(),
            api_key: Some(api_key.into()),
            gateway_bearer: None,
            model: OPENAI_DEFAULT_MODEL.into(),
            dimensions: DEFAULT_DIMENSIONS,
            timeout: Duration::from_secs(30),
        }
    }

    /// Calls `OpenAI` through a provider-native Cloudflare AI Gateway route.
    /// The upstream key can be added with [`Self::with_api_key`], or omitted
    /// when the gateway uses stored keys or Unified Billing.
    #[must_use]
    pub fn through_cloudflare(
        endpoint: impl Into<String>,
        gateway_bearer: impl Into<String>,
    ) -> Self {
        Self {
            endpoint: endpoint.into(),
            api_key: None,
            gateway_bearer: Some(gateway_bearer.into()),
            model: OPENAI_DEFAULT_MODEL.into(),
            dimensions: DEFAULT_DIMENSIONS,
            timeout: Duration::from_secs(30),
        }
    }

    /// Sends an upstream `OpenAI` API key through the gateway. Omit this when
    /// the gateway uses stored keys or Unified Billing.
    #[must_use]
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
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
            .field("api_key", &self.api_key.as_ref().map(|_| "[REDACTED]"))
            .field(
                "gateway_bearer",
                &self.gateway_bearer.as_ref().map(|_| "[REDACTED]"),
            )
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
        if config.gateway_bearer.is_some() {
            validate_cloudflare_openai_endpoint(&config.endpoint)?;
        } else {
            validate_endpoint(&config.endpoint, "api.openai.com", "/v1/embeddings")?;
        }
        if config.model.trim().is_empty() {
            return Err(RemoteEmbeddingConfigError::InvalidModel);
        }
        if config.dimensions == 0 || config.dimensions > MAX_DIMENSIONS {
            return Err(RemoteEmbeddingConfigError::InvalidDimensions);
        }
        let inner = HttpEmbeddingClient::new(HttpEmbeddingConfig {
            endpoint: config.endpoint,
            bearer: config.api_key,
            gateway_bearer: config.gateway_bearer,
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

fn validate_cloudflare_openai_endpoint(value: &str) -> Result<(), RemoteEmbeddingConfigError> {
    let endpoint =
        reqwest::Url::parse(value).map_err(|_| RemoteEmbeddingConfigError::InvalidEndpoint)?;
    if endpoint.scheme() != "https" {
        return Err(RemoteEmbeddingConfigError::InsecureEndpoint);
    }
    if endpoint.host_str() != Some("gateway.ai.cloudflare.com")
        || !endpoint.username().is_empty()
        || endpoint.password().is_some()
        || endpoint.port().is_some()
        || endpoint.query().is_some()
        || endpoint.fragment().is_some()
    {
        return Err(RemoteEmbeddingConfigError::InvalidEndpoint);
    }
    let Some(segments) = endpoint.path_segments() else {
        return Err(RemoteEmbeddingConfigError::InvalidEndpoint);
    };
    let segments = segments.collect::<Vec<_>>();
    if segments.len() != 5
        || segments[0] != "v1"
        || !is_safe_gateway_segment(segments[1])
        || !is_safe_gateway_segment(segments[2])
        || segments[3] != "openai"
        || segments[4] != "embeddings"
    {
        return Err(RemoteEmbeddingConfigError::InvalidEndpoint);
    }
    Ok(())
}

fn is_safe_gateway_segment(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
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

    const GATEWAY_ENDPOINT: &str =
        "https://gateway.ai.cloudflare.com/v1/test-account/test-gateway/openai/embeddings";

    #[test]
    fn request_and_identity_pin_symmetric_float_contract() {
        let adapter = OpenAiEmbeddings::new(OpenAiEmbeddingsConfig::direct("openai-key")).unwrap();
        let body = adapter
            .inner
            .request_body(InputKind::Query, &["where is the parser?"]);
        assert_eq!(body["model"], OPENAI_DEFAULT_MODEL);
        assert_eq!(body["dimensions"], DEFAULT_DIMENSIONS);
        assert_eq!(body["encoding_format"], "float");
        assert_eq!(adapter.identity().profile, RETRIEVAL_PROFILE);
        let request = adapter
            .inner
            .build_request(InputKind::Query, &["where is the parser?"])
            .unwrap();
        assert_eq!(request.url().as_str(), OPENAI_EMBEDDINGS_ENDPOINT);
        assert_eq!(
            request.headers().get("authorization").unwrap(),
            "Bearer openai-key"
        );
        assert!(request.headers().get("cf-aig-authorization").is_none());
    }

    #[test]
    fn configuration_rejects_secret_leaks_and_invalid_dimensions() {
        assert_eq!(
            OpenAiEmbeddings::new(OpenAiEmbeddingsConfig::direct(""))
                .unwrap_err()
                .to_string(),
            "invalid embedding bearer token"
        );
        assert!(
            OpenAiEmbeddings::new(
                OpenAiEmbeddingsConfig::direct("secret").with_dimensions(MAX_DIMENSIONS + 1)
            )
            .is_err()
        );
        let debug = format!(
            "{:?}",
            OpenAiEmbeddingsConfig::through_cloudflare(GATEWAY_ENDPOINT, "gateway-secret")
                .with_api_key("super-secret")
        );
        assert!(!debug.contains("super-secret"));
        assert!(!debug.contains("gateway-secret"));
    }

    #[test]
    fn cloudflare_gateway_request_separates_provider_and_gateway_credentials() {
        let adapter = OpenAiEmbeddings::new(
            OpenAiEmbeddingsConfig::through_cloudflare(GATEWAY_ENDPOINT, "gateway-token")
                .with_api_key("openai-key"),
        )
        .unwrap();
        let request = adapter
            .inner
            .build_request(InputKind::Query, &["where is the parser?"])
            .unwrap();

        assert_eq!(request.url().as_str(), GATEWAY_ENDPOINT);
        let provider = request.headers().get("authorization").unwrap();
        assert_eq!(provider, "Bearer openai-key");
        assert!(provider.is_sensitive());
        let gateway = request.headers().get("cf-aig-authorization").unwrap();
        assert_eq!(gateway, "Bearer gateway-token");
        assert!(gateway.is_sensitive());
    }

    #[test]
    fn cloudflare_stored_key_request_omits_upstream_authorization() {
        let adapter = OpenAiEmbeddings::new(OpenAiEmbeddingsConfig::through_cloudflare(
            GATEWAY_ENDPOINT,
            "gateway-token",
        ))
        .unwrap();
        let request = adapter
            .inner
            .build_request(InputKind::Document, &["fn main() {}"])
            .unwrap();

        assert!(request.headers().get("authorization").is_none());
        assert_eq!(
            request.headers().get("cf-aig-authorization").unwrap(),
            "Bearer gateway-token"
        );
    }

    #[test]
    fn cloudflare_gateway_configuration_fails_closed() {
        assert_eq!(
            OpenAiEmbeddings::new(OpenAiEmbeddingsConfig::through_cloudflare(
                GATEWAY_ENDPOINT,
                "",
            ))
            .unwrap_err()
            .to_string(),
            "invalid Cloudflare AI Gateway bearer token"
        );
        for endpoint in [
            "https://api.openai.com/v1/embeddings",
            "https://gateway.ai.cloudflare.com/v1/account/gateway/openai",
            "https://gateway.ai.cloudflare.com/v1/account/gateway/openai/embeddings?leak=true",
            "https://gateway.ai.cloudflare.com/v1/account/%2Fescape/openai/embeddings",
            "https://user@gateway.ai.cloudflare.com/v1/account/gateway/openai/embeddings",
        ] {
            assert!(
                OpenAiEmbeddings::new(OpenAiEmbeddingsConfig::through_cloudflare(
                    endpoint,
                    "gateway-token",
                ))
                .is_err()
            );
        }
    }
}
