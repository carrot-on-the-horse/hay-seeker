use std::fmt;
use std::time::Duration;

use cast_index::{
    BoxFuture, Embedder, EmbeddingIdentity, EmbeddingInput, EmbeddingVector, IndexError,
};

use crate::http::{
    Dialect, HttpEmbeddingClient, HttpEmbeddingConfig, InputKind, RemoteEmbeddingConfigError,
    RequestLimits, validate_endpoint,
};

/// Default `Cloudflare`-hosted embedding model used by the adapter.
pub const COTH_HAY_SEEKER_CLOUDFLARE_WORKERS_AI_MODEL: &str = "@cf/qwen/qwen3-embedding-0.6b";
/// `Google EmbeddingGemma 300M` as hosted by `Workers AI`.
pub const CLOUDFLARE_EMBEDDINGGEMMA_MODEL: &str = "@cf/google/embeddinggemma-300m";

const DIMENSIONS: usize = 1_024;
const RETRIEVAL_PROFILE: &str = "cloudflare-qwen3-code-query-document-v1";
const EMBEDDINGGEMMA_DIMENSIONS: usize = 768;
const EMBEDDINGGEMMA_PROFILE: &str = "cloudflare-embeddinggemma-300m-server-prompt-v1";

/// A `Workers AI` embedding model this adapter can drive.
///
/// Each variant is a complete relevance contract — wire format, width, and
/// prompt pair — because those three travel together into the index manifest.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum WorkersAiModel {
    /// `Qwen3 Embedding 0.6B` at 1024 dimensions, using its native
    /// query/document route with a code-search instruction.
    #[default]
    Qwen3Embedding06B,
    /// `EmbeddingGemma 300M` at 768 dimensions over the generic text route.
    ///
    /// Inputs are sent verbatim. `EmbeddingGemma` is a prompt-conditioned
    /// checkpoint, but the hosted route applies its own template: on the frozen
    /// seed suite, adding the documented `task: code retrieval | query: ` and
    /// `title: none | text: ` prompts *lowered* nDCG@10 from 0.874084 to
    /// 0.847158 and MRR from 0.880878 to 0.843378, which is what double
    /// prompting looks like. Raw text is therefore the measured contract, and
    /// the profile name records that the prompt is the provider's.
    EmbeddingGemma300M,
}

impl WorkersAiModel {
    /// Parses the model selector accepted by configuration.
    ///
    /// # Errors
    ///
    /// Returns [`RemoteEmbeddingConfigError::InvalidModel`] for a name this
    /// adapter has no verified contract for. Guessing at an unknown model's
    /// width or prompt would silently produce an unusable index.
    pub fn parse(name: &str) -> Result<Self, RemoteEmbeddingConfigError> {
        match name.trim() {
            "qwen3-embedding-0.6b" | COTH_HAY_SEEKER_CLOUDFLARE_WORKERS_AI_MODEL => {
                Ok(Self::Qwen3Embedding06B)
            }
            "embeddinggemma-300m" | CLOUDFLARE_EMBEDDINGGEMMA_MODEL => Ok(Self::EmbeddingGemma300M),
            _ => Err(RemoteEmbeddingConfigError::InvalidModel),
        }
    }

    /// Fully qualified `Workers AI` model ID used in the request path.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::Qwen3Embedding06B => COTH_HAY_SEEKER_CLOUDFLARE_WORKERS_AI_MODEL,
            Self::EmbeddingGemma300M => CLOUDFLARE_EMBEDDINGGEMMA_MODEL,
        }
    }

    /// Width of the returned vectors.
    #[must_use]
    pub const fn dimensions(self) -> usize {
        match self {
            Self::Qwen3Embedding06B => DIMENSIONS,
            Self::EmbeddingGemma300M => EMBEDDINGGEMMA_DIMENSIONS,
        }
    }

    /// Embedding profile recorded in the index manifest.
    #[must_use]
    pub const fn profile(self) -> &'static str {
        match self {
            Self::Qwen3Embedding06B => RETRIEVAL_PROFILE,
            Self::EmbeddingGemma300M => EMBEDDINGGEMMA_PROFILE,
        }
    }

    const fn dialect(self) -> Dialect {
        match self {
            Self::Qwen3Embedding06B => Dialect::CloudflareQwen,
            Self::EmbeddingGemma300M => Dialect::CloudflareText,
        }
    }
}

/// Configuration for native `Cloudflare Workers AI` embeddings.
pub struct CloudflareWorkersAiEmbeddingsConfig {
    account_id: String,
    endpoint: Option<String>,
    api_token: String,
    model: WorkersAiModel,
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
            model: WorkersAiModel::default(),
            timeout: Duration::from_secs(30),
        }
    }

    /// Selects which hosted model to embed with.
    #[must_use]
    pub const fn with_model(mut self, model: WorkersAiModel) -> Self {
        self.model = model;
        self
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
            .field("model", &self.model)
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
        let model = config.model;
        let path = format!(
            "/client/v4/accounts/{}/ai/run/{}",
            config.account_id,
            model.id()
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
                model: model.id().into(),
                dimensions: model.dimensions(),
                profile: model.profile().into(),
            },
            dialect: model.dialect(),
            // Workers AI answered 500 for a 128-chunk batch of real source, so the
            // request stays well under that.
            limits: RequestLimits {
                max_texts: 64,
                max_chars: 100_000,
            },
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

    /// The generic text route carries inputs verbatim under both kinds. See
    /// [`WorkersAiModel::EmbeddingGemma300M`] for the measurement behind this.
    #[test]
    fn embeddinggemma_sends_text_verbatim_on_the_generic_route() {
        let adapter = CloudflareWorkersAiEmbeddings::new(
            CloudflareWorkersAiEmbeddingsConfig::new("account", "token")
                .with_model(WorkersAiModel::EmbeddingGemma300M),
        )
        .unwrap();

        let documents = adapter
            .inner
            .request_body(InputKind::Document, &["fn main() {}"]);
        let query = adapter
            .inner
            .request_body(InputKind::Query, &["entry point"]);

        assert_eq!(
            documents["text"][0], "fn main() {}",
            "the hosted route owns EmbeddingGemma's prompt; ours would double it"
        );
        assert_eq!(query["text"][0], "entry point");
        assert_eq!(adapter.identity().dimensions, EMBEDDINGGEMMA_DIMENSIONS);
        assert_eq!(adapter.identity().model, CLOUDFLARE_EMBEDDINGGEMMA_MODEL);
    }

    /// Two models must never be mistaken for one another by an opened index.
    #[test]
    fn each_model_has_its_own_manifest_identity() {
        let qwen = WorkersAiModel::Qwen3Embedding06B;
        let gemma = WorkersAiModel::EmbeddingGemma300M;
        assert_ne!(qwen.id(), gemma.id());
        assert_ne!(qwen.profile(), gemma.profile());
        assert_ne!(qwen.dimensions(), gemma.dimensions());
    }

    #[test]
    fn the_model_selector_accepts_short_and_qualified_names() {
        assert_eq!(
            WorkersAiModel::parse("embeddinggemma-300m").unwrap(),
            WorkersAiModel::EmbeddingGemma300M
        );
        assert_eq!(
            WorkersAiModel::parse(CLOUDFLARE_EMBEDDINGGEMMA_MODEL).unwrap(),
            WorkersAiModel::EmbeddingGemma300M
        );
        assert_eq!(
            WorkersAiModel::parse(" qwen3-embedding-0.6b ").unwrap(),
            WorkersAiModel::Qwen3Embedding06B
        );
        assert_eq!(WorkersAiModel::default(), WorkersAiModel::Qwen3Embedding06B);
    }

    /// An unverified model has no known width or prompt, and guessing would
    /// build an index that cannot be searched correctly.
    #[test]
    fn an_unknown_model_is_refused_rather_than_guessed() {
        assert!(WorkersAiModel::parse("@cf/baai/bge-m3").is_err());
        assert!(WorkersAiModel::parse("").is_err());
    }

    #[test]
    fn the_selected_model_appears_in_the_request_path() {
        let adapter = CloudflareWorkersAiEmbeddings::new(
            CloudflareWorkersAiEmbeddingsConfig::new("account", "token")
                .with_model(WorkersAiModel::EmbeddingGemma300M),
        )
        .unwrap();
        assert!(
            adapter
                .inner
                .endpoint_for_tests()
                .ends_with("/ai/run/@cf/google/embeddinggemma-300m"),
            "the route must address the selected model"
        );
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
