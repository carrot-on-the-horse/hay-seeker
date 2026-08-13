use std::fmt;
use std::num::NonZeroU64;
use std::time::Duration;

use cast_index::{
    BoxFuture, Embedder, EmbeddingIdentity, EmbeddingInput, EmbeddingVector, IndexError,
    IndexErrorKind, RetryAdvice,
};
use futures::TryStreamExt;
use futures::stream::FuturesUnordered;
use reqwest::header::{HeaderMap, HeaderValue, RETRY_AFTER};
use reqwest::{Client, Request, StatusCode, Url};
use serde::{Deserialize, Serialize};
use thiserror::Error;

const PROVIDER_ID: &str = "cloudflare-ai-gateway/google-vertex-ai";
const MODEL_ID: &str = "gemini-embedding-2";
const RETRIEVAL_PROFILE: &str = "gemini-embedding-2-retrieval-prefix-v1";
const CODE_RETRIEVAL_PROFILE: &str = "gemini-embedding-2-code-retrieval-prefix-v1";
const DEFAULT_DIMENSIONS: usize = 768;
const MIN_DIMENSIONS: usize = 128;
const MAX_DIMENSIONS: usize = 3_072;
const DEFAULT_MAX_CONCURRENCY: usize = 8;
const MAX_RESPONSE_BYTES: u64 = 1_000_000;
const GEMINI_2_OPERATION: &str = "gemini-embedding-2:embedContent";

/// Configuration for Gemini Embedding 2 through Cloudflare AI Gateway.
pub struct CloudflareVertexGemini2Config {
    endpoint: String,
    gateway_bearer: String,
    dimensions: usize,
    query_task: GeminiQueryTask,
    max_concurrency: usize,
    timeout: Duration,
}

/// Retrieval task named in Gemini Embedding 2's query prefix.
///
/// Gemini Embedding 2 replaced the older `taskType` field with a versioned text
/// prefix, so the task is part of the query string and therefore part of the
/// relevance contract. Each task gets its own embedding profile: vectors built
/// under one task must never be searched with another, and the manifest is what
/// enforces that.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GeminiQueryTask {
    /// General retrieval: `task: search result | query: ...`.
    #[default]
    SearchResult,
    /// Code-specific retrieval: `task: code retrieval | query: ...`.
    CodeRetrieval,
}

impl GeminiQueryTask {
    /// Parses the task selector accepted by configuration.
    ///
    /// # Errors
    ///
    /// Returns [`GeminiConfigError::InvalidQueryTask`] for an unrecognized name
    /// rather than silently falling back to general retrieval.
    pub fn parse(name: &str) -> Result<Self, GeminiConfigError> {
        match name.trim() {
            "search-result" | "search result" => Ok(Self::SearchResult),
            "code-retrieval" | "code retrieval" => Ok(Self::CodeRetrieval),
            _ => Err(GeminiConfigError::InvalidQueryTask),
        }
    }

    const fn query_prefix(self) -> &'static str {
        match self {
            Self::SearchResult => "task: search result | query: ",
            Self::CodeRetrieval => "task: code retrieval | query: ",
        }
    }

    /// Embedding profile recorded in the index manifest for this task.
    #[must_use]
    pub const fn profile(self) -> &'static str {
        match self {
            Self::SearchResult => RETRIEVAL_PROFILE,
            Self::CodeRetrieval => CODE_RETRIEVAL_PROFILE,
        }
    }
}

impl CloudflareVertexGemini2Config {
    /// Creates configuration using an explicit Cloudflare AI Gateway endpoint,
    /// 768 dimensions, and a 30-second request timeout.
    #[must_use]
    pub fn new(endpoint: impl Into<String>, gateway_bearer: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            gateway_bearer: gateway_bearer.into(),
            dimensions: DEFAULT_DIMENSIONS,
            query_task: GeminiQueryTask::SearchResult,
            max_concurrency: DEFAULT_MAX_CONCURRENCY,
            timeout: Duration::from_secs(30),
        }
    }

    /// Overrides the provider-native endpoint, primarily for deployment
    /// migration and contract testing.
    #[must_use]
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Selects an output dimension supported by Gemini Embedding 2.
    #[must_use]
    pub const fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = dimensions;
        self
    }

    /// Selects the retrieval task named in the query prefix.
    #[must_use]
    pub const fn with_query_task(mut self, query_task: GeminiQueryTask) -> Self {
        self.query_task = query_task;
        self
    }

    /// Sets the maximum number of concurrent single-content requests used to
    /// implement the batch contract.
    #[must_use]
    pub const fn with_max_concurrency(mut self, max_concurrency: usize) -> Self {
        self.max_concurrency = max_concurrency;
        self
    }

    /// Overrides the complete HTTP request timeout.
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

impl fmt::Debug for CloudflareVertexGemini2Config {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CloudflareVertexGemini2Config")
            .field("endpoint", &self.endpoint)
            .field("gateway_bearer", &"[REDACTED]")
            .field("dimensions", &self.dimensions)
            .field("query_task", &self.query_task)
            .field("max_concurrency", &self.max_concurrency)
            .field("timeout", &self.timeout)
            .finish()
    }
}

/// Invalid local provider configuration.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum GeminiConfigError {
    /// The endpoint is not a valid URL.
    #[error("invalid Gemini gateway endpoint")]
    InvalidEndpoint,
    /// Production endpoints must use HTTPS.
    #[error("Gemini gateway endpoint must use HTTPS")]
    InsecureEndpoint,
    /// The configured token is empty or cannot be represented as a header.
    #[error("invalid Cloudflare AI Gateway bearer token")]
    InvalidGatewayBearer,
    /// The requested embedding dimension is outside the supported range.
    #[error("Gemini Embedding 2 dimensions must be between 128 and 3072")]
    InvalidDimensions,
    /// At least one request must be allowed to make progress.
    #[error("Gemini gateway max concurrency must be greater than zero")]
    InvalidConcurrency,
    /// A zero timeout would make every request fail immediately.
    #[error("Gemini gateway timeout must be greater than zero")]
    InvalidTimeout,
    /// The query task name is not one this adapter has a prefix for.
    #[error("invalid Gemini query task")]
    InvalidQueryTask,
    /// The HTTP client could not be constructed.
    #[error("could not construct Gemini gateway HTTP client")]
    HttpClient,
}

/// Gemini Embedding 2 adapter using Cloudflare's provider-native Vertex route.
pub struct CloudflareVertexGemini2 {
    client: Client,
    endpoint: Url,
    gateway_authorization: HeaderValue,
    identity: EmbeddingIdentity,
    query_task: GeminiQueryTask,
    max_concurrency: usize,
}

impl fmt::Debug for CloudflareVertexGemini2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CloudflareVertexGemini2")
            .field("endpoint", &self.endpoint)
            .field("gateway_authorization", &"[REDACTED]")
            .field("identity", &self.identity)
            .finish_non_exhaustive()
    }
}

impl CloudflareVertexGemini2 {
    /// Builds a reusable, thread-safe embedding adapter.
    ///
    /// # Errors
    ///
    /// Returns [`GeminiConfigError`] for an invalid endpoint, bearer token,
    /// dimension, timeout, or HTTP client configuration.
    pub fn new(config: CloudflareVertexGemini2Config) -> Result<Self, GeminiConfigError> {
        let CloudflareVertexGemini2Config {
            endpoint,
            gateway_bearer,
            dimensions,
            query_task,
            max_concurrency,
            timeout,
        } = config;
        let endpoint = Url::parse(&endpoint).map_err(|_| GeminiConfigError::InvalidEndpoint)?;
        if endpoint.scheme() != "https" && !is_test_loopback(&endpoint) {
            return Err(GeminiConfigError::InsecureEndpoint);
        }
        if !is_test_loopback(&endpoint) && !is_cloudflare_vertex_gemini_2_endpoint(&endpoint) {
            return Err(GeminiConfigError::InvalidEndpoint);
        }
        if !(MIN_DIMENSIONS..=MAX_DIMENSIONS).contains(&dimensions) {
            return Err(GeminiConfigError::InvalidDimensions);
        }
        if max_concurrency == 0 {
            return Err(GeminiConfigError::InvalidConcurrency);
        }
        if timeout.is_zero() {
            return Err(GeminiConfigError::InvalidTimeout);
        }

        let token = gateway_bearer.trim();
        if token.is_empty() {
            return Err(GeminiConfigError::InvalidGatewayBearer);
        }
        let authorization = token.strip_prefix("Bearer ").unwrap_or(token);
        let mut gateway_authorization = HeaderValue::from_str(&format!("Bearer {authorization}"))
            .map_err(|_| GeminiConfigError::InvalidGatewayBearer)?;
        gateway_authorization.set_sensitive(true);

        let client = Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|_| GeminiConfigError::HttpClient)?;
        Ok(Self {
            client,
            endpoint,
            gateway_authorization,
            identity: EmbeddingIdentity {
                provider: PROVIDER_ID.into(),
                model: MODEL_ID.into(),
                dimensions,
                profile: query_task.profile().into(),
            },
            query_task,
            max_concurrency,
        })
    }

    /// Embeds one corpus document using Gemini Embedding 2's retrieval-document
    /// prefix contract.
    ///
    /// # Errors
    ///
    /// Returns [`IndexError`] for empty input, transport/provider errors, or an
    /// invalid response vector.
    pub async fn embed_document(&self, text: &str) -> Result<EmbeddingVector, IndexError> {
        self.embed_one(InputKind::Document, text).await
    }

    /// Embeds one search query using Gemini Embedding 2's asymmetric retrieval
    /// prefix contract.
    ///
    /// # Errors
    ///
    /// Returns [`IndexError`] for empty input, transport/provider errors, or an
    /// invalid response vector.
    pub async fn embed_query(&self, text: &str) -> Result<EmbeddingVector, IndexError> {
        self.embed_one(InputKind::Query, text).await
    }

    async fn embed_one(&self, kind: InputKind, text: &str) -> Result<EmbeddingVector, IndexError> {
        if text.trim().is_empty() {
            return Err(IndexError::new(
                IndexErrorKind::Embedding,
                "gemini_empty_input",
                "Gemini embedding input must not be empty",
            ));
        }

        let request = self.build_request(kind, text)?;
        let response = self
            .client
            .execute(request)
            .await
            .map_err(|error| map_transport_error(&error))?;
        let status = response.status();
        let headers = response.headers().clone();
        if !status.is_success() {
            return Err(map_status_error(status, &headers));
        }
        if response
            .content_length()
            .is_some_and(|length| length > MAX_RESPONSE_BYTES)
        {
            return Err(response_too_large());
        }
        let body = response
            .bytes()
            .await
            .map_err(|error| map_transport_error(&error))?;
        if u64::try_from(body.len()).unwrap_or(u64::MAX) > MAX_RESPONSE_BYTES {
            return Err(response_too_large());
        }

        let response: EmbedContentResponse = serde_json::from_slice(&body).map_err(|_| {
            IndexError::new(
                IndexErrorKind::Embedding,
                "gemini_invalid_json",
                "Gemini gateway returned invalid JSON",
            )
        })?;
        let values = response.into_values().ok_or_else(|| {
            IndexError::new(
                IndexErrorKind::Embedding,
                "gemini_missing_embedding",
                "Gemini gateway response did not contain one embedding",
            )
        })?;
        let vector = EmbeddingVector {
            identity: self.identity.clone(),
            values,
        };
        vector.validate().map_err(|error| {
            IndexError::new(
                IndexErrorKind::Invariant,
                "gemini_invalid_vector",
                error.to_string(),
            )
        })?;
        Ok(vector)
    }

    async fn embed_indexed(
        &self,
        index: usize,
        text: &str,
    ) -> Result<(usize, EmbeddingVector), IndexError> {
        self.embed_document(text)
            .await
            .map(|vector| (index, vector))
    }

    fn build_request(&self, kind: InputKind, text: &str) -> Result<Request, IndexError> {
        let body = EmbedContentRequest {
            content: Content {
                role: "user",
                parts: [Part {
                    text: kind.format(text, self.query_task),
                }],
            },
            output_dimensionality: self.identity.dimensions,
        };
        self.client
            .post(self.endpoint.clone())
            .header("cf-aig-authorization", self.gateway_authorization.clone())
            .json(&body)
            .build()
            .map_err(|_| {
                IndexError::new(
                    IndexErrorKind::Configuration,
                    "gemini_request_build",
                    "could not build Gemini gateway request",
                )
            })
    }
}

impl Embedder for CloudflareVertexGemini2 {
    fn identity(&self) -> &EmbeddingIdentity {
        &self.identity
    }

    fn embed_batch<'a>(
        &'a self,
        inputs: &'a [EmbeddingInput<'a>],
    ) -> BoxFuture<'a, Result<Vec<EmbeddingVector>, IndexError>> {
        Box::pin(async move {
            let mut pending = FuturesUnordered::new();
            let mut indexed = Vec::with_capacity(inputs.len());
            for (index, input) in inputs.iter().enumerate() {
                pending.push(self.embed_indexed(index, input.text));
                if pending.len() >= self.max_concurrency {
                    let Some(vector) = pending.try_next().await? else {
                        return Err(IndexError::new(
                            IndexErrorKind::Embedding,
                            "gemini_concurrency_invariant",
                            "Gemini request queue became empty before reaching its configured concurrency limit",
                        ));
                    };
                    indexed.push(vector);
                }
            }
            while let Some(vector) = pending.try_next().await? {
                indexed.push(vector);
            }
            indexed.sort_unstable_by_key(|(index, _)| *index);
            Ok(indexed.into_iter().map(|(_, vector)| vector).collect())
        })
    }

    fn embed_query<'a>(
        &'a self,
        text: &'a str,
    ) -> BoxFuture<'a, Result<EmbeddingVector, IndexError>> {
        Box::pin(async move { CloudflareVertexGemini2::embed_query(self, text).await })
    }
}

#[derive(Clone, Copy, Debug)]
enum InputKind {
    Document,
    Query,
}

impl InputKind {
    fn format(self, text: &str, query_task: GeminiQueryTask) -> String {
        match self {
            Self::Document => format!("title: none | text: {text}"),
            Self::Query => format!("{}{text}", query_task.query_prefix()),
        }
    }
}

#[derive(Debug, Serialize)]
struct EmbedContentRequest {
    content: Content,
    #[serde(rename = "output_dimensionality")]
    output_dimensionality: usize,
}

#[derive(Debug, Serialize)]
struct Content {
    role: &'static str,
    parts: [Part; 1],
}

#[derive(Debug, Serialize)]
struct Part {
    text: String,
}

#[derive(Debug, Deserialize)]
struct EmbedContentResponse {
    embedding: Option<ContentEmbedding>,
    #[serde(default)]
    embeddings: Vec<ContentEmbedding>,
}

impl EmbedContentResponse {
    fn into_values(self) -> Option<Vec<f32>> {
        match (self.embedding, self.embeddings.as_slice()) {
            (Some(embedding), []) => Some(embedding.values),
            (None, [embedding]) => Some(embedding.values.clone()),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
struct ContentEmbedding {
    values: Vec<f32>,
}

fn map_transport_error(error: &reqwest::Error) -> IndexError {
    if error.is_timeout() {
        return IndexError::new(
            IndexErrorKind::Timeout,
            "gemini_timeout",
            "Gemini gateway request timed out",
        )
        .with_retry(RetryAdvice::AfterMillis(non_zero_millis(1_000)));
    }
    IndexError::new(
        IndexErrorKind::Embedding,
        "gemini_transport",
        "Gemini gateway transport failed",
    )
    .with_retry(RetryAdvice::AfterMillis(non_zero_millis(250)))
}

fn response_too_large() -> IndexError {
    IndexError::new(
        IndexErrorKind::Embedding,
        "gemini_response_too_large",
        "Gemini gateway response exceeded the safety limit",
    )
}

fn map_status_error(status: StatusCode, headers: &HeaderMap) -> IndexError {
    let ray = headers
        .get("cf-ray")
        .and_then(|value| value.to_str().ok())
        .map_or_else(String::new, |value| format!("; cf-ray={value}"));
    let (kind, code, retry) = match status {
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => (
            IndexErrorKind::Configuration,
            "gemini_gateway_auth",
            RetryAdvice::Never,
        ),
        StatusCode::TOO_MANY_REQUESTS => (
            IndexErrorKind::Embedding,
            "gemini_rate_limited",
            RetryAdvice::AfterMillis(retry_after(headers)),
        ),
        status if status.is_server_error() => (
            IndexErrorKind::Embedding,
            "gemini_upstream_unavailable",
            RetryAdvice::AfterMillis(non_zero_millis(1_000)),
        ),
        _ => (
            IndexErrorKind::Embedding,
            "gemini_request_rejected",
            RetryAdvice::Never,
        ),
    };
    IndexError::new(
        kind,
        code,
        format!("Gemini gateway returned HTTP {status}{ray}"),
    )
    .with_retry(retry)
}

fn retry_after(headers: &HeaderMap) -> NonZeroU64 {
    let millis = headers
        .get(RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .and_then(|seconds| seconds.checked_mul(1_000))
        .unwrap_or(30_000);
    non_zero_millis(millis)
}

fn non_zero_millis(value: u64) -> NonZeroU64 {
    NonZeroU64::new(value).unwrap_or(NonZeroU64::MIN)
}

fn is_test_loopback(endpoint: &Url) -> bool {
    cfg!(test)
        && endpoint.scheme() == "http"
        && matches!(endpoint.host_str(), Some("127.0.0.1" | "localhost" | "::1"))
}

fn is_cloudflare_vertex_gemini_2_endpoint(endpoint: &Url) -> bool {
    if endpoint.host_str() != Some("gateway.ai.cloudflare.com")
        || endpoint.query().is_some()
        || endpoint.fragment().is_some()
    {
        return false;
    }
    let Some(segments) = endpoint.path_segments() else {
        return false;
    };
    let segments = segments.collect::<Vec<_>>();
    segments.len() == 13
        && segments[0] == "v1"
        && !segments[1].is_empty()
        && !segments[2].is_empty()
        && segments[3] == "google-vertex-ai"
        && segments[4] == "v1"
        && segments[5] == "projects"
        && !segments[6].is_empty()
        && segments[7] == "locations"
        && !segments[8].is_empty()
        && segments[9] == "publishers"
        && segments[10] == "google"
        && segments[11] == "models"
        && segments[12] == GEMINI_2_OPERATION
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_ENDPOINT: &str = "https://gateway.ai.cloudflare.com/v1/test-account/test-gateway/google-vertex-ai/v1/projects/test-project/locations/global/publishers/google/models/gemini-embedding-2:embedContent";

    fn adapter() -> CloudflareVertexGemini2 {
        CloudflareVertexGemini2::new(CloudflareVertexGemini2Config::new(
            TEST_ENDPOINT,
            "test-token",
        ))
        .expect("valid adapter")
    }

    #[test]
    fn request_uses_gateway_header_and_gemini_2_prefixes() {
        let adapter = adapter();
        let request = adapter
            .build_request(InputKind::Query, "where is chunking configured?")
            .unwrap();
        let header = request.headers().get("cf-aig-authorization").unwrap();
        assert_eq!(header, "Bearer test-token");
        assert!(header.is_sensitive());
        assert!(request.headers().get("authorization").is_none());

        let bytes = request.body().and_then(reqwest::Body::as_bytes).unwrap();
        let json: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        assert_eq!(json["output_dimensionality"], 768);
        assert_eq!(
            json["content"]["parts"][0]["text"],
            "task: search result | query: where is chunking configured?"
        );
        assert!(json.get("taskType").is_none());
    }

    /// A code-retrieval index and a general-retrieval index must be
    /// distinguishable by manifest, not just by the prefix they happened to use.
    #[test]
    fn the_code_retrieval_task_changes_both_the_prefix_and_the_profile() {
        let adapter = CloudflareVertexGemini2::new(
            CloudflareVertexGemini2Config::new(TEST_ENDPOINT, "token")
                .with_query_task(GeminiQueryTask::CodeRetrieval),
        )
        .unwrap();

        assert_eq!(
            InputKind::Query.format(
                "where is chunking configured?",
                GeminiQueryTask::CodeRetrieval
            ),
            "task: code retrieval | query: where is chunking configured?"
        );
        assert_eq!(
            InputKind::Document.format("fn main() {}", GeminiQueryTask::CodeRetrieval),
            "title: none | text: fn main() {}",
            "the document prefix does not carry a task"
        );
        assert_eq!(adapter.identity().profile, CODE_RETRIEVAL_PROFILE);
        assert_ne!(CODE_RETRIEVAL_PROFILE, RETRIEVAL_PROFILE);
    }

    #[test]
    fn an_unknown_query_task_is_refused() {
        assert!(GeminiQueryTask::parse("code-retrieval").is_ok());
        assert!(GeminiQueryTask::parse("search-result").is_ok());
        assert!(GeminiQueryTask::parse("question answering").is_err());
        assert_eq!(GeminiQueryTask::default(), GeminiQueryTask::SearchResult);
    }

    #[test]
    fn document_prefix_and_identity_are_versioned() {
        let adapter = adapter();
        let request = adapter
            .build_request(InputKind::Document, "fn main() {}")
            .unwrap();
        let bytes = request.body().and_then(reqwest::Body::as_bytes).unwrap();
        let json: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        assert_eq!(
            json["content"]["parts"][0]["text"],
            "title: none | text: fn main() {}"
        );
        assert_eq!(adapter.identity().model, MODEL_ID);
        assert_eq!(adapter.identity().profile, RETRIEVAL_PROFILE);
        adapter.identity().validate().unwrap();
    }

    #[test]
    fn response_accepts_singular_and_single_plural_shapes() {
        let singular: EmbedContentResponse =
            serde_json::from_str(r#"{"embedding":{"values":[1.0,2.0]}}"#).unwrap();
        assert_eq!(singular.into_values(), Some(vec![1.0, 2.0]));

        let plural: EmbedContentResponse =
            serde_json::from_str(r#"{"embeddings":[{"values":[3.0,4.0]}]}"#).unwrap();
        assert_eq!(plural.into_values(), Some(vec![3.0, 4.0]));
    }

    #[test]
    fn configuration_rejects_invalid_security_and_dimensions() {
        let insecure = CloudflareVertexGemini2Config::new("http://example.com/embed", "token");
        assert!(matches!(
            CloudflareVertexGemini2::new(insecure),
            Err(GeminiConfigError::InsecureEndpoint)
        ));

        let wrong_model = CloudflareVertexGemini2Config::new(
            "https://gateway.ai.cloudflare.com/v1/account/gateway/google-vertex-ai/v1/projects/project/locations/global/publishers/google/models/other:embedContent",
            "token",
        );
        assert!(matches!(
            CloudflareVertexGemini2::new(wrong_model),
            Err(GeminiConfigError::InvalidEndpoint)
        ));

        let dimensions = CloudflareVertexGemini2Config::new(TEST_ENDPOINT, "token")
            .with_dimensions(MAX_DIMENSIONS + 1);
        assert!(matches!(
            CloudflareVertexGemini2::new(dimensions),
            Err(GeminiConfigError::InvalidDimensions)
        ));

        let concurrency =
            CloudflareVertexGemini2Config::new(TEST_ENDPOINT, "token").with_max_concurrency(0);
        assert!(matches!(
            CloudflareVertexGemini2::new(concurrency),
            Err(GeminiConfigError::InvalidConcurrency)
        ));
    }

    #[test]
    fn provider_errors_preserve_retry_contract_without_response_body() {
        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, HeaderValue::from_static("7"));
        let error = map_status_error(StatusCode::TOO_MANY_REQUESTS, &headers);
        assert_eq!(error.code, "gemini_rate_limited");
        assert_eq!(
            error.retry,
            RetryAdvice::AfterMillis(non_zero_millis(7_000))
        );
        assert!(!error.message.contains("token"));
    }
}
