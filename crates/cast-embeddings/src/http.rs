use std::fmt;
use std::num::NonZeroU64;
use std::time::Duration;

use cast_index::{EmbeddingIdentity, EmbeddingVector, IndexError, IndexErrorKind, RetryAdvice};
use reqwest::header::{HeaderMap, HeaderValue, RETRY_AFTER};
use reqwest::{Client, StatusCode, Url};
use serde::Deserialize;
use serde_json::json;
use thiserror::Error;

const MAX_RESPONSE_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub(crate) enum InputKind {
    Document,
    Query,
}

impl InputKind {
    const fn voyage_value(self) -> &'static str {
        match self {
            Self::Document => "document",
            Self::Query => "query",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Dialect {
    OpenAi,
    Voyage,
    CloudflareQwen,
}

/// Invalid configuration shared by hosted embedding adapters.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum RemoteEmbeddingConfigError {
    /// The endpoint is not a valid URL or does not match the selected provider.
    #[error("invalid embedding endpoint")]
    InvalidEndpoint,
    /// Production endpoints must use HTTPS.
    #[error("embedding endpoint must use HTTPS")]
    InsecureEndpoint,
    /// The bearer token is blank or cannot be represented as a header.
    #[error("invalid embedding bearer token")]
    InvalidBearer,
    /// A provider or model identifier is blank or malformed.
    #[error("invalid embedding provider or model identifier")]
    InvalidModel,
    /// The requested dimension is not supported by the selected model.
    #[error("invalid embedding dimensions")]
    InvalidDimensions,
    /// The `Cloudflare` account ID is blank or contains unsafe path characters.
    #[error("invalid Cloudflare account ID")]
    InvalidAccountId,
    /// A zero timeout would make every request fail immediately.
    #[error("embedding timeout must be greater than zero")]
    InvalidTimeout,
    /// The HTTP client could not be constructed.
    #[error("could not construct embedding HTTP client")]
    HttpClient,
}

pub(crate) struct HttpEmbeddingConfig {
    pub endpoint: String,
    pub bearer: String,
    pub identity: EmbeddingIdentity,
    pub dialect: Dialect,
    pub timeout: Duration,
    pub error_prefix: &'static str,
    pub provider_name: &'static str,
}

pub(crate) struct HttpEmbeddingClient {
    client: Client,
    endpoint: Url,
    authorization: HeaderValue,
    identity: EmbeddingIdentity,
    dialect: Dialect,
    error_prefix: &'static str,
    provider_name: &'static str,
}

impl fmt::Debug for HttpEmbeddingClient {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HttpEmbeddingClient")
            .field("endpoint", &self.endpoint)
            .field("authorization", &"[REDACTED]")
            .field("identity", &self.identity)
            .field("dialect", &self.dialect)
            .finish_non_exhaustive()
    }
}

impl HttpEmbeddingClient {
    pub(crate) fn new(config: HttpEmbeddingConfig) -> Result<Self, RemoteEmbeddingConfigError> {
        let endpoint = Url::parse(&config.endpoint)
            .map_err(|_| RemoteEmbeddingConfigError::InvalidEndpoint)?;
        if endpoint.scheme() != "https" && !is_test_loopback(&endpoint) {
            return Err(RemoteEmbeddingConfigError::InsecureEndpoint);
        }
        if config.timeout.is_zero() {
            return Err(RemoteEmbeddingConfigError::InvalidTimeout);
        }
        config
            .identity
            .validate()
            .map_err(|_| RemoteEmbeddingConfigError::InvalidModel)?;
        let token = config.bearer.trim();
        if token.is_empty() {
            return Err(RemoteEmbeddingConfigError::InvalidBearer);
        }
        let token = token.strip_prefix("Bearer ").unwrap_or(token);
        let mut authorization = HeaderValue::from_str(&format!("Bearer {token}"))
            .map_err(|_| RemoteEmbeddingConfigError::InvalidBearer)?;
        authorization.set_sensitive(true);
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|_| RemoteEmbeddingConfigError::HttpClient)?;
        Ok(Self {
            client,
            endpoint,
            authorization,
            identity: config.identity,
            dialect: config.dialect,
            error_prefix: config.error_prefix,
            provider_name: config.provider_name,
        })
    }

    pub(crate) fn identity(&self) -> &EmbeddingIdentity {
        &self.identity
    }

    pub(crate) async fn embed(
        &self,
        kind: InputKind,
        texts: &[&str],
    ) -> Result<Vec<EmbeddingVector>, IndexError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if texts.iter().any(|text| text.trim().is_empty()) {
            return Err(self.error(
                IndexErrorKind::Embedding,
                "empty_input",
                format!("{} embedding input must not be empty", self.provider_name),
            ));
        }
        let body = self.request_body(kind, texts);
        let request = self
            .client
            .post(self.endpoint.clone())
            .header("authorization", self.authorization.clone())
            .json(&body)
            .build()
            .map_err(|_| {
                self.error(
                    IndexErrorKind::Configuration,
                    "request_build",
                    format!("could not build {} embedding request", self.provider_name),
                )
            })?;
        let response = self
            .client
            .execute(request)
            .await
            .map_err(|error| self.transport_error(&error))?;
        let status = response.status();
        let headers = response.headers().clone();
        if !status.is_success() {
            return Err(self.status_error(status, &headers));
        }
        if response
            .content_length()
            .is_some_and(|length| length > MAX_RESPONSE_BYTES)
        {
            return Err(self.response_too_large());
        }
        let bytes = response
            .bytes()
            .await
            .map_err(|error| self.transport_error(&error))?;
        if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_RESPONSE_BYTES {
            return Err(self.response_too_large());
        }
        let values = match self.dialect {
            Dialect::OpenAi | Dialect::Voyage => self.decode_indexed(&bytes, texts.len())?,
            Dialect::CloudflareQwen => self.decode_cloudflare(&bytes, texts.len())?,
        };
        values
            .into_iter()
            .map(|values| {
                let vector = EmbeddingVector {
                    identity: self.identity.clone(),
                    values,
                };
                vector.validate().map_err(|error| {
                    self.error(
                        IndexErrorKind::Invariant,
                        "invalid_vector",
                        error.to_string(),
                    )
                })?;
                Ok(vector)
            })
            .collect()
    }

    pub(crate) fn request_body(&self, kind: InputKind, texts: &[&str]) -> serde_json::Value {
        match self.dialect {
            Dialect::OpenAi => json!({
                "input": texts,
                "model": self.identity.model,
                "dimensions": self.identity.dimensions,
                "encoding_format": "float"
            }),
            Dialect::Voyage => json!({
                "input": texts,
                "model": self.identity.model,
                "input_type": kind.voyage_value(),
                "truncation": false,
                "output_dimension": self.identity.dimensions,
                "output_dtype": "float"
            }),
            Dialect::CloudflareQwen => match kind {
                InputKind::Document => json!({ "documents": texts }),
                InputKind::Query => json!({
                    "queries": texts,
                    "instruction": "Given a code search query, retrieve relevant source code passages that answer the query"
                }),
            },
        }
    }

    fn decode_indexed(&self, bytes: &[u8], expected: usize) -> Result<Vec<Vec<f32>>, IndexError> {
        let mut response: IndexedResponse = serde_json::from_slice(bytes).map_err(|_| {
            self.error(
                IndexErrorKind::Embedding,
                "invalid_json",
                format!("{} returned invalid JSON", self.provider_name),
            )
        })?;
        response.data.sort_unstable_by_key(|item| item.index);
        if response.data.len() != expected
            || response
                .data
                .iter()
                .enumerate()
                .any(|(index, item)| item.index != index)
        {
            return Err(self.invalid_response("response indices did not match request order"));
        }
        Ok(response
            .data
            .into_iter()
            .map(|item| item.embedding)
            .collect())
    }

    fn decode_cloudflare(
        &self,
        bytes: &[u8],
        expected: usize,
    ) -> Result<Vec<Vec<f32>>, IndexError> {
        let response: CloudflareEnvelope = serde_json::from_slice(bytes).map_err(|_| {
            self.error(
                IndexErrorKind::Embedding,
                "invalid_json",
                "Cloudflare Workers AI returned invalid JSON",
            )
        })?;
        if !response.success {
            return Err(self.invalid_response("Cloudflare response reported failure"));
        }
        let result = response
            .result
            .ok_or_else(|| self.invalid_response("Cloudflare response omitted result"))?;
        if result.data.len() != expected
            || result
                .data
                .iter()
                .any(|values| values.len() != self.identity.dimensions)
        {
            return Err(self.invalid_response("Cloudflare response shape did not match request"));
        }
        if !result.shape.is_empty() && result.shape != [expected, self.identity.dimensions] {
            return Err(self.invalid_response("Cloudflare response declared an invalid shape"));
        }
        Ok(result.data)
    }

    fn transport_error(&self, error: &reqwest::Error) -> IndexError {
        if error.is_timeout() {
            return self
                .error(
                    IndexErrorKind::Timeout,
                    "timeout",
                    format!("{} embedding request timed out", self.provider_name),
                )
                .with_retry(RetryAdvice::AfterMillis(non_zero_millis(1_000)));
        }
        self.error(
            IndexErrorKind::Embedding,
            "transport",
            format!("{} embedding transport failed", self.provider_name),
        )
        .with_retry(RetryAdvice::AfterMillis(non_zero_millis(250)))
    }

    fn status_error(&self, status: StatusCode, headers: &HeaderMap) -> IndexError {
        let (kind, suffix, retry) = match status {
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                (IndexErrorKind::Configuration, "auth", RetryAdvice::Never)
            }
            StatusCode::TOO_MANY_REQUESTS => (
                IndexErrorKind::Embedding,
                "rate_limited",
                RetryAdvice::AfterMillis(retry_after(headers)),
            ),
            status if status.is_server_error() => (
                IndexErrorKind::Embedding,
                "upstream_unavailable",
                RetryAdvice::AfterMillis(non_zero_millis(1_000)),
            ),
            _ => (
                IndexErrorKind::Embedding,
                "request_rejected",
                RetryAdvice::Never,
            ),
        };
        self.error(
            kind,
            suffix,
            format!("{} returned HTTP {status}", self.provider_name),
        )
        .with_retry(retry)
    }

    fn response_too_large(&self) -> IndexError {
        self.error(
            IndexErrorKind::Embedding,
            "response_too_large",
            format!(
                "{} embedding response exceeded the safety limit",
                self.provider_name
            ),
        )
    }

    fn invalid_response(&self, message: &str) -> IndexError {
        self.error(
            IndexErrorKind::Embedding,
            "invalid_response",
            format!("{}: {message}", self.provider_name),
        )
    }

    fn error(&self, kind: IndexErrorKind, suffix: &str, message: impl Into<String>) -> IndexError {
        IndexError::new(kind, format!("{}_{suffix}", self.error_prefix), message)
    }
}

#[derive(Debug, Deserialize)]
struct IndexedResponse {
    data: Vec<IndexedEmbedding>,
}

#[derive(Debug, Deserialize)]
struct IndexedEmbedding {
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct CloudflareEnvelope {
    success: bool,
    result: Option<CloudflareEmbeddingResult>,
}

#[derive(Debug, Deserialize)]
struct CloudflareEmbeddingResult {
    #[serde(default)]
    shape: Vec<usize>,
    data: Vec<Vec<f32>>,
}

pub(crate) fn validate_endpoint(
    value: &str,
    host: &str,
    path: &str,
) -> Result<(), RemoteEmbeddingConfigError> {
    let endpoint = Url::parse(value).map_err(|_| RemoteEmbeddingConfigError::InvalidEndpoint)?;
    if is_test_loopback(&endpoint) {
        return Ok(());
    }
    if endpoint.scheme() != "https" {
        return Err(RemoteEmbeddingConfigError::InsecureEndpoint);
    }
    if endpoint.host_str() != Some(host)
        || endpoint.path() != path
        || endpoint.query().is_some()
        || endpoint.fragment().is_some()
    {
        return Err(RemoteEmbeddingConfigError::InvalidEndpoint);
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    fn client(dialect: Dialect) -> HttpEmbeddingClient {
        HttpEmbeddingClient::new(HttpEmbeddingConfig {
            endpoint: "http://127.0.0.1:9/embeddings".into(),
            bearer: "do-not-print".into(),
            identity: EmbeddingIdentity {
                provider: "test".into(),
                model: "test-model".into(),
                dimensions: 2,
                profile: "test-profile".into(),
            },
            dialect,
            timeout: Duration::from_secs(1),
            error_prefix: "test",
            provider_name: "Test provider",
        })
        .unwrap()
    }

    #[test]
    fn indexed_responses_are_reordered_and_must_be_complete() {
        let client = client(Dialect::OpenAi);
        let values = client
            .decode_indexed(
                br#"{"data":[{"index":1,"embedding":[3.0,4.0]},{"index":0,"embedding":[1.0,2.0]}]}"#,
                2,
            )
            .unwrap();
        assert_eq!(values, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert!(
            client
                .decode_indexed(br#"{"data":[{"index":0,"embedding":[1.0,2.0]}]}"#, 2)
                .is_err()
        );
    }

    #[test]
    fn cloudflare_envelope_must_report_success_and_exact_shape() {
        let client = client(Dialect::CloudflareQwen);
        let values = client
            .decode_cloudflare(
                br#"{"success":true,"result":{"shape":[2,2],"data":[[1.0,2.0],[3.0,4.0]]}}"#,
                2,
            )
            .unwrap();
        assert_eq!(values.len(), 2);
        assert!(
            client
                .decode_cloudflare(
                    br#"{"success":true,"result":{"shape":[1,2],"data":[[1.0,2.0],[3.0,4.0]]}}"#,
                    2
                )
                .is_err()
        );
    }

    #[test]
    fn retry_classification_is_shared_without_leaking_credentials() {
        let client = client(Dialect::Voyage);
        let rate_limit = client.status_error(StatusCode::TOO_MANY_REQUESTS, &HeaderMap::new());
        let bad_request = client.status_error(StatusCode::BAD_REQUEST, &HeaderMap::new());
        assert!(matches!(rate_limit.retry, RetryAdvice::AfterMillis(_)));
        assert_eq!(bad_request.retry, RetryAdvice::Never);
        assert!(!format!("{client:?}").contains("do-not-print"));
    }
}
