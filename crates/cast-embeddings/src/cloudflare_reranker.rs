use std::fmt;
use std::time::Duration;

use cast_index::{
    BoxFuture, IndexError, IndexErrorKind, RerankIdentity, RerankRequest, RerankScores, Reranker,
};
use reqwest::header::HeaderValue;
use reqwest::{Client, StatusCode};
use serde::Deserialize;
use serde_json::json;

use crate::http::{RemoteEmbeddingConfigError, validate_endpoint};

/// `BAAI` cross-encoder reranker hosted on `Workers AI`.
pub const CLOUDFLARE_RERANKER_MODEL: &str = "@cf/baai/bge-reranker-base";

const MAX_RESPONSE_BYTES: u64 = 16 * 1024 * 1024;
/// Passages scored in one request.
///
/// A reranker request carries the query plus every passage, so it grows far
/// faster than an embedding request. Sixteen six-kilobyte source chunks did not
/// merely get rejected: the connection failed during the TLS handshake's record
/// exchange with `AlertReceived(BadRecordMac)`, and Workers AI answered 500 to
/// comparably sized embedding batches from the same host. Four passages and 20k
/// characters complete reliably, so the bound stays there until someone measures
/// where the real ceiling is.
const MAX_PASSAGES_PER_REQUEST: usize = 4;
const MAX_CHARS_PER_REQUEST: usize = 20_000;

/// Configuration for the hosted `Workers AI` reranker.
pub struct CloudflareRerankerConfig {
    account_id: String,
    api_token: String,
    revision: String,
    endpoint: Option<String>,
    timeout: Duration,
}

impl CloudflareRerankerConfig {
    /// Uses `bge-reranker-base` and a 30-second request timeout.
    #[must_use]
    pub fn new(
        account_id: impl Into<String>,
        api_token: impl Into<String>,
        revision: impl Into<String>,
    ) -> Self {
        Self {
            account_id: account_id.into(),
            api_token: api_token.into(),
            revision: revision.into(),
            endpoint: None,
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

impl fmt::Debug for CloudflareRerankerConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CloudflareRerankerConfig")
            .field("account_id", &self.account_id)
            .field("api_token", &"[REDACTED]")
            .field("revision", &self.revision)
            .field("endpoint", &self.endpoint)
            .field("timeout", &self.timeout)
            .finish()
    }
}

/// Scores query/passage pairs through `Workers AI`'s reranker route.
#[derive(Debug)]
pub struct CloudflareReranker {
    client: Client,
    endpoint: String,
    authorization: HeaderValue,
    identity: RerankIdentity,
}

impl CloudflareReranker {
    /// Builds a reusable, thread-safe reranker.
    ///
    /// # Errors
    ///
    /// Returns [`RemoteEmbeddingConfigError`] for an invalid account, token,
    /// revision, endpoint, timeout, or HTTP client configuration.
    pub fn new(config: CloudflareRerankerConfig) -> Result<Self, RemoteEmbeddingConfigError> {
        if config.account_id.trim().is_empty()
            || !config
                .account_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
        {
            return Err(RemoteEmbeddingConfigError::InvalidAccountId);
        }
        if config.timeout.is_zero() {
            return Err(RemoteEmbeddingConfigError::InvalidTimeout);
        }
        let identity = RerankIdentity {
            provider: "cloudflare-workers-ai".into(),
            model: CLOUDFLARE_RERANKER_MODEL.into(),
            revision: config.revision.trim().into(),
        };
        identity
            .validate()
            .map_err(|_| RemoteEmbeddingConfigError::InvalidModel)?;
        let path = format!(
            "/client/v4/accounts/{}/ai/run/{CLOUDFLARE_RERANKER_MODEL}",
            config.account_id
        );
        let endpoint = config
            .endpoint
            .unwrap_or_else(|| format!("https://api.cloudflare.com{path}"));
        validate_endpoint(&endpoint, "api.cloudflare.com", &path)?;
        let token = config.api_token.trim();
        if token.is_empty() {
            return Err(RemoteEmbeddingConfigError::InvalidBearer);
        }
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
            identity,
        })
    }

    /// Splits passages into request-sized windows, preserving caller order.
    fn windows(passages: &[&str]) -> Vec<(usize, usize)> {
        let mut windows = Vec::new();
        let mut start = 0;
        while start < passages.len() {
            let mut end = start;
            let mut chars = 0;
            while end < passages.len() && end - start < MAX_PASSAGES_PER_REQUEST {
                let next = chars + passages[end].chars().count();
                if end > start && next > MAX_CHARS_PER_REQUEST {
                    break;
                }
                chars = next;
                end += 1;
            }
            windows.push((start, end));
            start = end;
        }
        windows
    }

    fn body(query: &str, passages: &[&str]) -> serde_json::Value {
        json!({
            "query": query,
            "contexts": passages.iter().map(|text| json!({ "text": text })).collect::<Vec<_>>(),
        })
    }

    async fn score_window(&self, query: &str, passages: &[&str]) -> Result<Vec<f32>, IndexError> {
        let response = self
            .client
            .post(&self.endpoint)
            .header("authorization", self.authorization.clone())
            .json(&Self::body(query, passages))
            .send()
            .await
            .map_err(|error| Self::transport_error(&error))?;
        let status = response.status();
        if !status.is_success() {
            return Err(Self::status_error(status));
        }
        if response
            .content_length()
            .is_some_and(|length| length > MAX_RESPONSE_BYTES)
        {
            return Err(Self::error(
                IndexErrorKind::Embedding,
                "response_too_large",
                "response",
            ));
        }
        let bytes = response
            .bytes()
            .await
            .map_err(|error| Self::transport_error(&error))?;
        let envelope: RerankEnvelope = serde_json::from_slice(&bytes).map_err(|_| {
            Self::error(
                IndexErrorKind::Embedding,
                "invalid_json",
                "Cloudflare reranker returned invalid JSON",
            )
        })?;
        if !envelope.success {
            return Err(Self::error(
                IndexErrorKind::Embedding,
                "upstream_failure",
                "Cloudflare reranker reported failure",
            ));
        }
        let result = envelope.result.ok_or_else(|| {
            Self::error(
                IndexErrorKind::Embedding,
                "missing_result",
                "Cloudflare reranker omitted its result",
            )
        })?;
        // The route returns results ordered by score and identifies each by its
        // index in the request, so order is restored by position rather than
        // trusted from the response.
        let mut scores = vec![f32::NAN; passages.len()];
        for entry in result.response {
            let slot = scores.get_mut(entry.id).ok_or_else(|| {
                Self::error(
                    IndexErrorKind::Embedding,
                    "index_out_of_range",
                    format!(
                        "Cloudflare reranker scored context {} of {}",
                        entry.id,
                        passages.len()
                    ),
                )
            })?;
            *slot = entry.score;
        }
        if let Some(position) = scores.iter().position(|score| score.is_nan()) {
            return Err(Self::error(
                IndexErrorKind::Embedding,
                "missing_score",
                format!("Cloudflare reranker left context {position} unscored"),
            ));
        }
        Ok(scores)
    }

    fn transport_error(error: &reqwest::Error) -> IndexError {
        let kind = if error.is_timeout() {
            IndexErrorKind::Timeout
        } else {
            IndexErrorKind::Embedding
        };
        Self::error(kind, "transport", "Cloudflare reranker request failed")
    }

    fn status_error(status: StatusCode) -> IndexError {
        let code = match status {
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => "unauthorized",
            StatusCode::TOO_MANY_REQUESTS => "throttled",
            status if status.is_server_error() => "upstream_unavailable",
            _ => "request_rejected",
        };
        Self::error(
            IndexErrorKind::Embedding,
            code,
            format!("Cloudflare reranker returned HTTP {status}"),
        )
    }

    fn error(kind: IndexErrorKind, code: &str, message: impl Into<String>) -> IndexError {
        IndexError::new(kind, format!("cloudflare_reranker_{code}"), message)
    }
}

impl Reranker for CloudflareReranker {
    fn identity(&self) -> &RerankIdentity {
        &self.identity
    }

    fn rerank<'a>(
        &'a self,
        request: RerankRequest<'a>,
    ) -> BoxFuture<'a, Result<RerankScores, IndexError>> {
        Box::pin(async move {
            if request.query.trim().is_empty() {
                return Err(Self::error(
                    IndexErrorKind::Embedding,
                    "empty_query",
                    "rerank query must not be empty",
                ));
            }
            let mut scores = Vec::with_capacity(request.passages.len());
            for (start, end) in Self::windows(request.passages) {
                scores.extend(
                    self.score_window(request.query, &request.passages[start..end])
                        .await?,
                );
            }
            Ok(RerankScores {
                identity: self.identity.clone(),
                scores,
            })
        })
    }
}

#[derive(Debug, Deserialize)]
struct RerankEnvelope {
    success: bool,
    result: Option<RerankResult>,
}

#[derive(Debug, Deserialize)]
struct RerankResult {
    response: Vec<RerankEntry>,
}

#[derive(Debug, Deserialize)]
struct RerankEntry {
    id: usize,
    score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter() -> CloudflareReranker {
        CloudflareReranker::new(CloudflareRerankerConfig::new(
            "account",
            "token",
            "benchmark-revision",
        ))
        .unwrap()
    }

    #[test]
    fn the_request_names_the_query_and_indexes_every_context() {
        let body =
            CloudflareReranker::body("where is retry configured", &["fn a() {}", "fn b() {}"]);
        assert_eq!(body["query"], "where is retry configured");
        assert_eq!(body["contexts"][0]["text"], "fn a() {}");
        assert_eq!(body["contexts"][1]["text"], "fn b() {}");
    }

    /// A reranker request carries every passage at once, so an unbounded batch
    /// would fail the same way the embedding adapters did on real source.
    #[test]
    fn passages_are_split_into_bounded_requests() {
        let many = vec!["x"; 40];
        let windows = CloudflareReranker::windows(&many);
        assert_eq!(windows.len(), 10);
        assert_eq!(windows[0], (0, 4));
        assert_eq!(windows[9], (36, 40));

        let long = "y".repeat(12_000);
        let heavy = vec![long.as_str(), long.as_str(), long.as_str()];
        let windows = CloudflareReranker::windows(&heavy);
        assert_eq!(
            windows.len(),
            3,
            "each passage exceeds half the char budget"
        );
    }

    #[test]
    fn one_oversized_passage_still_travels_alone() {
        let huge = "z".repeat(MAX_CHARS_PER_REQUEST * 2);
        let windows = CloudflareReranker::windows(&[huge.as_str()]);
        assert_eq!(windows, vec![(0, 1)]);
    }

    #[test]
    fn account_id_cannot_become_endpoint_path_input() {
        assert!(
            CloudflareReranker::new(CloudflareRerankerConfig::new(
                "../other-account",
                "token",
                "revision"
            ))
            .is_err()
        );
    }

    /// An unattributable ranking must not be reportable as pinned evidence.
    #[test]
    fn a_blank_revision_is_refused() {
        assert!(
            CloudflareReranker::new(CloudflareRerankerConfig::new("account", "token", "   "))
                .is_err()
        );
    }

    #[test]
    fn identity_names_the_hosted_cross_encoder() {
        let adapter = adapter();
        assert_eq!(adapter.identity().model, "@cf/baai/bge-reranker-base");
        assert_eq!(adapter.identity().provider, "cloudflare-workers-ai");
        assert_eq!(adapter.identity().revision, "benchmark-revision");
    }
}
