use std::fmt;
use std::num::NonZeroU64;
use std::time::Duration;

use cast_index::{
    BoxFuture, IndexError, IndexErrorKind, RerankIdentity, RerankRequest, RerankScores, Reranker,
    RetryAdvice,
};
use reqwest::header::HeaderValue;
use reqwest::{Client, StatusCode};
use serde::Deserialize;
use serde_json::json;

use crate::http::{RemoteEmbeddingConfigError, validate_endpoint};

/// `Voyage`'s reranking route.
pub const VOYAGE_RERANK_ENDPOINT: &str = "https://api.voyageai.com/v1/rerank";
/// Reranking model selected by default.
pub const VOYAGE_DEFAULT_RERANK_MODEL: &str = "rerank-2.5";

const MAX_RESPONSE_BYTES: u64 = 16 * 1024 * 1024;
/// Passages scored in one request.
///
/// Voyage accepts a thousand documents and six hundred thousand tokens per
/// request, so a candidate list of fifty fits in one call — unlike the hosted
/// Cloudflare reranker, which needed thirteen. The character bound is
/// pessimistic at one character per token: source code measured 1.41 characters
/// per token on this corpus, so 400k characters cannot approach the token
/// ceiling whatever the input looks like.
const MAX_PASSAGES_PER_REQUEST: usize = 128;
const MAX_CHARS_PER_REQUEST: usize = 400_000;

/// Configuration for the hosted `Voyage` reranker.
pub struct VoyageRerankerConfig {
    api_key: String,
    model: String,
    revision: String,
    endpoint: String,
    timeout: Duration,
}

impl VoyageRerankerConfig {
    /// Uses [`VOYAGE_DEFAULT_RERANK_MODEL`] and a 30-second request timeout.
    #[must_use]
    pub fn new(api_key: impl Into<String>, revision: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: VOYAGE_DEFAULT_RERANK_MODEL.into(),
            revision: revision.into(),
            endpoint: VOYAGE_RERANK_ENDPOINT.into(),
            timeout: Duration::from_secs(30),
        }
    }

    /// Selects a different reranking model.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Overrides the endpoint for contract tests.
    #[must_use]
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Overrides the complete HTTP request timeout.
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

impl fmt::Debug for VoyageRerankerConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VoyageRerankerConfig")
            .field("api_key", &"[REDACTED]")
            .field("model", &self.model)
            .field("revision", &self.revision)
            .field("endpoint", &self.endpoint)
            .field("timeout", &self.timeout)
            .finish()
    }
}

/// Scores query/passage pairs through `Voyage`'s reranking route.
#[derive(Debug)]
pub struct VoyageReranker {
    client: Client,
    endpoint: String,
    authorization: HeaderValue,
    model: String,
    identity: RerankIdentity,
}

impl VoyageReranker {
    /// Builds a reusable, thread-safe reranker.
    ///
    /// # Errors
    ///
    /// Returns [`RemoteEmbeddingConfigError`] for an invalid key, model,
    /// revision, endpoint, timeout, or HTTP client configuration.
    pub fn new(config: VoyageRerankerConfig) -> Result<Self, RemoteEmbeddingConfigError> {
        if config.timeout.is_zero() {
            return Err(RemoteEmbeddingConfigError::InvalidTimeout);
        }
        let model = config.model.trim().to_owned();
        if model.is_empty() {
            return Err(RemoteEmbeddingConfigError::InvalidModel);
        }
        let identity = RerankIdentity {
            provider: "voyage".into(),
            model: model.clone(),
            revision: config.revision.trim().into(),
        };
        identity
            .validate()
            .map_err(|_| RemoteEmbeddingConfigError::InvalidModel)?;
        validate_endpoint(&config.endpoint, "api.voyageai.com", "/v1/rerank")?;
        let key = config.api_key.trim();
        if key.is_empty() {
            return Err(RemoteEmbeddingConfigError::InvalidBearer);
        }
        let mut authorization = HeaderValue::from_str(&format!("Bearer {key}"))
            .map_err(|_| RemoteEmbeddingConfigError::InvalidBearer)?;
        authorization.set_sensitive(true);
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|_| RemoteEmbeddingConfigError::HttpClient)?;
        Ok(Self {
            client,
            endpoint: config.endpoint,
            authorization,
            model,
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

    fn body(&self, query: &str, passages: &[&str]) -> serde_json::Value {
        json!({
            "query": query,
            "documents": passages,
            "model": self.model,
            // Truncation is left enabled, the opposite of the embedding path.
            // There, silently shortening a document would change what the index
            // stores and make a vector unreproducible; here it only affects one
            // transient score, and refusing the request would lose the ranking
            // for an entire query because one chunk was long.
            "truncation": true,
        })
    }

    async fn score_window(&self, query: &str, passages: &[&str]) -> Result<Vec<f32>, IndexError> {
        let response = self
            .client
            .post(&self.endpoint)
            .header("authorization", self.authorization.clone())
            .json(&self.body(query, passages))
            .send()
            .await
            .map_err(|error| Self::transport_error(&error))?;
        let status = response.status();
        if !status.is_success() {
            return Err(Self::status_error(status));
        }
        let bytes = response
            .bytes()
            .await
            .map_err(|error| Self::transport_error(&error))?;
        if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_RESPONSE_BYTES {
            return Err(Self::error(
                IndexErrorKind::Embedding,
                "response_too_large",
                "Voyage reranker response exceeded the accepted size",
            ));
        }
        let parsed: RerankResponse = serde_json::from_slice(&bytes).map_err(|_| {
            Self::error(
                IndexErrorKind::Embedding,
                "invalid_json",
                "Voyage reranker returned invalid JSON",
            )
        })?;
        // Results come back ordered by score and identify each passage by its
        // request index, so caller order is restored by position rather than
        // trusted from the response.
        let mut scores = vec![f32::NAN; passages.len()];
        for entry in parsed.data {
            let slot = scores.get_mut(entry.index).ok_or_else(|| {
                Self::error(
                    IndexErrorKind::Embedding,
                    "index_out_of_range",
                    format!(
                        "Voyage reranker scored document {} of {}",
                        entry.index,
                        passages.len()
                    ),
                )
            })?;
            *slot = entry.relevance_score;
        }
        if let Some(position) = scores.iter().position(|score| score.is_nan()) {
            return Err(Self::error(
                IndexErrorKind::Embedding,
                "missing_score",
                format!("Voyage reranker left document {position} unscored"),
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
        Self::error(kind, "transport", "Voyage reranker request failed")
            .with_retry(RetryAdvice::Immediate)
    }

    fn status_error(status: StatusCode) -> IndexError {
        let code = match status {
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => "unauthorized",
            StatusCode::TOO_MANY_REQUESTS => "throttled",
            status if status.is_server_error() => "upstream_unavailable",
            _ => "request_rejected",
        };
        let error = Self::error(
            IndexErrorKind::Embedding,
            code,
            format!("Voyage reranker returned HTTP {status}"),
        );
        match code {
            "throttled" => error.with_retry(RetryAdvice::AfterMillis(non_zero_millis(2_000))),
            "upstream_unavailable" => error.with_retry(RetryAdvice::Immediate),
            _ => error,
        }
    }

    fn error(kind: IndexErrorKind, code: &str, message: impl Into<String>) -> IndexError {
        IndexError::new(kind, format!("voyage_reranker_{code}"), message)
    }
}

impl Reranker for VoyageReranker {
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
struct RerankResponse {
    data: Vec<RerankEntry>,
}

#[derive(Debug, Deserialize)]
struct RerankEntry {
    index: usize,
    relevance_score: f32,
}

/// Non-zero milliseconds for retry advice, with a safe floor.
fn non_zero_millis(milliseconds: u64) -> NonZeroU64 {
    NonZeroU64::new(milliseconds).unwrap_or(NonZeroU64::MIN)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter() -> VoyageReranker {
        VoyageReranker::new(VoyageRerankerConfig::new("key", "benchmark-revision")).unwrap()
    }

    #[test]
    fn the_request_names_the_model_and_every_document() {
        let body = adapter().body("where is retry configured", &["fn a() {}", "fn b() {}"]);
        assert_eq!(body["query"], "where is retry configured");
        assert_eq!(body["documents"][0], "fn a() {}");
        assert_eq!(body["documents"][1], "fn b() {}");
        assert_eq!(body["model"], VOYAGE_DEFAULT_RERANK_MODEL);
    }

    /// A long chunk must cost one truncated score, not the whole query's ranking.
    #[test]
    fn truncation_stays_enabled_for_scoring() {
        assert_eq!(adapter().body("q", &["text"])["truncation"], true);
    }

    /// Fifty candidates are the working case and must not be split at all.
    #[test]
    fn a_realistic_candidate_list_is_one_request() {
        let passages = vec!["a".repeat(1_500); 50];
        let borrowed = passages.iter().map(String::as_str).collect::<Vec<_>>();
        assert_eq!(VoyageReranker::windows(&borrowed), vec![(0, 50)]);
    }

    #[test]
    fn oversized_batches_are_split_on_both_bounds() {
        let many = vec!["x"; 300];
        assert_eq!(VoyageReranker::windows(&many).len(), 3);

        let long = "y".repeat(250_000);
        let heavy = vec![long.as_str(), long.as_str(), long.as_str()];
        assert_eq!(VoyageReranker::windows(&heavy).len(), 3);
    }

    #[test]
    fn one_oversized_passage_still_travels_alone() {
        let huge = "z".repeat(MAX_CHARS_PER_REQUEST * 2);
        assert_eq!(VoyageReranker::windows(&[huge.as_str()]), vec![(0, 1)]);
    }

    #[test]
    fn a_blank_revision_is_refused() {
        assert!(VoyageReranker::new(VoyageRerankerConfig::new("key", "  ")).is_err());
    }

    #[test]
    fn a_blank_key_is_refused() {
        assert!(VoyageReranker::new(VoyageRerankerConfig::new("   ", "revision")).is_err());
    }

    #[test]
    fn an_endpoint_on_another_host_is_refused() {
        assert!(
            VoyageReranker::new(
                VoyageRerankerConfig::new("key", "revision")
                    .with_endpoint("https://example.com/v1/rerank")
            )
            .is_err()
        );
    }

    #[test]
    fn identity_names_the_selected_model() {
        let adapter = VoyageReranker::new(
            VoyageRerankerConfig::new("key", "revision").with_model("rerank-2.5-lite"),
        )
        .unwrap();
        assert_eq!(adapter.identity().provider, "voyage");
        assert_eq!(adapter.identity().model, "rerank-2.5-lite");
    }
}
