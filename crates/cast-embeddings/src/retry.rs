use std::future::Future;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use cast_index::{
    BoxFuture, Embedder, EmbeddingIdentity, EmbeddingInput, EmbeddingVector, IndexError,
    RerankIdentity, RerankRequest, RerankScores, Reranker, RetryAdvice,
};
use thiserror::Error;

const DEFAULT_MAX_ATTEMPTS: usize = 4;
const MAX_ATTEMPTS: usize = 10;
const DEFAULT_MAX_DELAY: Duration = Duration::from_secs(30);
const DEFAULT_TOTAL_BUDGET: Duration = Duration::from_secs(90);
const JITTER_PERCENT: u128 = 20;

/// Bounded retry policy applied by embedding orchestration, not providers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetryPolicy {
    max_attempts: NonZeroUsize,
    max_delay: Duration,
    total_budget: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: NonZeroUsize::new(DEFAULT_MAX_ATTEMPTS).unwrap_or(NonZeroUsize::MIN),
            max_delay: DEFAULT_MAX_DELAY,
            total_budget: DEFAULT_TOTAL_BUDGET,
        }
    }
}

impl RetryPolicy {
    /// Creates the default delay/budget policy with a selected attempt count.
    ///
    /// # Errors
    ///
    /// Returns [`RetryPolicyError`] unless attempts are between 1 and 10.
    pub fn with_max_attempts(max_attempts: usize) -> Result<Self, RetryPolicyError> {
        let max_attempts = NonZeroUsize::new(max_attempts)
            .filter(|attempts| attempts.get() <= MAX_ATTEMPTS)
            .ok_or(RetryPolicyError::InvalidAttempts)?;
        Ok(Self {
            max_attempts,
            ..Self::default()
        })
    }

    /// Creates a fully bounded policy, primarily for service tuning and tests.
    ///
    /// # Errors
    ///
    /// Returns [`RetryPolicyError`] for an invalid attempt count or zero delay
    /// or elapsed-time bounds.
    pub fn bounded(
        max_attempts: usize,
        max_delay: Duration,
        total_budget: Duration,
    ) -> Result<Self, RetryPolicyError> {
        let mut policy = Self::with_max_attempts(max_attempts)?;
        if max_delay.is_zero() {
            return Err(RetryPolicyError::InvalidMaxDelay);
        }
        if total_budget.is_zero() {
            return Err(RetryPolicyError::InvalidTotalBudget);
        }
        policy.max_delay = max_delay;
        policy.total_budget = total_budget;
        Ok(policy)
    }
}

/// Invalid retry-orchestration configuration.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum RetryPolicyError {
    /// Attempts are bounded to prevent accidental infinite loops.
    #[error("embedding max attempts must be between 1 and 10")]
    InvalidAttempts,
    /// A zero delay cap can turn delayed retries into a hot loop.
    #[error("embedding maximum retry delay must be greater than zero")]
    InvalidMaxDelay,
    /// A zero total budget cannot permit delayed retries.
    #[error("embedding total retry budget must be greater than zero")]
    InvalidTotalBudget,
}

/// Provider-neutral [`Embedder`] decorator that executes typed retry advice.
///
/// The inner provider remains responsible only for classifying an error as
/// permanent, immediate, or delayed. This orchestration layer owns attempt
/// count, capped jitter, and a total elapsed-time budget.
pub struct RetryingEmbedder<E> {
    inner: E,
    policy: RetryPolicy,
}

impl<E> RetryingEmbedder<E> {
    /// Wraps an embedder with the supplied validated retry policy.
    #[must_use]
    pub const fn new(inner: E, policy: RetryPolicy) -> Self {
        Self { inner, policy }
    }

    /// Returns the wrapped provider adapter.
    #[must_use]
    pub fn into_inner(self) -> E {
        self.inner
    }
}

impl<E: Embedder> Embedder for RetryingEmbedder<E> {
    fn identity(&self) -> &EmbeddingIdentity {
        self.inner.identity()
    }

    fn embed_batch<'a>(
        &'a self,
        inputs: &'a [EmbeddingInput<'a>],
    ) -> BoxFuture<'a, Result<Vec<EmbeddingVector>, IndexError>> {
        Box::pin(
            async move { retry_operation(|| self.inner.embed_batch(inputs), self.policy).await },
        )
    }

    fn embed_query<'a>(
        &'a self,
        text: &'a str,
    ) -> BoxFuture<'a, Result<EmbeddingVector, IndexError>> {
        Box::pin(async move { retry_operation(|| self.inner.embed_query(text), self.policy).await })
    }
}

/// Applies a retry policy to a reranker.
///
/// A reranker is one request per query, so a single 429 loses that query's whole
/// ranking. The embedding path already owned attempts through
/// [`RetryingEmbedder`]; this is the same policy for the same reason.
pub struct RetryingReranker<R> {
    inner: R,
    policy: RetryPolicy,
}

impl<R> RetryingReranker<R> {
    /// Wraps a reranker with the supplied validated retry policy.
    #[must_use]
    pub const fn new(inner: R, policy: RetryPolicy) -> Self {
        Self { inner, policy }
    }

    /// Returns the wrapped provider adapter.
    #[must_use]
    pub fn into_inner(self) -> R {
        self.inner
    }
}

impl<R: Reranker> Reranker for RetryingReranker<R> {
    fn identity(&self) -> &RerankIdentity {
        self.inner.identity()
    }

    fn rerank<'a>(
        &'a self,
        request: RerankRequest<'a>,
    ) -> BoxFuture<'a, Result<RerankScores, IndexError>> {
        Box::pin(async move { retry_operation(|| self.inner.rerank(request), self.policy).await })
    }
}

async fn retry_operation<T, F, Fut>(mut operation: F, policy: RetryPolicy) -> Result<T, IndexError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, IndexError>>,
{
    let started = Instant::now();
    let mut attempt = 1_usize;
    loop {
        match operation().await {
            Ok(value) => return Ok(value),
            Err(error) => {
                let Some(delay) = retry_delay(error.retry, attempt, policy) else {
                    return Err(error);
                };
                if started.elapsed().saturating_add(delay) > policy.total_budget {
                    return Err(error);
                }
                if delay.is_zero() {
                    tokio::task::yield_now().await;
                } else {
                    tokio::time::sleep(delay).await;
                }
                attempt = attempt.saturating_add(1);
            }
        }
    }
}

fn retry_delay(advice: RetryAdvice, attempt: usize, policy: RetryPolicy) -> Option<Duration> {
    if attempt >= policy.max_attempts.get() {
        return None;
    }
    match advice {
        RetryAdvice::Never => None,
        RetryAdvice::Immediate => Some(Duration::ZERO),
        RetryAdvice::AfterMillis(milliseconds) => {
            let base = Duration::from_millis(milliseconds.get()).min(policy.max_delay);
            Some(jitter(base, attempt).min(policy.max_delay))
        }
    }
}

fn jitter(base: Duration, attempt: usize) -> Duration {
    let ceiling = base.as_millis().saturating_mul(JITTER_PERCENT) / 100;
    if ceiling == 0 {
        return base;
    }
    let clock = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let salt = clock ^ u128::from(std::process::id()) ^ attempt as u128;
    let extra = salt % ceiling.saturating_add(1);
    base.saturating_add(Duration::from_millis(
        u64::try_from(extra).unwrap_or(u64::MAX),
    ))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use cast_index::{IndexErrorKind, RetryAdvice};

    use super::*;

    #[test]
    fn policy_rejects_unbounded_or_zero_configuration() {
        assert_eq!(
            RetryPolicy::with_max_attempts(0),
            Err(RetryPolicyError::InvalidAttempts)
        );
        assert_eq!(
            RetryPolicy::with_max_attempts(MAX_ATTEMPTS + 1),
            Err(RetryPolicyError::InvalidAttempts)
        );
        assert_eq!(
            RetryPolicy::bounded(2, Duration::ZERO, Duration::from_secs(1)),
            Err(RetryPolicyError::InvalidMaxDelay)
        );
    }

    #[tokio::test]
    async fn transient_operation_is_retried_and_then_succeeds() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let observed = attempts.clone();
        let value = retry_operation(
            move || {
                let attempt = observed.fetch_add(1, Ordering::SeqCst);
                async move {
                    if attempt == 0 {
                        Err(
                            IndexError::new(IndexErrorKind::Embedding, "rate_limited", "retry")
                                .with_retry(RetryAdvice::Immediate),
                        )
                    } else {
                        Ok(42_u8)
                    }
                }
            },
            RetryPolicy::bounded(2, Duration::from_millis(1), Duration::from_secs(1)).unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(value, 42);
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn permanent_error_and_attempt_limit_stop_retries() {
        let attempts = AtomicUsize::new(0);
        let error = retry_operation(
            || {
                attempts.fetch_add(1, Ordering::SeqCst);
                async {
                    Err::<(), _>(IndexError::new(
                        IndexErrorKind::Configuration,
                        "auth",
                        "permanent",
                    ))
                }
            },
            RetryPolicy::default(),
        )
        .await
        .unwrap_err();
        assert_eq!(error.code, "auth");
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }
}
