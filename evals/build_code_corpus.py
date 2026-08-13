#!/usr/bin/env python3
"""Builds the code evaluation set from authored concept clusters.

Why a generator: the artifact is JSONL with embedded source, which is unreadable
and unmaintainable by hand. The authored content lives here, in one reviewable
place, and `corpus.jsonl` plus `queries.jsonl` are regenerated from it.

Why clusters: the frozen seed set judges 1.41 documents per query out of 60, so
nDCG@10 is decided almost entirely by whether one gold document lands first.
That cannot separate two competent models. Each cluster here implements one
concept in several languages and adds a lexical near-miss — a document sharing
the concept's vocabulary while doing something else. Queries then grade the
exact answer 3, the same concept in another language 2, and the adjacent or
near-miss document 1, so a model is measured on ordering rather than on a single
hit.

Every fixture is original, compact, and synthetic. Repository source is never
copied in: CONTRIBUTING forbids it, and licence-mixed corpora cannot be
redistributed with the tests.

Regenerate with:

    python3 evals/build_code_corpus.py
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

EVALS = Path(__file__).resolve().parent
# The frozen prose set is an input, never edited: `evals/seed-v1/` reproduces
# every number recorded before this set existed.
FROZEN_CORPUS = EVALS / "seed-v1" / "corpus.jsonl"


def code(text: str) -> str:
    """Normalizes an indented fixture into file-like source."""
    return dedent(text).strip("\n") + "\n"


# Each cluster: an id, its documents, and the queries judged against them.
# Document ids are stable, human-chosen strings so judgments survive any change
# to chunking or hashing.
CLUSTERS: list[dict] = [
    {
        "id": "http-retry-backoff",
        "docs": [
            {
                "id": "retry-backoff-rust",
                "path": "src/transport/retry.rs",
                "language": "rust",
                "text": code(
                    """
                    /// Retries a request with exponential backoff and jitter.
                    ///
                    /// Only transport failures and 429/5xx responses are retried;
                    /// a 4xx other than 429 is permanent and returns immediately.
                    pub async fn send_with_retry(
                        client: &Client,
                        request: Request,
                        max_attempts: u32,
                    ) -> Result<Response, TransportError> {
                        let mut delay = Duration::from_millis(200);
                        for attempt in 1..=max_attempts {
                            match client.execute(request.try_clone().unwrap()).await {
                                Ok(response) if response.status().is_success() => return Ok(response),
                                Ok(response) if !is_retryable(response.status()) => {
                                    return Err(TransportError::Permanent(response.status()));
                                }
                                Ok(_) | Err(_) if attempt == max_attempts => break,
                                _ => {}
                            }
                            sleep(delay + jitter(delay)).await;
                            delay = (delay * 2).min(Duration::from_secs(30));
                        }
                        Err(TransportError::Exhausted { attempts: max_attempts })
                    }
                    """
                ),
            },
            {
                "id": "retry-backoff-python",
                "path": "clients/python/uploader.py",
                "language": "python",
                "text": code(
                    """
                    def upload_with_retry(session, url, payload, max_attempts=4):
                        \"\"\"Uploads a payload, retrying throttled and server errors.

                        Honors Retry-After when the server sends it, capping each
                        individual sleep at 30 seconds so a hostile header cannot
                        stall the worker indefinitely.
                        \"\"\"
                        delay = 0.2
                        for attempt in range(1, max_attempts + 1):
                            response = session.post(url, json=payload)
                            if response.ok:
                                return response
                            if response.status_code not in RETRYABLE_STATUS:
                                response.raise_for_status()
                            if attempt == max_attempts:
                                break
                            hinted = response.headers.get("Retry-After")
                            time.sleep(min(float(hinted), 30.0) if hinted else delay)
                            delay = min(delay * 2, 30.0)
                        raise UploadExhausted(url, max_attempts)
                    """
                ),
            },
            {
                "id": "retry-backoff-go",
                "path": "internal/httpx/retry.go",
                "language": "go",
                "text": code(
                    """
                    // Do issues the request, retrying transient failures with
                    // exponential backoff. The context deadline always wins: a
                    // cancelled context stops retrying immediately.
                    func (c *Client) Do(ctx context.Context, req *http.Request) (*http.Response, error) {
                        delay := 200 * time.Millisecond
                        for attempt := 1; attempt <= c.maxAttempts; attempt++ {
                            resp, err := c.inner.Do(req.WithContext(ctx))
                            if err == nil && resp.StatusCode < 500 && resp.StatusCode != 429 {
                                return resp, nil
                            }
                            if attempt == c.maxAttempts {
                                return nil, fmt.Errorf("exhausted %d attempts: %w", attempt, err)
                            }
                            select {
                            case <-ctx.Done():
                                return nil, ctx.Err()
                            case <-time.After(delay + jitter(delay)):
                            }
                            delay = min(delay*2, 30*time.Second)
                        }
                        return nil, errRetryExhausted
                    }
                    """
                ),
            },
            {
                # Lexical near-miss: "retry" vocabulary, unrelated concern.
                "id": "retry-button-toast",
                "path": "web/components/UploadToast.tsx",
                "language": "typescript",
                "text": code(
                    """
                    // Shows a failed-upload toast with a Retry button. The retry
                    // itself is delegated to the caller; this component only owns
                    // presentation and the dismiss timer.
                    export function UploadToast({ error, onRetry, onDismiss }: Props) {
                      useEffect(() => {
                        const timer = setTimeout(onDismiss, 8000);
                        return () => clearTimeout(timer);
                      }, [onDismiss]);
                      return (
                        <div role="alert" className="toast toast--error">
                          <span>{error.message}</span>
                          <button type="button" onClick={onRetry}>
                            Retry upload
                          </button>
                        </div>
                      );
                    }
                    """
                ),
            },
        ],
        "queries": [
            {
                "query": "how do we back off and retry a throttled HTTP request",
                "grades": {
                    "retry-backoff-rust": 3,
                    "retry-backoff-python": 3,
                    "retry-backoff-go": 3,
                    "retry-button-toast": 1,
                },
            },
            {
                "query": "does the python uploader respect the Retry-After header",
                "grades": {"retry-backoff-python": 3, "retry-backoff-go": 1},
            },
            {
                "query": "stop retrying when the context is cancelled",
                "grades": {"retry-backoff-go": 3, "retry-backoff-rust": 1},
            },
            {
                "query": "which status codes are permanent and must not be retried",
                "grades": {
                    "retry-backoff-rust": 3,
                    "retry-backoff-python": 2,
                    "retry-backoff-go": 2,
                },
            },
        ],
    },
    {
        "id": "token-bucket-rate-limit",
        "docs": [
            {
                "id": "token-bucket-go",
                "path": "internal/limit/bucket.go",
                "language": "go",
                "text": code(
                    """
                    // Allow reports whether one unit of work may proceed now.
                    // Tokens refill continuously from elapsed time rather than on a
                    // ticker, so an idle limiter does not accumulate a burst beyond
                    // its configured capacity.
                    func (b *Bucket) Allow() bool {
                        b.mu.Lock()
                        defer b.mu.Unlock()
                        now := time.Now()
                        b.tokens = math.Min(
                            b.capacity,
                            b.tokens+now.Sub(b.last).Seconds()*b.refillPerSecond,
                        )
                        b.last = now
                        if b.tokens < 1 {
                            return false
                        }
                        b.tokens--
                        return true
                    }
                    """
                ),
            },
            {
                "id": "token-bucket-python",
                "path": "server/throttle.py",
                "language": "python",
                "text": code(
                    """
                    class TokenBucket:
                        \"\"\"Per-tenant request admission with a burst allowance.

                        capacity is the burst size; rate is the sustained refill in
                        tokens per second. take() never blocks: callers decide
                        whether to queue, shed, or return 429.
                        \"\"\"

                        def __init__(self, capacity: float, rate: float) -> None:
                            self.capacity = capacity
                            self.rate = rate
                            self.tokens = capacity
                            self.updated = time.monotonic()

                        def take(self, cost: float = 1.0) -> bool:
                            now = time.monotonic()
                            self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                            self.updated = now
                            if self.tokens < cost:
                                return False
                            self.tokens -= cost
                            return True
                    """
                ),
            },
            {
                "id": "token-bucket-rust",
                "path": "src/admission/limiter.rs",
                "language": "rust",
                "text": code(
                    """
                    /// Fixed-window counter used for coarse per-IP admission.
                    ///
                    /// Cheaper than a token bucket but allows a double burst across
                    /// a window boundary, which is acceptable for abuse control and
                    /// not for quota enforcement.
                    impl WindowLimiter {
                        pub fn admit(&self, key: &IpAddr) -> Admission {
                            let window = self.clock.now().duration_since(self.epoch).as_secs() / self.window_secs;
                            let mut counters = self.counters.lock();
                            let counter = counters.entry((*key, window)).or_insert(0);
                            *counter += 1;
                            if *counter > self.max_per_window {
                                Admission::Rejected { retry_after: self.window_secs }
                            } else {
                                Admission::Allowed
                            }
                        }
                    }
                    """
                ),
            },
        ],
        "queries": [
            {
                "query": "rate limit requests with a burst allowance that refills over time",
                "grades": {
                    "token-bucket-go": 3,
                    "token-bucket-python": 3,
                    "token-bucket-rust": 1,
                },
            },
            {
                "query": "why does an idle limiter not build up an unlimited burst",
                "grades": {"token-bucket-go": 3, "token-bucket-python": 2},
            },
            {
                "query": "coarse per-IP throttling for abuse control",
                "grades": {"token-bucket-rust": 3, "token-bucket-go": 1},
            },
        ],
    },
    {
        "id": "jwt-verify-expiry",
        "docs": [
            {
                "id": "jwt-verify-typescript",
                "path": "api/auth/verifyToken.ts",
                "language": "typescript",
                "text": code(
                    """
                    // Verifies a bearer token: signature first, then claims.
                    // Expiry is checked with a small clock-skew allowance because
                    // issuer and verifier clocks are never exactly aligned.
                    export async function verifyToken(token: string, jwks: JwkSet): Promise<Claims> {
                      const { header, payload, signature } = decodeJwt(token);
                      const key = jwks.find((candidate) => candidate.kid === header.kid);
                      if (!key) throw new AuthError("unknown key id");
                      if (!(await verifySignature(header, payload, signature, key))) {
                        throw new AuthError("signature mismatch");
                      }
                      const now = Math.floor(Date.now() / 1000);
                      if (payload.exp !== undefined && payload.exp + CLOCK_SKEW_SECONDS < now) {
                        throw new AuthError("token expired");
                      }
                      if (payload.nbf !== undefined && payload.nbf - CLOCK_SKEW_SECONDS > now) {
                        throw new AuthError("token not yet valid");
                      }
                      return payload;
                    }
                    """
                ),
            },
            {
                "id": "jwt-verify-php",
                "path": "src/Auth/TokenValidator.php",
                "language": "php",
                "text": code(
                    """
                    <?php
                    /**
                     * Validates a signed session token.
                     *
                     * Signature comparison is constant time; a mismatch and an
                     * expired token return the same generic failure so the caller
                     * cannot distinguish them from timing or message.
                     */
                    final class TokenValidator
                    {
                        public function validate(string $token): Claims
                        {
                            [$head, $body, $mac] = explode('.', $token) + [null, null, null];
                            $expected = hash_hmac('sha256', "$head.$body", $this->secret, true);
                            if ($mac === null || !hash_equals($expected, self::base64UrlDecode($mac))) {
                                throw new AuthFailure('invalid token');
                            }
                            $claims = json_decode(self::base64UrlDecode($body), true);
                            if (($claims['exp'] ?? 0) < time() - self::SKEW) {
                                throw new AuthFailure('invalid token');
                            }
                            return Claims::fromArray($claims);
                        }
                    }
                    """
                ),
            },
            {
                "id": "session-cookie-rotate-python",
                "path": "server/session.py",
                "language": "python",
                "text": code(
                    """
                    def rotate_session(response, session, *, ttl=timedelta(hours=12)):
                        \"\"\"Issues a fresh session cookie and invalidates the old id.

                        Rotation happens on privilege change, not on every request, so
                        a stolen cookie stops working once the victim re-authenticates.
                        \"\"\"
                        store.revoke(session.id)
                        fresh = store.create(session.user_id, expires_at=utcnow() + ttl)
                        response.set_cookie(
                            "sid",
                            fresh.id,
                            max_age=int(ttl.total_seconds()),
                            httponly=True,
                            secure=True,
                            samesite="Lax",
                        )
                        return fresh
                    """
                ),
            },
        ],
        "queries": [
            {
                "query": "validate a JWT signature and reject expired tokens",
                "grades": {"jwt-verify-typescript": 3, "jwt-verify-php": 3},
            },
            {
                "query": "why allow clock skew when checking token expiry",
                "grades": {"jwt-verify-typescript": 3, "jwt-verify-php": 2},
            },
            {
                "query": "avoid leaking whether a token was forged or merely expired",
                "grades": {"jwt-verify-php": 3, "jwt-verify-typescript": 1},
            },
            {
                "query": "set a secure httponly session cookie",
                "grades": {"session-cookie-rotate-python": 3},
            },
        ],
    },
    {
        "id": "cursor-pagination",
        "docs": [
            {
                "id": "cursor-pagination-python",
                "path": "api/pagination.py",
                "language": "python",
                "text": code(
                    """
                    def page_after(query, cursor: str | None, limit: int = 50):
                        \"\"\"Keyset pagination over (created_at, id).

                        Offset pagination drifts when rows are inserted mid-scan, so
                        the cursor encodes the last row's sort key and the next page
                        seeks past it. The composite key breaks ties on identical
                        timestamps, without which rows can repeat or vanish.
                        \"\"\"
                        if cursor:
                            created_at, last_id = decode_cursor(cursor)
                            query = query.filter(
                                or_(
                                    Row.created_at < created_at,
                                    and_(Row.created_at == created_at, Row.id < last_id),
                                )
                            )
                        rows = query.order_by(Row.created_at.desc(), Row.id.desc()).limit(limit + 1).all()
                        has_more = len(rows) > limit
                        rows = rows[:limit]
                        return rows, encode_cursor(rows[-1]) if has_more and rows else None
                    """
                ),
            },
            {
                "id": "cursor-pagination-typescript",
                "path": "web/src/api/useInfiniteFeed.ts",
                "language": "typescript",
                "text": code(
                    """
                    // Loads successive feed pages by cursor. The cursor is opaque to
                    // the client: treating it as a page number breaks as soon as the
                    // server changes its sort key.
                    export function useInfiniteFeed(pageSize = 50) {
                      const [pages, setPages] = useState<Page[]>([]);
                      const [cursor, setCursor] = useState<string | null>(null);
                      const [done, setDone] = useState(false);

                      const loadMore = useCallback(async () => {
                        if (done) return;
                        const params = new URLSearchParams({ limit: String(pageSize) });
                        if (cursor) params.set("after", cursor);
                        const page = await fetchJson<Page>(`/api/feed?${params}`);
                        setPages((current) => [...current, page]);
                        setCursor(page.nextCursor);
                        setDone(page.nextCursor === null);
                      }, [cursor, done, pageSize]);

                      return { pages, loadMore, done };
                    }
                    """
                ),
            },
            {
                "id": "offset-pagination-go",
                "path": "internal/store/list.go",
                "language": "go",
                "text": code(
                    """
                    // ListPage returns one page using LIMIT/OFFSET. Retained for the
                    // admin export, which runs inside a repeatable-read transaction
                    // where the drift that makes offsets unsafe cannot occur.
                    func (s *Store) ListPage(ctx context.Context, page, size int) ([]Row, error) {
                        if page < 1 || size < 1 || size > maxPageSize {
                            return nil, ErrInvalidPage
                        }
                        rows, err := s.db.QueryContext(ctx,
                            `SELECT id, created_at, body FROM rows ORDER BY created_at DESC, id DESC LIMIT $1 OFFSET $2`,
                            size, (page-1)*size)
                        if err != nil {
                            return nil, err
                        }
                        defer rows.Close()
                        return scanRows(rows)
                    }
                    """
                ),
            },
        ],
        "queries": [
            {
                "query": "paginate a list without skipping or repeating rows when new ones arrive",
                "grades": {
                    "cursor-pagination-python": 3,
                    "cursor-pagination-typescript": 2,
                    "offset-pagination-go": 1,
                },
            },
            {
                "query": "why does the cursor include the id as well as the timestamp",
                "grades": {"cursor-pagination-python": 3},
            },
            {
                "query": "load the next page of an infinite scrolling feed",
                "grades": {
                    "cursor-pagination-typescript": 3,
                    "cursor-pagination-python": 1,
                },
            },
            {
                "query": "when is LIMIT OFFSET paging still acceptable",
                "grades": {"offset-pagination-go": 3, "cursor-pagination-python": 1},
            },
        ],
    },
    {
        "id": "lru-cache-eviction",
        "docs": [
            {
                "id": "lru-cache-rust",
                "path": "src/cache/lru.rs",
                "language": "rust",
                "text": code(
                    """
                    /// Bounded cache evicting the least recently used entry.
                    ///
                    /// get() promotes the key, so a scan of cold keys cannot evict the
                    /// working set in one pass any further than capacity allows.
                    impl<K: Hash + Eq + Clone, V> LruCache<K, V> {
                        pub fn get(&mut self, key: &K) -> Option<&V> {
                            let entry = self.map.get(key)?;
                            self.order.promote(entry.token);
                            Some(&entry.value)
                        }

                        pub fn put(&mut self, key: K, value: V) -> Option<V> {
                            if let Some(existing) = self.map.get_mut(&key) {
                                self.order.promote(existing.token);
                                return Some(std::mem::replace(&mut existing.value, value));
                            }
                            if self.map.len() == self.capacity {
                                if let Some(victim) = self.order.pop_back() {
                                    self.map.remove(&victim);
                                }
                            }
                            let token = self.order.push_front(key.clone());
                            self.map.insert(key, Entry { value, token });
                            None
                        }
                    }
                    """
                ),
            },
            {
                "id": "lru-cache-java",
                "path": "src/main/java/cache/BoundedCache.java",
                "language": "java",
                "text": code(
                    """
                    /**
                     * Access-ordered cache with a hard entry ceiling.
                     *
                     * LinkedHashMap already implements the promotion and eviction
                     * order; only removeEldestEntry needs overriding. Wrapped in a
                     * synchronized view because callers share one instance per JVM.
                     */
                    public final class BoundedCache<K, V> {
                        private final Map<K, V> entries;

                        public BoundedCache(int capacity) {
                            this.entries = Collections.synchronizedMap(
                                new LinkedHashMap<>(capacity, 0.75f, true) {
                                    @Override
                                    protected boolean removeEldestEntry(Map.Entry<K, V> eldest) {
                                        return size() > capacity;
                                    }
                                });
                        }

                        public V get(K key) { return entries.get(key); }
                        public void put(K key, V value) { entries.put(key, value); }
                    }
                    """
                ),
            },
            {
                "id": "ttl-cache-python",
                "path": "server/cache.py",
                "language": "python",
                "text": code(
                    """
                    class TtlCache:
                        \"\"\"Time-expiring cache with no size bound.

                        Entries leave on age, never on pressure, so this is only safe
                        for a key space that is small and known. Use the LRU cache
                        when the key space is unbounded, such as per-request URLs.
                        \"\"\"

                        def get(self, key):
                            entry = self._entries.get(key)
                            if entry is None:
                                return None
                            if entry.expires_at <= time.monotonic():
                                del self._entries[key]
                                return None
                            return entry.value

                        def put(self, key, value, ttl_seconds):
                            self._entries[key] = Entry(value, time.monotonic() + ttl_seconds)
                    """
                ),
            },
        ],
        "queries": [
            {
                "query": "evict the least recently used entry when the cache is full",
                "grades": {"lru-cache-rust": 3, "lru-cache-java": 3, "ttl-cache-python": 1},
            },
            {
                "query": "cache entries that should expire after a while rather than on memory pressure",
                "grades": {"ttl-cache-python": 3, "lru-cache-rust": 1},
            },
            {
                "query": "use LinkedHashMap access order to bound a cache",
                "grades": {"lru-cache-java": 3},
            },
        ],
    },
    {
        "id": "graceful-shutdown-drain",
        "docs": [
            {
                "id": "graceful-shutdown-go",
                "path": "cmd/server/shutdown.go",
                "language": "go",
                "text": code(
                    """
                    // Run serves until SIGTERM, then stops accepting connections and
                    // waits for in-flight requests up to the drain timeout. Exceeding
                    // the timeout is reported rather than silently dropping work, so
                    // an operator learns the deadline is too short.
                    func Run(ctx context.Context, srv *http.Server, drain time.Duration) error {
                        signals := make(chan os.Signal, 1)
                        signal.Notify(signals, syscall.SIGTERM, syscall.SIGINT)
                        errs := make(chan error, 1)
                        go func() { errs <- srv.ListenAndServe() }()

                        select {
                        case err := <-errs:
                            return err
                        case <-signals:
                        }
                        shutdownCtx, cancel := context.WithTimeout(context.Background(), drain)
                        defer cancel()
                        if err := srv.Shutdown(shutdownCtx); err != nil {
                            return fmt.Errorf("drain exceeded %s: %w", drain, err)
                        }
                        return nil
                    }
                    """
                ),
            },
            {
                "id": "graceful-shutdown-rust",
                "path": "src/bin/worker.rs",
                "language": "rust",
                "text": code(
                    """
                    /// Consumes the queue until a shutdown signal, then finishes the
                    /// message in flight and commits its offset before returning.
                    /// Committing after the handler, never before, is what makes a
                    /// crash mid-message a redelivery rather than a silent loss.
                    async fn consume(mut queue: Queue, mut shutdown: Receiver<()>) -> Result<()> {
                        loop {
                            let message = tokio::select! {
                                message = queue.next() => match message {
                                    Some(message) => message,
                                    None => break,
                                },
                                _ = shutdown.recv() => break,
                            };
                            let offset = message.offset;
                            handle(message).await?;
                            queue.commit(offset).await?;
                        }
                        queue.flush().await
                    }
                    """
                ),
            },
            {
                "id": "shutdown-hook-java",
                "path": "src/main/java/app/Lifecycle.java",
                "language": "java",
                "text": code(
                    """
                    /**
                     * Registers an orderly stop for the executor pool.
                     *
                     * shutdown() refuses new tasks and lets queued ones finish;
                     * shutdownNow() is only reached after the grace period, and its
                     * dropped task list is logged because those units of work never ran.
                     */
                    public void registerShutdownHook(ExecutorService pool, Duration grace) {
                        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                            pool.shutdown();
                            try {
                                if (!pool.awaitTermination(grace.toMillis(), TimeUnit.MILLISECONDS)) {
                                    List<Runnable> dropped = pool.shutdownNow();
                                    log.warn("dropped {} queued tasks after {}", dropped.size(), grace);
                                }
                            } catch (InterruptedException e) {
                                pool.shutdownNow();
                                Thread.currentThread().interrupt();
                            }
                        }));
                    }
                    """
                ),
            },
        ],
        "queries": [
            {
                "query": "finish in-flight requests on SIGTERM before exiting",
                "grades": {
                    "graceful-shutdown-go": 3,
                    "shutdown-hook-java": 2,
                    "graceful-shutdown-rust": 2,
                },
            },
            {
                "query": "why commit the queue offset after handling instead of before",
                "grades": {"graceful-shutdown-rust": 3},
            },
            {
                "query": "log the tasks discarded when a thread pool is forced to stop",
                "grades": {"shutdown-hook-java": 3, "graceful-shutdown-go": 1},
            },
        ],
    },
    {
        "id": "utf8-safe-truncate",
        "docs": [
            {
                "id": "utf8-truncate-rust",
                "path": "src/text/truncate.rs",
                "language": "rust",
                "text": code(
                    """
                    /// Truncates to at most `max_bytes` without splitting a character.
                    ///
                    /// Slicing a &str at an arbitrary byte index panics on a
                    /// multi-byte boundary, so the cut walks back to the nearest char
                    /// boundary. Grapheme clusters are still splittable: an emoji with
                    /// a skin-tone modifier can lose the modifier.
                    pub fn truncate_bytes(text: &str, max_bytes: usize) -> &str {
                        if text.len() <= max_bytes {
                            return text;
                        }
                        let mut end = max_bytes;
                        while end > 0 && !text.is_char_boundary(end) {
                            end -= 1;
                        }
                        &text[..end]
                    }
                    """
                ),
            },
            {
                "id": "utf8-truncate-python",
                "path": "lib/text.py",
                "language": "python",
                "text": code(
                    """
                    def truncate_utf8(text: str, max_bytes: int) -> str:
                        \"\"\"Trims text so its UTF-8 encoding fits max_bytes.

                        Python indexes by code point, so the byte budget has to be
                        applied to the encoded form and then decoded back with errors
                        ignored, which discards a trailing partial sequence.
                        \"\"\"
                        encoded = text.encode("utf-8")
                        if len(encoded) <= max_bytes:
                            return text
                        return encoded[:max_bytes].decode("utf-8", errors="ignore")
                    """
                ),
            },
            {
                "id": "utf8-validate-go",
                "path": "internal/text/validate.go",
                "language": "go",
                "text": code(
                    """
                    // SanitizeName rejects names that are not valid UTF-8 rather than
                    // repairing them. Replacing invalid bytes would let two different
                    // inputs collapse to one stored name, which downstream code treats
                    // as a uniqueness violation.
                    func SanitizeName(raw []byte) (string, error) {
                        if !utf8.Valid(raw) {
                            return "", ErrNotUTF8
                        }
                        name := strings.TrimSpace(string(raw))
                        if name == "" {
                            return "", ErrEmptyName
                        }
                        if utf8.RuneCountInString(name) > maxNameRunes {
                            return "", ErrNameTooLong
                        }
                        return name, nil
                    }
                    """
                ),
            },
        ],
        "queries": [
            {
                "query": "cut a string to a byte limit without breaking a multi-byte character",
                "grades": {
                    "utf8-truncate-rust": 3,
                    "utf8-truncate-python": 3,
                    "utf8-validate-go": 1,
                },
            },
            {
                "query": "why reject invalid UTF-8 instead of replacing the bad bytes",
                "grades": {"utf8-validate-go": 3},
            },
            {
                "query": "slicing a str panics on a character boundary",
                "grades": {"utf8-truncate-rust": 3, "utf8-truncate-python": 1},
            },
        ],
    },
    {
        "id": "checksum-verify",
        "docs": [
            {
                "id": "checksum-verify-rust",
                "path": "src/artifact/verify.rs",
                "language": "rust",
                "text": code(
                    """
                    /// Streams a file and compares its digest to the pinned value.
                    ///
                    /// The comparison is constant time and the file is read in blocks
                    /// so a multi-gigabyte artifact never has to be held in memory.
                    /// A mismatch deletes the temporary file: a half-trusted artifact
                    /// on disk is worse than no artifact.
                    pub fn verify_pinned(path: &Path, expected_hex: &str) -> Result<(), VerifyError> {
                        let mut file = File::open(path)?;
                        let mut context = Context::new(&SHA256);
                        let mut buffer = vec![0u8; 4 * 1024 * 1024];
                        loop {
                            let read = file.read(&mut buffer)?;
                            if read == 0 {
                                break;
                            }
                            context.update(&buffer[..read]);
                        }
                        let actual = hex(context.finish().as_ref());
                        if !constant_time_eq(actual.as_bytes(), expected_hex.as_bytes()) {
                            let _ = std::fs::remove_file(path);
                            return Err(VerifyError::Digest { expected: expected_hex.into(), actual });
                        }
                        Ok(())
                    }
                    """
                ),
            },
            {
                "id": "checksum-verify-python",
                "path": "tools/fetch_model.py",
                "language": "python",
                "text": code(
                    """
                    def download_verified(url: str, destination: Path, sha256: str, size: int) -> Path:
                        \"\"\"Downloads to a temporary file and promotes it only if both
                        the byte length and the digest match the pinned values.

                        Checking length as well as digest catches a truncated transfer
                        before the hash is computed over the whole file.
                        \"\"\"
                        temporary = destination.with_suffix(".partial")
                        digest = hashlib.sha256()
                        written = 0
                        with requests.get(url, stream=True, timeout=60) as response:
                            response.raise_for_status()
                            with temporary.open("wb") as handle:
                                for block in response.iter_content(1 << 22):
                                    digest.update(block)
                                    written += handle.write(block)
                        if written != size or not hmac.compare_digest(digest.hexdigest(), sha256):
                            temporary.unlink(missing_ok=True)
                            raise ArtifactMismatch(url, written, digest.hexdigest())
                        temporary.replace(destination)
                        return destination
                    """
                ),
            },
            {
                "id": "etag-cache-go",
                "path": "internal/fetch/etag.go",
                "language": "go",
                "text": code(
                    """
                    // Fetch revalidates a cached response with If-None-Match. An ETag
                    // proves the representation is unchanged for caching purposes; it
                    // is not an integrity check, because the server chooses it.
                    func (f *Fetcher) Fetch(ctx context.Context, url string) (*Body, error) {
                        cached, hit := f.cache.Lookup(url)
                        req, _ := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
                        if hit {
                            req.Header.Set("If-None-Match", cached.ETag)
                        }
                        resp, err := f.client.Do(req)
                        if err != nil {
                            return nil, err
                        }
                        defer resp.Body.Close()
                        if resp.StatusCode == http.StatusNotModified && hit {
                            return cached.Body, nil
                        }
                        return f.store(url, resp)
                    }
                    """
                ),
            },
        ],
        "queries": [
            {
                "query": "verify a downloaded file against a pinned sha256 before using it",
                "grades": {
                    "checksum-verify-rust": 3,
                    "checksum-verify-python": 3,
                    "etag-cache-go": 1,
                },
            },
            {
                "query": "why check the byte length in addition to the hash",
                "grades": {"checksum-verify-python": 3},
            },
            {
                "query": "delete the partial download when verification fails",
                "grades": {"checksum-verify-rust": 3, "checksum-verify-python": 3},
            },
            {
                "query": "revalidate a cached HTTP response without downloading it again",
                "grades": {"etag-cache-go": 3},
            },
        ],
    },
    {
        "id": "env-config-precedence",
        "docs": [
            {
                "id": "config-precedence-rust",
                "path": "src/config/resolve.rs",
                "language": "rust",
                "text": code(
                    """
                    /// Resolves one setting across the supported sources.
                    ///
                    /// Precedence is command line, then exported process environment,
                    /// then the dotenv file, then the built-in default. A malformed
                    /// dotenv file is an error rather than a skipped source, so a typo
                    /// cannot silently downgrade configuration to defaults.
                    pub fn resolve<T: FromStr>(
                        flag: Option<T>,
                        name: &str,
                        default: T,
                    ) -> Result<T, ConfigError> {
                        if let Some(value) = flag {
                            return Ok(value);
                        }
                        match std::env::var(name) {
                            Ok(raw) => raw
                                .trim()
                                .parse()
                                .map_err(|_| ConfigError::Invalid { name: name.into(), raw }),
                            Err(VarError::NotPresent) => Ok(default),
                            Err(VarError::NotUnicode(_)) => Err(ConfigError::NotUnicode(name.into())),
                        }
                    }
                    """
                ),
            },
            {
                "id": "config-precedence-python",
                "path": "app/settings.py",
                "language": "python",
                "text": code(
                    """
                    def load_settings(argv=None) -> Settings:
                        \"\"\"Builds settings from flags, environment, and .env.

                        Values already exported by the parent process win over the
                        file, so a container's injected secrets are never overwritten
                        by a stale .env baked into the image.
                        \"\"\"
                        file_values = dotenv_values(find_dotenv(usecwd=True)) or {}
                        merged = {**file_values, **os.environ}
                        flags = parse_args(argv)
                        return Settings(
                            endpoint=flags.endpoint or merged.get("API_ENDPOINT", DEFAULT_ENDPOINT),
                            timeout=float(flags.timeout or merged.get("API_TIMEOUT", 30)),
                            debug=as_bool(merged.get("DEBUG", "false")),
                        )
                    """
                ),
            },
            {
                "id": "feature-flag-typescript",
                "path": "web/src/config/flags.ts",
                "language": "typescript",
                "text": code(
                    """
                    // Reads build-time feature flags. These are inlined by the bundler,
                    // so a flag flip requires a rebuild and cannot be changed per
                    // request. Runtime toggles belong in the remote config service.
                    export const flags = {
                      newEditor: import.meta.env.VITE_FLAG_NEW_EDITOR === "true",
                      inlineDiff: import.meta.env.VITE_FLAG_INLINE_DIFF === "true",
                    } as const;

                    export function assertKnownFlag(name: string): asserts name is keyof typeof flags {
                      if (!(name in flags)) {
                        throw new Error(`unknown feature flag: ${name}`);
                      }
                    }
                    """
                ),
            },
        ],
        "queries": [
            {
                "query": "what takes precedence between a command line flag, the environment, and a dotenv file",
                "grades": {
                    "config-precedence-rust": 3,
                    "config-precedence-python": 3,
                    "feature-flag-typescript": 1,
                },
            },
            {
                "query": "a stale dotenv file must not override injected container secrets",
                "grades": {"config-precedence-python": 3, "config-precedence-rust": 2},
            },
            {
                "query": "fail startup on a malformed configuration value instead of using the default",
                "grades": {"config-precedence-rust": 3},
            },
            {
                "query": "build time feature flags inlined by the bundler",
                "grades": {"feature-flag-typescript": 3},
            },
        ],
    },
    {
        "id": "idempotency-dedupe",
        "docs": [
            {
                "id": "idempotency-key-go",
                "path": "internal/payments/idempotency.go",
                "language": "go",
                "text": code(
                    """
                    // Charge is safe to call twice with the same key. The key is
                    // inserted first inside the same transaction as the charge, so a
                    // duplicate request loses the insert race and returns the stored
                    // result rather than charging again.
                    func (s *Service) Charge(ctx context.Context, key string, amount Money) (*Receipt, error) {
                        tx, err := s.db.BeginTx(ctx, nil)
                        if err != nil {
                            return nil, err
                        }
                        defer tx.Rollback()
                        if _, err := tx.ExecContext(ctx,
                            `INSERT INTO idempotency_keys (key) VALUES ($1)`, key); err != nil {
                            if isUniqueViolation(err) {
                                return s.loadReceipt(ctx, key)
                            }
                            return nil, err
                        }
                        receipt, err := s.capture(ctx, tx, key, amount)
                        if err != nil {
                            return nil, err
                        }
                        return receipt, tx.Commit()
                    }
                    """
                ),
            },
            {
                "id": "idempotency-client-typescript",
                "path": "web/src/api/submitOrder.ts",
                "language": "typescript",
                "text": code(
                    """
                    // Submits an order with a stable idempotency key so a retry after a
                    // timeout cannot create a second order. The key is generated once
                    // per attempt-set and kept in session storage, because a fresh key
                    // on retry would defeat the server's deduplication entirely.
                    export async function submitOrder(cart: Cart): Promise<Order> {
                      const key = sessionStorage.getItem(ORDER_KEY) ?? crypto.randomUUID();
                      sessionStorage.setItem(ORDER_KEY, key);
                      const order = await postJson<Order>("/api/orders", cart, {
                        headers: { "Idempotency-Key": key },
                      });
                      sessionStorage.removeItem(ORDER_KEY);
                      return order;
                    }
                    """
                ),
            },
            {
                "id": "dedupe-window-python",
                "path": "worker/dedupe.py",
                "language": "python",
                "text": code(
                    """
                    def process_once(message) -> bool:
                        \"\"\"Drops a message already handled inside the dedupe window.

                        This is best effort: the window is a Redis TTL, so a message
                        redelivered after it expires is processed again. Only use it
                        for idempotent side effects such as cache warming, never for
                        payments.
                        \"\"\"
                        added = redis.set(f"seen:{message.id}", "1", nx=True, ex=DEDUPE_TTL_SECONDS)
                        if not added:
                            metrics.increment("worker.duplicate_dropped")
                            return False
                        handle(message)
                        return True
                    """
                ),
            },
        ],
        "queries": [
            {
                "query": "make a payment request safe to retry without double charging",
                "grades": {
                    "idempotency-key-go": 3,
                    "idempotency-client-typescript": 2,
                    "dedupe-window-python": 1,
                },
            },
            {
                "query": "reuse the same idempotency key when the client retries",
                "grades": {"idempotency-client-typescript": 3, "idempotency-key-go": 1},
            },
            {
                "query": "best effort duplicate suppression with a redis TTL",
                "grades": {"dedupe-window-python": 3},
            },
            {
                "query": "handle the unique constraint violation from a concurrent duplicate insert",
                "grades": {"idempotency-key-go": 3},
            },
        ],
    },
    {
        "id": "circuit-breaker",
        "docs": [
            {
                "id": "circuit-breaker-java",
                "path": "src/main/java/resilience/CircuitBreaker.java",
                "language": "java",
                "text": code(
                    """
                    /**
                     * Three-state breaker: closed, open, half-open.
                     *
                     * A single trial call is admitted in half-open. Admitting more
                     * would stampede a recovering dependency, which is the failure
                     * this class exists to prevent.
                     */
                    public <T> T call(Supplier<T> action) throws BreakerOpen {
                        State state = state();
                        if (state == State.OPEN) {
                            throw new BreakerOpen(retryAfter());
                        }
                        if (state == State.HALF_OPEN && !trial.compareAndSet(false, true)) {
                            throw new BreakerOpen(retryAfter());
                        }
                        try {
                            T result = action.get();
                            onSuccess();
                            return result;
                        } catch (RuntimeException failure) {
                            onFailure();
                            throw failure;
                        }
                    }
                    """
                ),
            },
            {
                "id": "circuit-breaker-go",
                "path": "internal/resilience/breaker.go",
                "language": "go",
                "text": code(
                    """
                    // trip opens the breaker once the failure ratio over the rolling
                    // window exceeds the threshold. A ratio rather than a raw count so
                    // a low-traffic endpoint is not tripped by two unlucky requests.
                    func (b *Breaker) record(success bool) {
                        b.mu.Lock()
                        defer b.mu.Unlock()
                        b.window.Add(success)
                        if b.window.Total() < b.minimumRequests {
                            return
                        }
                        if b.window.FailureRatio() > b.threshold {
                            b.openedAt = time.Now()
                            b.state = StateOpen
                        }
                    }
                    """
                ),
            },
            {
                "id": "timeout-budget-rust",
                "path": "src/client/deadline.rs",
                "language": "rust",
                "text": code(
                    """
                    /// Splits a caller's remaining deadline across downstream calls.
                    ///
                    /// Each hop gets a share of what is left, not a fresh fixed
                    /// timeout, so a chain of three services cannot take three times
                    /// the budget the original caller was willing to wait.
                    impl Deadline {
                        pub fn child(&self, share: f32) -> Option<Deadline> {
                            let remaining = self.remaining()?;
                            if remaining < MIN_USEFUL_BUDGET {
                                return None;
                            }
                            Some(Self::after(remaining.mul_f32(share.clamp(0.1, 1.0))))
                        }
                    }
                    """
                ),
            },
        ],
        "queries": [
            {
                "query": "stop calling a failing dependency and probe it before fully reopening",
                "grades": {"circuit-breaker-java": 3, "circuit-breaker-go": 2},
            },
            {
                "query": "why admit only one request while the breaker is half open",
                "grades": {"circuit-breaker-java": 3},
            },
            {
                "query": "trip on a failure ratio rather than an absolute failure count",
                "grades": {"circuit-breaker-go": 3, "circuit-breaker-java": 1},
            },
            {
                "query": "propagate the remaining time budget to downstream calls",
                "grades": {"timeout-budget-rust": 3},
            },
        ],
    },
    {
        "id": "batch-upsert",
        "docs": [
            {
                "id": "batch-upsert-python",
                "path": "etl/load.py",
                "language": "python",
                "text": code(
                    """
                    def upsert_rows(connection, rows, chunk_size=1000):
                        \"\"\"Upserts in bounded batches inside one transaction.

                        Batching keeps the statement under the driver's parameter limit
                        and keeps peak memory flat; one transaction keeps the load
                        atomic, so a mid-load failure leaves the table as it was.
                        \"\"\"
                        with connection.begin():
                            for start in range(0, len(rows), chunk_size):
                                batch = rows[start : start + chunk_size]
                                connection.execute(
                                    text(
                                        \"\"\"INSERT INTO metrics (key, day, value)
                                            VALUES (:key, :day, :value)
                                            ON CONFLICT (key, day)
                                            DO UPDATE SET value = EXCLUDED.value\"\"\"
                                    ),
                                    batch,
                                )
                        return len(rows)
                    """
                ),
            },
            {
                "id": "batch-upsert-go",
                "path": "internal/store/upsert.go",
                "language": "go",
                "text": code(
                    """
                    // UpsertBatch writes rows with a single multi-values statement per
                    // chunk. Placeholders are generated rather than interpolated, so a
                    // key containing a quote is data and never syntax.
                    func (s *Store) UpsertBatch(ctx context.Context, rows []Row) error {
                        const perChunk = 500
                        for start := 0; start < len(rows); start += perChunk {
                            chunk := rows[start:min(start+perChunk, len(rows))]
                            placeholders := make([]string, 0, len(chunk))
                            args := make([]any, 0, len(chunk)*3)
                            for i, row := range chunk {
                                placeholders = append(placeholders,
                                    fmt.Sprintf("($%d,$%d,$%d)", i*3+1, i*3+2, i*3+3))
                                args = append(args, row.Key, row.Day, row.Value)
                            }
                            query := `INSERT INTO metrics (key, day, value) VALUES ` +
                                strings.Join(placeholders, ",") +
                                ` ON CONFLICT (key, day) DO UPDATE SET value = EXCLUDED.value`
                            if _, err := s.db.ExecContext(ctx, query, args...); err != nil {
                                return fmt.Errorf("upsert chunk at %d: %w", start, err)
                            }
                        }
                        return nil
                    }
                    """
                ),
            },
            {
                "id": "bulk-delete-typescript",
                "path": "server/maintenance/purge.ts",
                "language": "typescript",
                "text": code(
                    """
                    // Deletes expired rows in bounded slices so the statement never
                    // holds a long lock. Sleeping between slices deliberately yields to
                    // foreground traffic; a single unbounded DELETE would block writers
                    // for the length of the scan.
                    export async function purgeExpired(db: Db, before: Date, slice = 5_000) {
                      let removed = 0;
                      for (;;) {
                        const { rowCount } = await db.query(
                          `DELETE FROM sessions WHERE id IN (
                             SELECT id FROM sessions WHERE expires_at < $1 LIMIT $2)`,
                          [before, slice],
                        );
                        removed += rowCount;
                        if (rowCount < slice) return removed;
                        await sleep(50);
                      }
                    }
                    """
                ),
            },
        ],
        "queries": [
            {
                "query": "insert or update many rows efficiently in batches",
                "grades": {
                    "batch-upsert-python": 3,
                    "batch-upsert-go": 3,
                    "bulk-delete-typescript": 1,
                },
            },
            {
                "query": "keep a bulk load atomic so a failure halfway leaves no partial data",
                "grades": {"batch-upsert-python": 3, "batch-upsert-go": 1},
            },
            {
                "query": "delete old rows without holding a long lock",
                "grades": {"bulk-delete-typescript": 3},
            },
            {
                "query": "generate SQL placeholders instead of interpolating values",
                "grades": {"batch-upsert-go": 3},
            },
        ],
    },
    {
        "id": "websocket-reconnect",
        "docs": [
            {
                "id": "ws-reconnect-typescript",
                "path": "web/src/live/socket.ts",
                "language": "typescript",
                "text": code(
                    """
                    // Keeps a live socket open. Reconnect delay grows to a ceiling so a
                    // server restart does not produce a thundering herd, and the timer
                    // resets only after a connection survives long enough to be real.
                    export function connect(url: string, onEvent: (event: LiveEvent) => void): Closer {
                      let attempt = 0;
                      let socket: WebSocket | null = null;
                      let closed = false;

                      const open = () => {
                        socket = new WebSocket(url);
                        const settled = setTimeout(() => (attempt = 0), 10_000);
                        socket.onmessage = (raw) => onEvent(JSON.parse(raw.data));
                        socket.onclose = () => {
                          clearTimeout(settled);
                          if (closed) return;
                          const delay = Math.min(1000 * 2 ** attempt++, 30_000);
                          setTimeout(open, delay + Math.random() * 250);
                        };
                      };

                      open();
                      return () => { closed = true; socket?.close(); };
                    }
                    """
                ),
            },
            {
                "id": "ws-heartbeat-python",
                "path": "server/live/heartbeat.py",
                "language": "python",
                "text": code(
                    """
                    async def heartbeat(socket, interval=20.0, grace=10.0) -> None:
                        \"\"\"Pings the peer and closes a silent connection.

                        A TCP connection can stay open long after the peer is gone, so
                        liveness needs an application-level ping. Missing one pong is
                        tolerated; missing two closes, which keeps a brief network
                        hiccup from dropping an otherwise healthy session.
                        \"\"\"
                        missed = 0
                        while not socket.closed:
                            await asyncio.sleep(interval)
                            try:
                                await asyncio.wait_for(socket.ping(), timeout=grace)
                                missed = 0
                            except asyncio.TimeoutError:
                                missed += 1
                                if missed >= 2:
                                    await socket.close(code=1001, reason="no pong")
                                    return
                    """
                ),
            },
            {
                "id": "sse-stream-go",
                "path": "internal/live/sse.go",
                "language": "go",
                "text": code(
                    """
                    // StreamEvents writes server-sent events. Each write is flushed
                    // immediately, because a buffered proxy would otherwise hold events
                    // until the buffer fills and make a live feed look frozen.
                    func StreamEvents(w http.ResponseWriter, r *http.Request, events <-chan Event) {
                        flusher, ok := w.(http.Flusher)
                        if !ok {
                            http.Error(w, "streaming unsupported", http.StatusInternalServerError)
                            return
                        }
                        w.Header().Set("Content-Type", "text/event-stream")
                        w.Header().Set("Cache-Control", "no-cache")
                        for {
                            select {
                            case <-r.Context().Done():
                                return
                            case event, open := <-events:
                                if !open {
                                    return
                                }
                                fmt.Fprintf(w, "id: %s\\ndata: %s\\n\\n", event.ID, event.JSON())
                                flusher.Flush()
                            }
                        }
                    }
                    """
                ),
            },
        ],
        "queries": [
            {
                "query": "reconnect a dropped websocket with increasing delay",
                "grades": {"ws-reconnect-typescript": 3, "ws-heartbeat-python": 1},
            },
            {
                "query": "detect a peer that stopped responding on an open connection",
                "grades": {"ws-heartbeat-python": 3, "ws-reconnect-typescript": 1},
            },
            {
                "query": "why flush every server-sent event immediately",
                "grades": {"sse-stream-go": 3},
            },
            {
                "query": "avoid a thundering herd of clients reconnecting at once",
                "grades": {"ws-reconnect-typescript": 3, "ws-heartbeat-python": 1},
            },
        ],
    },
    {
        "id": "transaction-savepoint",
        "docs": [
            {
                "id": "savepoint-python",
                "path": "app/repository/orders.py",
                "language": "python",
                "text": code(
                    """
                    def import_orders(session, rows):
                        \"\"\"Imports rows, isolating per-row failures with savepoints.

                        A nested savepoint lets one bad row roll back alone while the
                        outer transaction keeps the good ones. Without it, the first
                        integrity error poisons the whole transaction and every later
                        statement fails.
                        \"\"\"
                        imported, rejected = 0, []
                        for row in rows:
                            try:
                                with session.begin_nested():
                                    session.add(Order.from_row(row))
                                imported += 1
                            except IntegrityError as error:
                                rejected.append((row.external_id, str(error.orig)))
                        session.commit()
                        return imported, rejected
                    """
                ),
            },
            {
                "id": "transaction-rollback-php",
                "path": "src/Repository/LedgerRepository.php",
                "language": "php",
                "text": code(
                    """
                    <?php
                    /**
                     * Moves funds between accounts atomically.
                     *
                     * Both rows are locked in a deterministic id order; locking in
                     * arrival order instead is what produces deadlocks between two
                     * concurrent transfers of the same pair.
                     */
                    public function transfer(int $from, int $to, int $cents): void
                    {
                        $this->connection->beginTransaction();
                        try {
                            [$first, $second] = $from < $to ? [$from, $to] : [$to, $from];
                            $this->lockAccount($first);
                            $this->lockAccount($second);
                            $this->debit($from, $cents);
                            $this->credit($to, $cents);
                            $this->connection->commit();
                        } catch (\\Throwable $error) {
                            $this->connection->rollBack();
                            throw new TransferFailed($from, $to, $error);
                        }
                    }
                    """
                ),
            },
            {
                "id": "retry-serialization-java",
                "path": "src/main/java/store/SerializableRetry.java",
                "language": "java",
                "text": code(
                    """
                    /**
                     * Retries a transaction that lost a serialization conflict.
                     *
                     * Only the 40001 class is retried: the database is telling us the
                     * transaction can succeed if run again. A constraint violation is
                     * a bug in the data and must not be retried in a loop.
                     */
                    public <T> T inSerializableTransaction(Function<Connection, T> work) throws SQLException {
                        for (int attempt = 1; ; attempt++) {
                            try (Connection connection = pool.getConnection()) {
                                connection.setTransactionIsolation(Connection.TRANSACTION_SERIALIZABLE);
                                connection.setAutoCommit(false);
                                T result = work.apply(connection);
                                connection.commit();
                                return result;
                            } catch (SQLException failure) {
                                if (!"40001".equals(failure.getSQLState()) || attempt >= maxAttempts) {
                                    throw failure;
                                }
                                sleepWithJitter(attempt);
                            }
                        }
                    }
                    """
                ),
            },
        ],
        "queries": [
            {
                "query": "let one bad row fail without aborting the whole import transaction",
                "grades": {"savepoint-python": 3, "retry-serialization-java": 1},
            },
            {
                "query": "avoid deadlocks when two transfers touch the same two accounts",
                "grades": {"transaction-rollback-php": 3},
            },
            {
                "query": "retry only the database errors that are safe to retry",
                "grades": {"retry-serialization-java": 3, "savepoint-python": 1},
            },
            {
                "query": "roll back and wrap the failure when a money transfer fails",
                "grades": {"transaction-rollback-php": 3, "savepoint-python": 1},
            },
        ],
    },
    {
        "id": "optimistic-locking",
        "docs": [
            {
                "id": "optimistic-lock-java",
                "path": "src/main/java/store/DocumentStore.java",
                "language": "java",
                "text": code(
                    """
                    /**
                     * Saves a document only if nobody else changed it first.
                     *
                     * The update is conditional on the version the caller read, so a
                     * concurrent edit is reported as a conflict instead of silently
                     * overwriting the other author's work.
                     */
                    public Document save(Document document) throws ConflictException {
                        int updated = jdbc.update(
                            "UPDATE documents SET body = ?, version = version + 1 "
                                + "WHERE id = ? AND version = ?",
                            document.body(), document.id(), document.version());
                        if (updated == 0) {
                            throw new ConflictException(document.id(), document.version());
                        }
                        return document.withVersion(document.version() + 1);
                    }
                    """
                ),
            },
            {
                "id": "optimistic-lock-python",
                "path": "app/repository/settings.py",
                "language": "python",
                "text": code(
                    """
                    def update_if_unchanged(session, record_id, expected_etag, changes):
                        \"\"\"Applies changes only when the stored etag still matches.

                        Returns None on conflict rather than raising, because the HTTP
                        layer turns it into 412 Precondition Failed and the caller is
                        expected to re-read and merge.
                        \"\"\"
                        result = session.execute(
                            update(Settings)
                            .where(Settings.id == record_id, Settings.etag == expected_etag)
                            .values(**changes, etag=new_etag())
                            .returning(Settings)
                        ).scalar_one_or_none()
                        session.commit()
                        return result
                    """
                ),
            },
            {
                "id": "pessimistic-lock-go",
                "path": "internal/store/reserve.go",
                "language": "go",
                "text": code(
                    """
                    // ReserveSeat takes a row lock for the duration of the booking.
                    // SELECT FOR UPDATE is used instead of a version check because a
                    // conflict here means a lost sale, and holding the lock briefly is
                    // cheaper than asking the customer to try again.
                    func (s *Store) ReserveSeat(ctx context.Context, seatID int64, holder string) error {
                        return s.inTx(ctx, func(tx *sql.Tx) error {
                            var takenBy sql.NullString
                            if err := tx.QueryRowContext(ctx,
                                `SELECT held_by FROM seats WHERE id = $1 FOR UPDATE`, seatID).
                                Scan(&takenBy); err != nil {
                                return err
                            }
                            if takenBy.Valid {
                                return ErrSeatTaken
                            }
                            _, err := tx.ExecContext(ctx,
                                `UPDATE seats SET held_by = $1, held_at = now() WHERE id = $2`, holder, seatID)
                            return err
                        })
                    }
                    """
                ),
            },
        ],
        "queries": [
            {
                "query": "prevent one user's save from overwriting another's concurrent edit",
                "grades": {
                    "optimistic-lock-java": 3,
                    "optimistic-lock-python": 3,
                    "pessimistic-lock-go": 1,
                },
            },
            {
                "query": "return 412 when the etag no longer matches",
                "grades": {"optimistic-lock-python": 3},
            },
            {
                "query": "when to take a row lock instead of using a version column",
                "grades": {"pessimistic-lock-go": 3, "optimistic-lock-java": 1},
            },
        ],
    },
    {
        "id": "debounce-watch",
        "docs": [
            {
                "id": "debounce-watch-rust",
                "path": "src/watch/debounce.rs",
                "language": "rust",
                "text": code(
                    """
                    /// Collapses a burst of filesystem events into one rebuild.
                    ///
                    /// Editors write a file as several events, and a compiler run per
                    /// event would thrash. The timer restarts on every event, so the
                    /// rebuild fires once the tree has been quiet for the window.
                    pub async fn debounced(mut events: Receiver<PathBuf>, window: Duration) -> Vec<PathBuf> {
                        let mut pending = BTreeSet::new();
                        loop {
                            match timeout(window, events.recv()).await {
                                Ok(Some(path)) => {
                                    pending.insert(path);
                                }
                                Ok(None) => break,
                                Err(_) if !pending.is_empty() => break,
                                Err(_) => {}
                            }
                        }
                        pending.into_iter().collect()
                    }
                    """
                ),
            },
            {
                "id": "debounce-input-typescript",
                "path": "web/src/hooks/useDebouncedValue.ts",
                "language": "typescript",
                "text": code(
                    """
                    // Delays a value until typing pauses, so a search box issues one
                    // request per pause instead of one per keystroke. The cleanup
                    // cancels the pending timer, without which an unmounted component
                    // would still fire a state update.
                    export function useDebouncedValue<T>(value: T, delayMs = 250): T {
                      const [settled, setSettled] = useState(value);
                      useEffect(() => {
                        const timer = setTimeout(() => setSettled(value), delayMs);
                        return () => clearTimeout(timer);
                      }, [value, delayMs]);
                      return settled;
                    }
                    """
                ),
            },
            {
                "id": "watch-ignore-python",
                "path": "tools/watch.py",
                "language": "python",
                "text": code(
                    """
                    def should_watch(path: Path) -> bool:
                        \"\"\"Filters build output and version control noise.

                        Watching a build directory creates a loop: the build writes into
                        it, the write triggers a rebuild, and the rebuild writes again.
                        \"\"\"
                        parts = set(path.parts)
                        if parts & {".git", "node_modules", "target", "__pycache__", "dist"}:
                            return False
                        if path.name.startswith(".#") or path.name.endswith("~"):
                            return False
                        return path.suffix in WATCHED_SUFFIXES
                    """
                ),
            },
        ],
        "queries": [
            {
                "query": "collapse a burst of file change events into a single rebuild",
                "grades": {
                    "debounce-watch-rust": 3,
                    "debounce-input-typescript": 2,
                    "watch-ignore-python": 1,
                },
            },
            {
                "query": "wait until the user stops typing before searching",
                "grades": {"debounce-input-typescript": 3, "debounce-watch-rust": 1},
            },
            {
                "query": "stop the watcher from triggering itself through build output",
                "grades": {"watch-ignore-python": 3},
            },
        ],
    },
    {
        "id": "streaming-parse",
        "docs": [
            {
                "id": "csv-stream-python",
                "path": "etl/read_csv.py",
                "language": "python",
                "text": code(
                    """
                    def iter_rows(path: Path, expected: Sequence[str]) -> Iterator[dict]:
                        \"\"\"Yields rows one at a time from a large CSV.

                        Streaming keeps memory flat regardless of file size. The header
                        is validated once up front so a column rename fails immediately
                        rather than producing thousands of rows with missing fields.
                        \"\"\"
                        with path.open(newline="", encoding="utf-8") as handle:
                            reader = csv.DictReader(handle)
                            missing = set(expected) - set(reader.fieldnames or [])
                            if missing:
                                raise SchemaMismatch(path, sorted(missing))
                            for number, row in enumerate(reader, start=2):
                                yield {"_line": number, **row}
                    """
                ),
            },
            {
                "id": "ndjson-stream-go",
                "path": "internal/ingest/ndjson.go",
                "language": "go",
                "text": code(
                    """
                    // DecodeStream reads newline-delimited JSON without loading the file.
                    // The scanner buffer is raised because one oversized line would
                    // otherwise abort the whole ingest with a bufio.ErrTooLong.
                    func DecodeStream(r io.Reader, handle func(Record) error) error {
                        scanner := bufio.NewScanner(r)
                        scanner.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
                        line := 0
                        for scanner.Scan() {
                            line++
                            if len(bytes.TrimSpace(scanner.Bytes())) == 0 {
                                continue
                            }
                            var record Record
                            if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
                                return fmt.Errorf("line %d: %w", line, err)
                            }
                            if err := handle(record); err != nil {
                                return fmt.Errorf("line %d: %w", line, err)
                            }
                        }
                        return scanner.Err()
                    }
                    """
                ),
            },
            {
                "id": "json-load-all-rust",
                "path": "src/config/load.rs",
                "language": "rust",
                "text": code(
                    """
                    /// Reads a whole manifest into memory and rejects unknown fields.
                    ///
                    /// Whole-file reading is correct here because a manifest is bounded
                    /// and every field is needed at once; the size cap is what keeps
                    /// that assumption true if someone points this at a data file.
                    pub fn load_manifest(path: &Path) -> Result<Manifest, LoadError> {
                        let metadata = std::fs::metadata(path)?;
                        if metadata.len() > MAX_MANIFEST_BYTES {
                            return Err(LoadError::TooLarge { bytes: metadata.len() });
                        }
                        let text = std::fs::read_to_string(path)?;
                        serde_json::from_str(&text).map_err(|error| LoadError::Parse {
                            path: path.to_path_buf(),
                            message: error.to_string(),
                        })
                    }
                    """
                ),
            },
        ],
        "queries": [
            {
                "query": "process a huge delimited file without loading it into memory",
                "grades": {
                    "csv-stream-python": 3,
                    "ndjson-stream-go": 3,
                    "json-load-all-rust": 1,
                },
            },
            {
                "query": "report the line number when a record fails to parse",
                "grades": {"ndjson-stream-go": 3, "csv-stream-python": 2},
            },
            {
                "query": "fail fast when a CSV column was renamed",
                "grades": {"csv-stream-python": 3},
            },
            {
                "query": "cap the size of a file that is read entirely into memory",
                "grades": {"json-load-all-rust": 3, "ndjson-stream-go": 1},
            },
        ],
    },
    {
        "id": "password-hashing",
        "docs": [
            {
                "id": "password-hash-php",
                "path": "src/Auth/PasswordService.php",
                "language": "php",
                "text": code(
                    """
                    <?php
                    /**
                     * Hashes and verifies user passwords.
                     *
                     * The algorithm and cost live in the hash string, so verify()
                     * keeps working after a cost bump and needsRehash() upgrades the
                     * stored hash during a successful login, when the plaintext is
                     * available for the only time.
                     */
                    final class PasswordService
                    {
                        public function hash(string $plaintext): string
                        {
                            return password_hash($plaintext, PASSWORD_ARGON2ID, self::OPTIONS);
                        }

                        public function verify(string $plaintext, string $stored, callable $persist): bool
                        {
                            if (!password_verify($plaintext, $stored)) {
                                return false;
                            }
                            if (password_needs_rehash($stored, PASSWORD_ARGON2ID, self::OPTIONS)) {
                                $persist($this->hash($plaintext));
                            }
                            return true;
                        }
                    }
                    """
                ),
            },
            {
                "id": "password-hash-python",
                "path": "app/auth/passwords.py",
                "language": "python",
                "text": code(
                    """
                    def verify_password(plaintext: str, stored: str) -> bool:
                        \"\"\"Checks a password in constant time relative to the secret.

                        A missing user is still run through a dummy verification so the
                        response time does not reveal whether the account exists.
                        \"\"\"
                        try:
                            hasher.verify(stored or DUMMY_HASH, plaintext)
                            return stored is not None
                        except VerificationError:
                            return False


                    def hash_password(plaintext: str) -> str:
                        if len(plaintext) > MAX_PASSWORD_BYTES:
                            raise PasswordTooLong(MAX_PASSWORD_BYTES)
                        return hasher.hash(plaintext)
                    """
                ),
            },
            {
                "id": "api-key-hash-java",
                "path": "src/main/java/auth/ApiKeyStore.java",
                "language": "java",
                "text": code(
                    """
                    /**
                     * Stores API keys as SHA-256 digests with a visible prefix.
                     *
                     * Keys are high-entropy and looked up per request, so a slow
                     * password hash would be the wrong trade here; the prefix makes
                     * the lookup indexable without revealing the secret.
                     */
                    public ApiKey issue(String owner) {
                        byte[] secret = new byte[32];
                        random.nextBytes(secret);
                        String presented = "hk_" + base64Url(secret);
                        jdbc.update("INSERT INTO api_keys (owner, prefix, digest) VALUES (?, ?, ?)",
                            owner, presented.substring(0, 11), sha256(presented));
                        return new ApiKey(presented);
                    }
                    """
                ),
            },
        ],
        "queries": [
            {
                "query": "hash a user password and upgrade the stored hash over time",
                "grades": {"password-hash-php": 3, "password-hash-python": 2},
            },
            {
                "query": "do not let response timing reveal whether an account exists",
                "grades": {"password-hash-python": 3, "password-hash-php": 1},
            },
            {
                "query": "why store API keys with a fast digest instead of argon2",
                "grades": {"api-key-hash-java": 3, "password-hash-php": 1},
            },
        ],
    },
    {
        "id": "structured-logging",
        "docs": [
            {
                "id": "log-redaction-typescript",
                "path": "server/logging/redact.ts",
                "language": "typescript",
                "text": code(
                    """
                    // Removes secrets before a record reaches a log sink. Redaction runs
                    // on the structured record, not the formatted string, so a nested
                    // token cannot survive by being serialized early.
                    const SENSITIVE = /^(authorization|cookie|set-cookie|password|token|secret|api[-_]?key)$/i;

                    export function redact(value: unknown, depth = 0): unknown {
                      if (depth > 6 || value === null || typeof value !== "object") return value;
                      if (Array.isArray(value)) return value.map((item) => redact(item, depth + 1));
                      return Object.fromEntries(
                        Object.entries(value).map(([key, inner]) =>
                          SENSITIVE.test(key) ? [key, "[REDACTED]"] : [key, redact(inner, depth + 1)],
                        ),
                      );
                    }
                    """
                ),
            },
            {
                "id": "log-context-python",
                "path": "app/logging/context.py",
                "language": "python",
                "text": code(
                    """
                    request_id: ContextVar[str] = ContextVar("request_id", default="-")


                    class ContextFilter(logging.Filter):
                        \"\"\"Attaches the current request id to every record.

                        A ContextVar rather than thread-local state, because the async
                        handlers share threads and thread-locals would leak one
                        request's id into another's log lines.
                        \"\"\"

                        def filter(self, record: logging.LogRecord) -> bool:
                            record.request_id = request_id.get()
                            return True
                    """
                ),
            },
            {
                "id": "log-sampling-java",
                "path": "src/main/java/observability/SampledLogger.java",
                "language": "java",
                "text": code(
                    """
                    /**
                     * Logs at most one message per key per interval.
                     *
                     * A tight loop that fails on every iteration would otherwise fill
                     * the log with identical lines and hide everything else. The
                     * suppressed count is emitted with the next admitted line so the
                     * true volume is never lost.
                     */
                    public void warnThrottled(String key, String message) {
                        Window window = windows.computeIfAbsent(key, unused -> new Window());
                        long suppressed = window.recordAndCount(clock.millis(), intervalMillis);
                        if (suppressed >= 0) {
                            log.warn("{} (suppressed {} similar)", message, suppressed);
                        }
                    }
                    """
                ),
            },
        ],
        "queries": [
            {
                "query": "strip authorization headers and tokens out of logs",
                "grades": {"log-redaction-typescript": 3},
            },
            {
                "query": "attach a request id to every log line in async code",
                "grades": {"log-context-python": 3},
            },
            {
                "query": "keep a failing loop from flooding the log with identical lines",
                "grades": {"log-sampling-java": 3, "log-redaction-typescript": 1},
            },
            {
                "query": "why redact the structured record rather than the formatted message",
                "grades": {"log-redaction-typescript": 3},
            },
        ],
    },
    {
        "id": "connection-pool",
        "docs": [
            {
                "id": "pool-checkout-java",
                "path": "src/main/java/store/Pool.java",
                "language": "java",
                "text": code(
                    """
                    /**
                     * Hands out a pooled connection or fails fast.
                     *
                     * Waiting forever for a connection turns a slow database into an
                     * unbounded queue of stuck threads, so the checkout has a deadline
                     * and a validation probe that discards a connection the server
                     * already closed.
                     */
                    public Connection borrow(Duration timeout) throws PoolExhausted {
                        Connection pooled = idle.poll(timeout.toMillis(), TimeUnit.MILLISECONDS);
                        if (pooled == null) {
                            throw new PoolExhausted(size(), timeout);
                        }
                        if (!isUsable(pooled)) {
                            discard(pooled);
                            return open();
                        }
                        return pooled;
                    }
                    """
                ),
            },
            {
                "id": "pool-size-go",
                "path": "internal/store/pool.go",
                "language": "go",
                "text": code(
                    """
                    // configurePool bounds both pool dimensions and connection lifetime.
                    // MaxIdleConns above MaxOpenConns silently wastes memory, and an
                    // unbounded lifetime keeps connections pinned to a database instance
                    // that a failover has already replaced.
                    func configurePool(db *sql.DB, maxOpen int) {
                        db.SetMaxOpenConns(maxOpen)
                        db.SetMaxIdleConns(maxOpen / 2)
                        db.SetConnMaxLifetime(30 * time.Minute)
                        db.SetConnMaxIdleTime(5 * time.Minute)
                    }
                    """
                ),
            },
            {
                "id": "pool-leak-python",
                "path": "app/db/session.py",
                "language": "python",
                "text": code(
                    """
                    @contextmanager
                    def session_scope():
                        \"\"\"Yields a session and always returns it to the pool.

                        The finally block is the whole point: an early return or a raised
                        exception inside the caller would otherwise leak the connection,
                        and a leaked connection is invisible until the pool is empty.
                        \"\"\"
                        session = SessionLocal()
                        try:
                            yield session
                            session.commit()
                        except Exception:
                            session.rollback()
                            raise
                        finally:
                            session.close()
                    """
                ),
            },
        ],
        "queries": [
            {
                "query": "fail fast instead of blocking forever when the connection pool is empty",
                "grades": {"pool-checkout-java": 3, "pool-size-go": 1},
            },
            {
                "query": "make sure a database connection is always returned to the pool",
                "grades": {"pool-leak-python": 3, "pool-checkout-java": 1},
            },
            {
                "query": "why limit how long a pooled connection may live",
                "grades": {"pool-size-go": 3},
            },
            {
                "query": "discard a pooled connection the server already closed",
                "grades": {"pool-checkout-java": 3, "pool-size-go": 2},
            },
        ],
    },
    {
        "id": "binary-search-boundary",
        "docs": [
            {
                "id": "binary-search-rust",
                "path": "src/index/search.rs",
                "language": "rust",
                "text": code(
                    """
                    /// Returns the insertion point for `needle` in a sorted slice.
                    ///
                    /// Returning the position on a miss, not just "not found", is what
                    /// lets the caller insert without a second scan. The midpoint is
                    /// computed as low + (high - low) / 2 to avoid overflow on large
                    /// index arrays.
                    pub fn insertion_point(sorted: &[u64], needle: u64) -> usize {
                        let (mut low, mut high) = (0usize, sorted.len());
                        while low < high {
                            let mid = low + (high - low) / 2;
                            if sorted[mid] < needle {
                                low = mid + 1;
                            } else {
                                high = mid;
                            }
                        }
                        low
                    }
                    """
                ),
            },
            {
                "id": "range-scan-go",
                "path": "internal/index/rangescan.go",
                "language": "go",
                "text": code(
                    """
                    // Between returns the sub-slice covering [from, to). Both bounds are
                    // found with sort.Search, so a range query costs two logarithmic
                    // probes instead of a full scan.
                    func Between(sorted []int64, from, to int64) []int64 {
                        start := sort.Search(len(sorted), func(i int) bool { return sorted[i] >= from })
                        end := sort.Search(len(sorted), func(i int) bool { return sorted[i] >= to })
                        if start > end {
                            return nil
                        }
                        return sorted[start:end]
                    }
                    """
                ),
            },
            {
                "id": "linear-scan-python",
                "path": "tools/lookup.py",
                "language": "python",
                "text": code(
                    """
                    def find_first_match(rows, predicate):
                        \"\"\"Linear scan for the first row satisfying predicate.

                        Deliberately linear: the predicate is arbitrary and the rows are
                        unsorted, so bisect does not apply. Sorting first would cost more
                        than the scan for the sizes this is used on.
                        \"\"\"
                        for index, row in enumerate(rows):
                            if predicate(row):
                                return index, row
                        return None, None
                    """
                ),
            },
        ],
        "queries": [
            {
                "query": "find where a value should be inserted in a sorted array",
                "grades": {"binary-search-rust": 3, "range-scan-go": 2, "linear-scan-python": 1},
            },
            {
                "query": "compute the midpoint without integer overflow",
                "grades": {"binary-search-rust": 3},
            },
            {
                "query": "select a range of sorted keys with two logarithmic probes",
                "grades": {"range-scan-go": 3, "binary-search-rust": 2},
            },
            {
                "query": "when a linear scan is the right choice over bisect",
                "grades": {"linear-scan-python": 3},
            },
        ],
    },
    {
        "id": "migration-guard",
        "docs": [
            {
                "id": "migration-guard-python",
                "path": "tools/migrate.py",
                "language": "python",
                "text": code(
                    """
                    def apply_pending(connection, migrations: list[Migration]) -> list[str]:
                        \"\"\"Applies unapplied migrations under an advisory lock.

                        The lock serializes concurrent deployers; without it two pods
                        starting together both see the same pending list and apply it
                        twice. Each migration commits on its own so a failure leaves a
                        recorded, resumable position.
                        \"\"\"
                        applied = []
                        with advisory_lock(connection, MIGRATION_LOCK_ID):
                            done = set(fetch_applied(connection))
                            for migration in migrations:
                                if migration.id in done:
                                    continue
                                with connection.begin():
                                    connection.execute(text(migration.sql))
                                    record_applied(connection, migration.id, migration.checksum)
                                applied.append(migration.id)
                        return applied
                    """
                ),
            },
            {
                "id": "migration-checksum-rust",
                "path": "src/migrate/verify.rs",
                "language": "rust",
                "text": code(
                    """
                    /// Refuses to run when an already-applied migration was edited.
                    ///
                    /// The recorded checksum is compared against the file on disk. A
                    /// changed file means the database and the repository disagree about
                    /// history, and continuing would apply the new statements nowhere.
                    pub fn verify_history(applied: &[AppliedMigration], files: &[MigrationFile]) -> Result<()> {
                        for record in applied {
                            let file = files
                                .iter()
                                .find(|file| file.id == record.id)
                                .ok_or_else(|| MigrateError::Missing(record.id.clone()))?;
                            if file.checksum != record.checksum {
                                return Err(MigrateError::Edited {
                                    id: record.id.clone(),
                                    recorded: record.checksum.clone(),
                                    found: file.checksum.clone(),
                                });
                            }
                        }
                        Ok(())
                    }
                    """
                ),
            },
            {
                "id": "backfill-batched-php",
                "path": "src/Maintenance/BackfillCommand.php",
                "language": "php",
                "text": code(
                    """
                    <?php
                    /**
                     * Backfills a new column in batches outside the migration.
                     *
                     * Schema change and data change are separated on purpose: an
                     * UPDATE over every row inside a migration holds locks for the
                     * length of the table and blocks deploys.
                     */
                    public function backfill(int $batch = 2000): int
                    {
                        $total = 0;
                        do {
                            $affected = $this->connection->executeStatement(
                                'UPDATE invoices SET currency = :currency
                                 WHERE currency IS NULL LIMIT :batch',
                                ['currency' => 'EUR', 'batch' => $batch]
                            );
                            $total += $affected;
                            usleep(50_000);
                        } while ($affected > 0);
                        return $total;
                    }
                    """
                ),
            },
        ],
        "queries": [
            {
                "query": "stop two deployments from applying the same migration twice",
                "grades": {"migration-guard-python": 3},
            },
            {
                "query": "detect that an already applied migration file was modified",
                "grades": {"migration-checksum-rust": 3, "migration-guard-python": 1},
            },
            {
                "query": "backfill a column without holding locks for the whole table",
                "grades": {"backfill-batched-php": 3, "migration-guard-python": 1},
            },
        ],
    },
    {
        "id": "cron-schedule",
        "docs": [
            {
                "id": "cron-parse-go",
                "path": "internal/schedule/cron.go",
                "language": "go",
                "text": code(
                    """
                    // Next returns the next firing at or after t, in the schedule's own
                    // location. Local time is used deliberately: a nightly job should
                    // stay nightly across a daylight saving change, which means one day
                    // a year has a 23 or 25 hour gap.
                    func (s *Schedule) Next(t time.Time) time.Time {
                        t = t.In(s.location).Truncate(time.Minute).Add(time.Minute)
                        for limit := 0; limit < maxSearchMinutes; limit++ {
                            if s.matches(t) {
                                return t
                            }
                            t = t.Add(time.Minute)
                        }
                        return time.Time{}
                    }
                    """
                ),
            },
            {
                "id": "cron-overlap-python",
                "path": "worker/scheduler.py",
                "language": "python",
                "text": code(
                    """
                    async def run_scheduled(job, interval_seconds: float) -> None:
                        \"\"\"Runs a job on an interval without overlapping executions.

                        The next run is scheduled from the end of the previous one, so a
                        job that takes longer than its interval falls behind instead of
                        piling up concurrent copies.
                        \"\"\"
                        while True:
                            started = time.monotonic()
                            try:
                                await job()
                            except Exception:
                                logger.exception("scheduled job failed", extra={"job": job.__name__})
                            elapsed = time.monotonic() - started
                            await asyncio.sleep(max(0.0, interval_seconds - elapsed))
                    """
                ),
            },
            {
                "id": "relative-time-typescript",
                "path": "web/src/format/relativeTime.ts",
                "language": "typescript",
                "text": code(
                    """
                    // Formats a timestamp as "3 minutes ago" in the viewer's locale.
                    // Intl.RelativeTimeFormat handles pluralization and translation, so
                    // no string table is needed and no locale is hard-coded.
                    export function relativeTime(when: Date, now = new Date(), locale?: string): string {
                      const seconds = Math.round((when.getTime() - now.getTime()) / 1000);
                      const formatter = new Intl.RelativeTimeFormat(locale, { numeric: "auto" });
                      for (const [unit, size] of UNITS) {
                        if (Math.abs(seconds) >= size || unit === "second") {
                          return formatter.format(Math.round(seconds / size), unit);
                        }
                      }
                      return formatter.format(seconds, "second");
                    }
                    """
                ),
            },
        ],
        "queries": [
            {
                "query": "compute the next run time of a cron expression across a daylight saving change",
                "grades": {"cron-parse-go": 3, "cron-overlap-python": 1},
            },
            {
                "query": "stop a slow periodic job from running twice at once",
                "grades": {"cron-overlap-python": 3, "cron-parse-go": 1},
            },
            {
                "query": "render a timestamp as a localized relative time",
                "grades": {"relative-time-typescript": 3},
            },
        ],
    },
]


def build() -> tuple[list[dict], list[dict]]:
    documents: list[dict] = []
    queries: list[dict] = []
    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    for cluster in CLUSTERS:
        for document in cluster["docs"]:
            if document["id"] in seen_ids:
                raise SystemExit(f"duplicate document id {document['id']}")
            seen_ids.add(document["id"])
            documents.append(
                {
                    "doc_id": document["id"],
                    "path": document["path"],
                    "language": document["language"],
                    "text": document["text"],
                }
            )
        for case in cluster["queries"]:
            if case["query"] in seen_queries:
                raise SystemExit(f"duplicate query {case['query']}")
            seen_queries.add(case["query"])
            unknown = set(case["grades"]) - seen_ids
            if unknown:
                raise SystemExit(f"query {case['query']!r} grades unknown ids {sorted(unknown)}")
            if not any(grade > 0 for grade in case["grades"].values()):
                raise SystemExit(f"query {case['query']!r} has no positive grade")
            queries.append({"query": case["query"], "graded_doc_ids": case["grades"]})
    return documents, queries


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    documents, queries = build()
    frozen = read_jsonl(FROZEN_CORPUS)
    frozen_ids = {document["doc_id"] for document in frozen}
    collisions = frozen_ids & {document["doc_id"] for document in documents}
    if collisions:
        raise SystemExit(f"code ids collide with the frozen prose set: {sorted(collisions)}")
    # The default corpus is prose plus code: a code query has to outrank 60
    # plausible prose documents, and vice versa, which is a harder and more
    # realistic haystack than either half alone.
    write_jsonl(EVALS / "corpus.jsonl", frozen + documents)
    write_jsonl(EVALS / "code-queries.jsonl", queries)
    print(f"frozen prose documents carried through: {len(frozen)}")
    grades = [grade for case in queries for grade in case["graded_doc_ids"].values()]
    judged = [len(case["graded_doc_ids"]) for case in queries]
    languages = sorted({document["language"] for document in documents})
    print(f"clusters:  {len(CLUSTERS)}")
    print(f"documents: {len(documents)} across {len(languages)} languages: {', '.join(languages)}")
    print(f"queries:   {len(queries)}")
    print(f"judged docs per query: mean {sum(judged) / len(judged):.2f}, max {max(judged)}")
    print(f"grade mix: 3={grades.count(3)} 2={grades.count(2)} 1={grades.count(1)}")
    print(f"default corpus now: {len(frozen) + len(documents)} documents")


if __name__ == "__main__":
    main()
