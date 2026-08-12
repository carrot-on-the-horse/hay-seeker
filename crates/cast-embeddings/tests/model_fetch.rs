//! Provisioning behavior for pinned model bundles.
//!
//! These tests drive the real HTTP path against a local origin so the length
//! and digest gates are exercised end to end rather than mocked away. The
//! catalog entry is a fixture: the shipped entries pin multi-megabyte upstream
//! artifacts, which a test must not download.

use std::io::{BufRead as _, BufReader, Read as _, Write as _};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use cast_embeddings::model_catalog::{LocalModelArtifact, LocalModelEntry, LocalModelKind};
use cast_embeddings::{ModelFetchConfig, ModelFetchError, ModelFetchEvent, ensure_bundle};

const TENSOR_BODY: &[u8] = b"pretend-static-tensor";
const CONFIG_BODY: &[u8] = b"{\"normalize\":true}";
const MANIFEST: &str = "{\n  \"schema_version\": 1\n}\n";

static TEST_ENTRY: LocalModelEntry = LocalModelEntry {
    key: "fixture-model",
    kind: LocalModelKind::Static,
    model_id: "fixture-org/fixture-model",
    revision: "0123456789abcdef0123456789abcdef01234567",
    dimensions: 256,
    embedding_profile: "fixture-profile-v1",
    manifest_file: "static-bundle.json",
    manifest_bytes: MANIFEST,
    artifacts: &[
        LocalModelArtifact {
            path: "model.safetensors",
            sha256: "53a2cfd6e2b61ff2cf7215cefdefb21f66692953f35111e74d3c920544ffdf9d",
            size_bytes: 21,
        },
        LocalModelArtifact {
            path: "config.json",
            sha256: "4f766d1335394ef2d2a53f5de07d2a249723969eaf5ea1e3daf8d465d5f9825a",
            size_bytes: 18,
        },
    ],
    license: "MIT",
};

/// How a test origin answers a request for a given artifact.
#[derive(Clone, Copy)]
enum Reply {
    /// Serve the pinned bytes.
    Correct,
    /// Serve the pinned number of bytes, but not the pinned build.
    Substituted,
    /// Serve fewer bytes than the catalog pins, declared honestly.
    Truncated,
    /// Refuse the request.
    Missing,
}

struct Origin {
    base_url: String,
    requests: Arc<AtomicUsize>,
}

impl Origin {
    fn start(reply: Reply) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test origin");
        let port = listener.local_addr().expect("origin address").port();
        let requests = Arc::new(AtomicUsize::new(0));
        let served = Arc::clone(&requests);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { break };
                served.fetch_add(1, Ordering::SeqCst);
                serve(stream, reply);
            }
        });
        Self {
            base_url: format!("http://127.0.0.1:{port}"),
            requests,
        }
    }

    fn request_count(&self) -> usize {
        self.requests.load(Ordering::SeqCst)
    }
}

fn serve(mut stream: TcpStream, reply: Reply) {
    let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return;
    }
    loop {
        let mut header = String::new();
        match reader.read_line(&mut header) {
            Ok(0) => break,
            Ok(_) if header.trim().is_empty() => break,
            Ok(_) => {}
            Err(_) => return,
        }
    }
    let path = request_line.split_whitespace().nth(1).unwrap_or_default();
    let body: &[u8] = if path.ends_with("model.safetensors") {
        TENSOR_BODY
    } else {
        CONFIG_BODY
    };
    let response = match reply {
        Reply::Missing => {
            b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec()
        }
        Reply::Correct => http_response(body),
        // Exactly as long as the pinned artifact, so only the digest gate can
        // catch it.
        Reply::Substituted => http_response(b"substituted-bytes!!!!"),
        Reply::Truncated => http_response(&body[..body.len() / 2]),
    };
    drop(stream.write_all(&response));
    drop(stream.flush());
}

fn http_response(body: &[u8]) -> Vec<u8> {
    let mut response = format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )
    .into_bytes();
    response.extend_from_slice(body);
    response
}

fn config(cache: &Path, origin: &Origin) -> ModelFetchConfig {
    ModelFetchConfig::new(cache).with_base_url(&origin.base_url)
}

#[tokio::test]
async fn cold_start_provisions_every_artifact_and_the_manifest() {
    let cache = tempfile::tempdir().expect("cache");
    let origin = Origin::start(Reply::Correct);
    let events = Mutex::new(Vec::new());
    let reporter = |event: ModelFetchEvent| {
        events.lock().expect("event log").push(format!("{event:?}"));
    };

    let directory = ensure_bundle(&TEST_ENTRY, &config(cache.path(), &origin), Some(&reporter))
        .await
        .expect("cold provisioning succeeds");

    assert_eq!(
        std::fs::read(directory.join("model.safetensors")).unwrap(),
        TENSOR_BODY
    );
    assert_eq!(
        std::fs::read(directory.join("config.json")).unwrap(),
        CONFIG_BODY
    );
    assert_eq!(
        std::fs::read_to_string(directory.join("static-bundle.json")).unwrap(),
        MANIFEST
    );
    assert_eq!(origin.request_count(), 2, "one request per artifact");

    let log = events.lock().expect("event log").join("\n");
    assert!(log.contains("Started"), "provisioning start is reported");
    assert!(log.contains("Finished"), "completion is reported");
    assert_eq!(
        log.matches("Verified").count(),
        2,
        "every artifact reports verification"
    );

    // No partial files survive a successful run.
    let leftovers = std::fs::read_dir(&directory)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_name().to_string_lossy().contains("partial"))
        .count();
    assert_eq!(leftovers, 0);
}

#[tokio::test]
async fn a_provisioned_bundle_is_reused_without_touching_the_origin() {
    let cache = tempfile::tempdir().expect("cache");
    let origin = Origin::start(Reply::Correct);
    ensure_bundle(&TEST_ENTRY, &config(cache.path(), &origin), None)
        .await
        .expect("cold provisioning succeeds");
    let after_cold = origin.request_count();

    let events = Mutex::new(Vec::new());
    let reporter = |event: ModelFetchEvent| {
        events.lock().expect("event log").push(format!("{event:?}"));
    };
    ensure_bundle(&TEST_ENTRY, &config(cache.path(), &origin), Some(&reporter))
        .await
        .expect("warm start succeeds");

    assert_eq!(origin.request_count(), after_cold, "warm start is offline");
    assert!(
        events.lock().expect("event log")[0].contains("Cached"),
        "a warm start reports the cache hit"
    );
}

#[tokio::test]
async fn substituted_bytes_are_rejected_and_never_committed() {
    let cache = tempfile::tempdir().expect("cache");
    let origin = Origin::start(Reply::Substituted);

    let error = ensure_bundle(&TEST_ENTRY, &config(cache.path(), &origin), None)
        .await
        .expect_err("a mirror serving different bytes must fail");

    assert!(
        matches!(error, ModelFetchError::Checksum { path, .. } if path == "model.safetensors"),
        "expected a checksum rejection, got {error:?}"
    );
    let directory = config(cache.path(), &origin).bundle_dir(&TEST_ENTRY);
    assert!(
        !directory.join("model.safetensors").exists(),
        "rejected bytes must not reach the artifact name"
    );
    assert!(
        !directory.join("static-bundle.json").exists(),
        "a failed run must not publish a manifest"
    );
}

#[tokio::test]
async fn short_transfers_are_rejected() {
    let cache = tempfile::tempdir().expect("cache");
    let origin = Origin::start(Reply::Truncated);

    let error = ensure_bundle(&TEST_ENTRY, &config(cache.path(), &origin), None)
        .await
        .expect_err("a truncated artifact must fail");

    assert!(
        matches!(error, ModelFetchError::Length { expected: 21, .. }),
        "expected a length rejection, got {error:?}"
    );
}

#[tokio::test]
async fn origin_failures_surface_the_status() {
    let cache = tempfile::tempdir().expect("cache");
    let origin = Origin::start(Reply::Missing);

    let error = ensure_bundle(&TEST_ENTRY, &config(cache.path(), &origin), None)
        .await
        .expect_err("a 404 must fail");

    assert!(
        matches!(error, ModelFetchError::Status { status: 404, .. }),
        "expected a status rejection, got {error:?}"
    );
}

#[tokio::test]
async fn downloads_can_be_refused_but_an_existing_bundle_still_opens() {
    let cache = tempfile::tempdir().expect("cache");
    let origin = Origin::start(Reply::Correct);
    let offline = config(cache.path(), &origin).with_allow_download(false);

    let error = ensure_bundle(&TEST_ENTRY, &offline, None)
        .await
        .expect_err("provisioning must not download when it is turned off");
    assert!(
        matches!(error, ModelFetchError::DownloadDisabled { .. }),
        "expected a disabled-download rejection, got {error:?}"
    );
    assert_eq!(origin.request_count(), 0, "nothing may reach the network");

    ensure_bundle(&TEST_ENTRY, &config(cache.path(), &origin), None)
        .await
        .expect("provisioning succeeds when downloads are allowed");
    ensure_bundle(&TEST_ENTRY, &offline, None)
        .await
        .expect("an already provisioned bundle opens with downloads off");
}

#[tokio::test]
async fn a_corrupted_cache_entry_is_refetched() {
    let cache = tempfile::tempdir().expect("cache");
    let origin = Origin::start(Reply::Correct);
    let directory = ensure_bundle(&TEST_ENTRY, &config(cache.path(), &origin), None)
        .await
        .expect("cold provisioning succeeds");
    std::fs::write(directory.join("model.safetensors"), b"truncated").unwrap();
    let after_cold = origin.request_count();

    ensure_bundle(&TEST_ENTRY, &config(cache.path(), &origin), None)
        .await
        .expect("a damaged artifact is replaced");

    assert_eq!(
        std::fs::read(directory.join("model.safetensors")).unwrap(),
        TENSOR_BODY
    );
    assert_eq!(
        origin.request_count(),
        after_cold + 1,
        "only the damaged artifact is refetched"
    );
}

/// Reading the whole body is what the fetcher does; this guards the fixture
/// origin itself so a broken test server cannot look like a product failure.
#[test]
fn fixture_origin_serves_the_pinned_bytes() {
    let origin = Origin::start(Reply::Correct);
    let mut stream = TcpStream::connect(origin.base_url.trim_start_matches("http://")).unwrap();
    stream
        .write_all(b"GET /model.safetensors HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).unwrap();
    assert!(response.ends_with(TENSOR_BODY));
}
