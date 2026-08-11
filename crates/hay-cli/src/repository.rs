use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Split};
use std::path::{Component, Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use cast_core::LanguageId;
use cast_index::{DocumentId, NormalizedPath};
use hay_search::{Chunker, ChunkerV1, CorpusDocument, SearchDocument, SearchError};
use ring::digest::{Context as DigestContext, SHA256, digest};
use serde::{Deserialize, Serialize};

const CHECKPOINT_VERSION: u32 = 2;

const BINARY_PROBE_BYTES: u64 = 8 * 1024;
const MAX_CONFIG_BYTES: u64 = 50 * 1024;
const MAX_SOURCE_BYTES: u64 = 5 * 1024 * 1024;
const MAX_SOURCE_BYTES_USIZE: usize = 5 * 1024 * 1024;
const DATA_HEURISTIC_MIN_BYTES: u64 = 750 * 1024;
const DATA_HEURISTIC_MAX_AVERAGE_LINE: usize = 100;

#[derive(Clone, Debug, Default, Serialize)]
pub struct RepositoryStats {
    pub files_seen: usize,
    pub files_indexed: usize,
    pub chunks_indexed: usize,
    pub files_unchanged: usize,
    pub chunks_reused: usize,
    pub chunks_deleted: usize,
    pub blank_chunks_skipped: usize,
    pub source_bytes: u64,
    pub discover_ms: u64,
    pub read_ms: u64,
    pub chunk_ms: u64,
    pub skipped: BTreeMap<&'static str, usize>,
}

#[derive(Clone, Debug)]
pub struct RepositoryProgress(Arc<Mutex<RepositoryProgressState>>);

#[derive(Debug)]
struct RepositoryProgressState {
    statistics: RepositoryStats,
    checkpoint: RepositoryCheckpoint,
    previous: BTreeMap<String, FileCheckpoint>,
    deletions: BTreeSet<DocumentId>,
    discover_time: Duration,
    read_time: Duration,
    chunk_time: Duration,
    last_completion: Instant,
    report_interval: Option<Duration>,
    last_report: Instant,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RepositoryCheckpoint {
    version: u32,
    root: String,
    fingerprint: String,
    manifest_hash: String,
    files: BTreeMap<String, FileCheckpoint>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct FileCheckpoint {
    content_hash: String,
    document_ids: Vec<DocumentId>,
}

impl RepositoryProgress {
    pub fn snapshot(&self) -> RepositoryStats {
        self.0.lock().map_or_else(
            |_| RepositoryStats::default(),
            |state| statistics_snapshot(&state),
        )
    }

    pub fn inactive_for(&self) -> Result<Duration> {
        self.0
            .lock()
            .map(|state| state.last_completion.elapsed())
            .map_err(|_| anyhow::anyhow!("repository progress lock was poisoned"))
    }

    pub fn configure_progress_reporting(&self, interval: Duration) -> Result<()> {
        self.0
            .lock()
            .map(|mut state| {
                state.report_interval = Some(interval);
                state.last_report = Instant::now();
            })
            .map_err(|_| anyhow::anyhow!("repository progress lock was poisoned"))
    }

    pub fn emit_progress_if_due(&self) {
        let report = self.0.lock().ok().and_then(|mut state| {
            let interval = state.report_interval?;
            if state.last_report.elapsed() < interval {
                return None;
            }
            state.last_report = Instant::now();
            Some((
                statistics_snapshot(&state),
                duration_millis(state.last_completion.elapsed()),
            ))
        });
        if let Some((statistics, inactive_ms)) = report {
            let event = serde_json::json!({
                "event": "index_progress",
                "inactive_ms": inactive_ms,
                "repository": statistics,
            });
            if let Ok(encoded) = serde_json::to_string(&event) {
                eprintln!("{encoded}");
            }
        }
    }

    pub fn checkpoint(&self) -> Result<RepositoryCheckpoint> {
        self.0
            .lock()
            .map(|state| state.checkpoint.clone())
            .map_err(|_| anyhow::anyhow!("repository checkpoint lock was poisoned"))
    }

    pub fn deletions(&self) -> Result<Vec<DocumentId>, SearchError> {
        self.0
            .lock()
            .map(|state| state.deletions.iter().cloned().collect())
            .map_err(|_| SearchError::Corpus("repository checkpoint lock was poisoned".into()))
    }

    fn update(&self, operation: impl FnOnce(&mut RepositoryStats)) {
        if let Ok(mut state) = self.0.lock() {
            operation(&mut state.statistics);
        }
    }

    fn skip(&self, reason: &'static str) {
        if let Ok(mut state) = self.0.lock() {
            *state.statistics.skipped.entry(reason).or_default() += 1;
            state.last_completion = Instant::now();
        }
        self.emit_progress_if_due();
    }
    fn begin_file(&self, path: &str) -> Option<FileCheckpoint> {
        self.0.lock().ok()?.previous.remove(path)
    }

    fn reuse(&self, path: String, previous: FileCheckpoint) {
        if let Ok(mut state) = self.0.lock() {
            state.statistics.files_unchanged += 1;
            state.statistics.chunks_reused = state
                .statistics
                .chunks_reused
                .saturating_add(previous.document_ids.len());
            state.checkpoint.files.insert(path, previous);
            state.last_completion = Instant::now();
        }
        self.emit_progress_if_due();
    }

    fn replace(&self, path: String, previous: Option<FileCheckpoint>, next: FileCheckpoint) {
        if let Ok(mut state) = self.0.lock() {
            if let Some(previous) = previous {
                state.deletions.extend(previous.document_ids);
            }
            state.checkpoint.files.insert(path, next);
            state.last_completion = Instant::now();
        }
        self.emit_progress_if_due();
    }

    fn discard(&self, previous: Option<FileCheckpoint>) {
        if let (Ok(mut state), Some(previous)) = (self.0.lock(), previous) {
            state.deletions.extend(previous.document_ids);
        }
    }

    fn finish(&self) {
        if let Ok(mut state) = self.0.lock() {
            let remaining = std::mem::take(&mut state.previous);
            for file in remaining.into_values() {
                state.deletions.extend(file.document_ids);
            }
            state.statistics.chunks_deleted = state.deletions.len();
            state.last_completion = Instant::now();
        }
        self.emit_progress_if_due();
    }

    fn record_duration(&self, phase: TimedPhase, duration: Duration) {
        if let Ok(mut state) = self.0.lock() {
            match phase {
                TimedPhase::Discover => state.discover_time += duration,
                TimedPhase::Read => state.read_time += duration,
                TimedPhase::Chunk => state.chunk_time += duration,
            }
        }
    }
}

#[derive(Clone, Copy)]
enum TimedPhase {
    Discover,
    Read,
    Chunk,
}

struct PhaseTimer {
    progress: RepositoryProgress,
    phase: TimedPhase,
    started: Instant,
}

impl PhaseTimer {
    fn start(progress: &RepositoryProgress, phase: TimedPhase) -> Self {
        Self {
            progress: progress.clone(),
            phase,
            started: Instant::now(),
        }
    }
}

impl Drop for PhaseTimer {
    fn drop(&mut self) {
        self.progress
            .record_duration(self.phase, self.started.elapsed());
    }
}

impl RepositoryCheckpoint {
    pub fn from_reader(reader: impl Read) -> Result<Self> {
        serde_json::from_reader(reader).context("parse repository checkpoint")
    }

    pub fn matches(&self, root: &Path, manifest: &hay_search::IndexManifest) -> Result<bool> {
        let root = root
            .canonicalize()
            .with_context(|| format!("canonicalize repository source {}", root.display()))?;
        Ok(self.version == CHECKPOINT_VERSION
            && self.root == root.to_string_lossy()
            && self.fingerprint == fingerprint_hex(manifest)?
            && self.manifest_hash == manifest_hex(manifest)?)
    }
}

pub struct RepositoryChunkStream {
    root: PathBuf,
    paths: RepositoryPaths,
    chunker: ChunkerV1,
    pending: VecDeque<SearchDocument>,
    progress: RepositoryProgress,
    fingerprint: Vec<u8>,
    finished: bool,
}

#[derive(Serialize)]
struct DocumentFingerprint<'a> {
    model_id: &'a str,
    model_revision: &'a str,
    embedding_profile: &'a str,
    embed_dim: usize,
    tokenizer_hash: &'a cast_index::ContentHash,
    chunker_version: &'a str,
    fde_params: &'a hay_search::FdeParams,
    schema_version: u32,
}

impl RepositoryChunkStream {
    #[cfg(test)]
    fn open(
        root: &Path,
        manifest: &hay_search::IndexManifest,
    ) -> Result<(Self, RepositoryProgress)> {
        Self::open_incremental(root, manifest, None)
    }

    pub fn open_incremental(
        root: &Path,
        manifest: &hay_search::IndexManifest,
        previous: Option<RepositoryCheckpoint>,
    ) -> Result<(Self, RepositoryProgress)> {
        if !root.is_dir() {
            bail!("repository source {} is not a directory", root.display());
        }
        let chunker = ChunkerV1::default();
        let chunker_profile = chunker.profile_id();
        if manifest.chunker_version != chunker_profile {
            bail!(
                "repository manifest chunker identity does not match the executable profile: manifest {:?}, executable {:?}",
                manifest.chunker_version,
                chunker_profile
            );
        }
        let root = root
            .canonicalize()
            .with_context(|| format!("canonicalize repository source {}", root.display()))?;
        let paths = RepositoryPaths::open(&root)?;
        let fingerprint = fingerprint_bytes(manifest)?;
        let fingerprint_hex = hex(&fingerprint);
        let previous_files = previous.map_or_else(BTreeMap::new, |state| state.files);
        let progress = RepositoryProgress(Arc::new(Mutex::new(RepositoryProgressState {
            statistics: RepositoryStats::default(),
            checkpoint: RepositoryCheckpoint {
                version: CHECKPOINT_VERSION,
                root: root.to_string_lossy().into_owned(),
                fingerprint: fingerprint_hex,
                manifest_hash: manifest_hex(manifest)?,
                files: BTreeMap::new(),
            },
            previous: previous_files,
            deletions: BTreeSet::new(),
            discover_time: Duration::ZERO,
            read_time: Duration::ZERO,
            chunk_time: Duration::ZERO,
            last_completion: Instant::now(),
            report_interval: None,
            last_report: Instant::now(),
        })));
        Ok((
            Self {
                root,
                paths,
                chunker,
                pending: VecDeque::new(),
                progress: progress.clone(),
                fingerprint,
                finished: false,
            },
            progress,
        ))
    }

    fn load_file(&mut self, relative: &Path) -> Result<(), SearchError> {
        let read_timer = PhaseTimer::start(&self.progress, TimedPhase::Read);
        self.progress
            .update(|statistics| statistics.files_seen += 1);
        let relative_key = relative.to_string_lossy().replace('\\', "/");
        let previous = self.progress.begin_file(&relative_key);
        if has_hidden_component(relative) {
            self.progress.skip("hidden");
            self.progress.discard(previous);
            return Ok(());
        }
        let absolute = self.root.join(relative);
        let metadata = fs::symlink_metadata(&absolute)
            .map_err(|error| corpus_error(&absolute, "read metadata", error))?;
        if metadata.file_type().is_symlink() {
            self.progress.skip("symlink");
            self.progress.discard(previous);
            return Ok(());
        }
        if !metadata.is_file() {
            self.progress.skip("not_file");
            self.progress.discard(previous);
            return Ok(());
        }
        if is_config_or_data(relative) && metadata.len() > MAX_CONFIG_BYTES {
            self.progress.skip("oversized_config");
            self.progress.discard(previous);
            return Ok(());
        }
        if metadata.len() > MAX_SOURCE_BYTES {
            self.progress.skip("oversized_source");
            self.progress.discard(previous);
            return Ok(());
        }
        let Some(language) = language_for_path(relative) else {
            self.progress.skip("unsupported_language");
            self.progress.discard(previous);
            return Ok(());
        };

        let mut file =
            File::open(&absolute).map_err(|error| corpus_error(&absolute, "open source", error))?;
        let mut bytes =
            Vec::with_capacity(usize::try_from(metadata.len()).unwrap_or(MAX_SOURCE_BYTES_USIZE));
        (&mut file)
            .take(BINARY_PROBE_BYTES)
            .read_to_end(&mut bytes)
            .map_err(|error| corpus_error(&absolute, "read binary probe", error))?;
        if bytes.contains(&0) {
            self.progress.skip("binary");
            self.progress.discard(previous);
            return Ok(());
        }
        if metadata.len() > DATA_HEURISTIC_MIN_BYTES
            && average_line_length(&bytes) > DATA_HEURISTIC_MAX_AVERAGE_LINE
        {
            self.progress.skip("data_like");
            self.progress.discard(previous);
            return Ok(());
        }
        file.read_to_end(&mut bytes)
            .map_err(|error| corpus_error(&absolute, "read source", error))?;
        if bytes.len() > MAX_SOURCE_BYTES_USIZE {
            self.progress.skip("oversized_source");
            self.progress.discard(previous);
            return Ok(());
        }
        let content_hash = digest(&SHA256, &bytes);
        let content_hash_hex = hex(content_hash.as_ref());
        self.progress.update(|statistics| {
            statistics.source_bytes = statistics
                .source_bytes
                .saturating_add(u64::try_from(bytes.len()).unwrap_or(u64::MAX));
        });
        if let Some(state) = previous
            .as_ref()
            .filter(|state| state.content_hash == content_hash_hex)
        {
            self.progress.reuse(relative_key, state.clone());
            return Ok(());
        }
        let Ok(text) = String::from_utf8(bytes) else {
            self.progress.skip("invalid_utf8");
            self.progress.discard(previous);
            return Ok(());
        };
        if text.trim().is_empty() {
            self.progress.skip("empty");
            self.progress.discard(previous);
            return Ok(());
        }
        drop(read_timer);
        let _chunk_timer = PhaseTimer::start(&self.progress, TimedPhase::Chunk);
        self.queue_chunks(
            relative,
            language,
            text,
            content_hash.as_ref(),
            content_hash_hex,
            previous,
        )
    }

    fn queue_chunks(
        &mut self,
        relative: &Path,
        language: &str,
        text: String,
        content_hash: &[u8],
        content_hash_hex: String,
        previous: Option<FileCheckpoint>,
    ) -> Result<(), SearchError> {
        let path = NormalizedPath::new(relative.to_string_lossy().replace('\\', "/"))
            .map_err(|error| SearchError::Corpus(error.to_string()))?;
        let source = CorpusDocument {
            doc_id: DocumentId::new(path.as_str())
                .map_err(|error| SearchError::Corpus(error.to_string()))?,
            path: path.clone(),
            language: LanguageId::new(language),
            text,
        };
        let chunks = self.chunker.chunk(&source)?;
        let original_chunk_count = chunks.len();
        let chunks = chunks
            .into_iter()
            .filter(|chunk| !chunk.text.trim().is_empty())
            .collect::<Vec<_>>();
        let blank_chunks = original_chunk_count.saturating_sub(chunks.len());
        self.progress.update(|statistics| {
            statistics.blank_chunks_skipped =
                statistics.blank_chunks_skipped.saturating_add(blank_chunks);
        });
        if chunks.is_empty() {
            self.progress.skip("empty_chunks");
            self.progress.discard(previous);
            return Ok(());
        }
        let chunk_count = chunks.len();
        let documents = chunks
            .into_iter()
            .map(|chunk| {
                Ok(SearchDocument {
                    doc_id: document_id(
                        &path,
                        content_hash,
                        &self.fingerprint,
                        chunk.ordinal,
                        chunk.core_range.start_byte,
                        chunk.core_range.end_byte,
                    )?,
                    path: path.clone(),
                    language: chunk.language,
                    text: chunk.text,
                    embedding: None,
                })
            })
            .collect::<Result<Vec<_>, SearchError>>()?;
        let document_ids = documents
            .iter()
            .map(|document| document.doc_id.clone())
            .collect();
        self.pending.extend(documents);
        self.progress.replace(
            path.as_str().to_owned(),
            previous,
            FileCheckpoint {
                content_hash: content_hash_hex,
                document_ids,
            },
        );
        self.progress.update(|statistics| {
            statistics.files_indexed += 1;
            statistics.chunks_indexed = statistics.chunks_indexed.saturating_add(chunk_count);
        });
        Ok(())
    }
}

impl Iterator for RepositoryChunkStream {
    type Item = Result<SearchDocument, SearchError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(document) = self.pending.pop_front() {
                return Some(Ok(document));
            }
            let discovered = Instant::now();
            let next_path = self.paths.next();
            self.progress
                .record_duration(TimedPhase::Discover, discovered.elapsed());
            let relative = match next_path {
                Some(Ok(relative)) => relative,
                Some(Err(error)) => return Some(Err(error)),
                None => {
                    if !self.finished {
                        self.finished = true;
                        self.progress.finish();
                    }
                    return None;
                }
            };
            if let Err(error) = self.load_file(&relative) {
                return Some(Err(error));
            }
        }
    }
}

fn fingerprint_bytes(manifest: &hay_search::IndexManifest) -> Result<Vec<u8>> {
    let fingerprint = DocumentFingerprint {
        model_id: &manifest.model_id,
        model_revision: &manifest.model_revision,
        embedding_profile: &manifest.embedding_profile,
        embed_dim: manifest.embed_dim,
        tokenizer_hash: &manifest.tokenizer_hash,
        chunker_version: &manifest.chunker_version,
        fde_params: &manifest.fde_params,
        schema_version: manifest.schema_version,
    };
    let encoded =
        serde_json::to_vec(&fingerprint).context("serialize document relevance fingerprint")?;
    Ok(digest(&SHA256, &encoded).as_ref().to_vec())
}

fn fingerprint_hex(manifest: &hay_search::IndexManifest) -> Result<String> {
    Ok(hex(&fingerprint_bytes(manifest)?))
}

fn manifest_hex(manifest: &hay_search::IndexManifest) -> Result<String> {
    let encoded = serde_json::to_vec(manifest).context("serialize complete index manifest")?;
    Ok(hex(digest(&SHA256, &encoded).as_ref()))
}

fn hex(bytes: &[u8]) -> String {
    const LOWER_HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(char::from(LOWER_HEX[usize::from(*byte >> 4)]));
        encoded.push(char::from(LOWER_HEX[usize::from(*byte & 0x0f)]));
    }
    encoded
}

fn duration_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn statistics_snapshot(state: &RepositoryProgressState) -> RepositoryStats {
    let mut statistics = state.statistics.clone();
    statistics.discover_ms = duration_millis(state.discover_time);
    statistics.read_ms = duration_millis(state.read_time);
    statistics.chunk_ms = duration_millis(state.chunk_time);
    statistics
}

enum RepositoryPaths {
    Git(GitPaths),
    Walk(WalkPaths),
}

impl RepositoryPaths {
    fn open(root: &Path) -> Result<Self> {
        match git_root(root)? {
            Some(git_root) if git_root == root => Ok(Self::Git(GitPaths::spawn(root)?)),
            Some(git_root) => bail!(
                "repository source {} is inside {}; pass the Git root explicitly",
                root.display(),
                git_root.display()
            ),
            None => Ok(Self::Walk(WalkPaths::new(root)?)),
        }
    }
}

impl Iterator for RepositoryPaths {
    type Item = Result<PathBuf, SearchError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Git(paths) => paths.next(),
            Self::Walk(paths) => paths.next(),
        }
    }
}

struct GitPaths {
    child: Child,
    paths: Split<BufReader<ChildStdout>>,
    finished: bool,
}

impl GitPaths {
    fn spawn(root: &Path) -> Result<Self> {
        let mut child = Command::new("git")
            .arg("-C")
            .arg(root)
            .args([
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("enumerate Git files in {}", root.display()))?;
        let stdout = child
            .stdout
            .take()
            .context("Git file enumeration has no stdout")?;
        Ok(Self {
            child,
            paths: BufReader::new(stdout).split(0),
            finished: false,
        })
    }
}

impl Iterator for GitPaths {
    type Item = Result<PathBuf, SearchError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        match self.paths.next() {
            Some(Ok(path)) => Some(
                String::from_utf8(path)
                    .map(PathBuf::from)
                    .map_err(|_| SearchError::Corpus("Git returned a non-UTF-8 path".into())),
            ),
            Some(Err(error)) => Some(Err(SearchError::Corpus(format!(
                "read Git file enumeration: {error}"
            )))),
            None => {
                self.finished = true;
                match self.child.wait() {
                    Ok(status) if status.success() => None,
                    Ok(status) => Some(Err(SearchError::Corpus(format!(
                        "git ls-files exited with {status}"
                    )))),
                    Err(error) => Some(Err(SearchError::Corpus(format!(
                        "wait for git ls-files: {error}"
                    )))),
                }
            }
        }
    }
}

impl Drop for GitPaths {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

struct WalkPaths {
    root: PathBuf,
    directories: Vec<SortedDirectory>,
}

struct SortedDirectory {
    entries: std::vec::IntoIter<fs::DirEntry>,
}

impl WalkPaths {
    fn new(root: &Path) -> Result<Self> {
        Ok(Self {
            root: root.to_owned(),
            directories: vec![read_sorted_directory(root)?],
        })
    }
}

impl Iterator for WalkPaths {
    type Item = Result<PathBuf, SearchError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let entry = {
                let directory = self.directories.last_mut()?;
                if let Some(entry) = directory.entries.next() {
                    entry
                } else {
                    self.directories.pop();
                    continue;
                }
            };
            let file_type = match entry.file_type() {
                Ok(file_type) => file_type,
                Err(error) => {
                    return Some(Err(corpus_error(&entry.path(), "inspect path", error)));
                }
            };
            let relative = match entry.path().strip_prefix(&self.root) {
                Ok(relative) => relative.to_owned(),
                Err(error) => {
                    return Some(Err(SearchError::Corpus(format!(
                        "make {} repository-relative: {error}",
                        entry.path().display()
                    ))));
                }
            };
            if has_hidden_component(&relative) || file_type.is_symlink() {
                continue;
            }
            if file_type.is_dir() {
                match read_sorted_directory(&entry.path()) {
                    Ok(directory) => self.directories.push(directory),
                    Err(error) => return Some(Err(SearchError::Corpus(error.to_string()))),
                }
                continue;
            }
            if file_type.is_file() {
                return Some(Ok(relative));
            }
        }
    }
}

fn read_sorted_directory(path: &Path) -> Result<SortedDirectory> {
    let mut entries = fs::read_dir(path)
        .with_context(|| format!("read directory {}", path.display()))?
        .collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(fs::DirEntry::file_name);
    Ok(SortedDirectory {
        entries: entries.into_iter(),
    })
}

fn git_root(root: &Path) -> Result<Option<PathBuf>> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .with_context(|| format!("detect Git repository for {}", root.display()))?;
    if !output.status.success() {
        return Ok(None);
    }
    let value = String::from_utf8(output.stdout).context("Git root is not UTF-8")?;
    Ok(Some(
        PathBuf::from(value.trim())
            .canonicalize()
            .context("canonicalize Git root")?,
    ))
}

fn has_hidden_component(path: &Path) -> bool {
    path.components().any(|component| {
        matches!(component, Component::Normal(value) if value.to_string_lossy().starts_with('.'))
    })
}

fn is_config_or_data(path: &Path) -> bool {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .map(str::to_ascii_lowercase);
    matches!(
        extension.as_deref(),
        Some(
            "json"
                | "yaml"
                | "yml"
                | "toml"
                | "ini"
                | "cfg"
                | "conf"
                | "properties"
                | "env"
                | "hcl"
                | "tf"
                | "tfvars"
        )
    )
}

fn language_for_path(path: &Path) -> Option<&'static str> {
    let file_name = path.file_name()?.to_str()?;
    if file_name.eq_ignore_ascii_case("Dockerfile") {
        return Some("dockerfile");
    }
    let extension = path.extension()?.to_str()?.to_ascii_lowercase();
    match extension.as_str() {
        "c" | "h" => Some("c"),
        "cc" | "cpp" | "cxx" | "hh" | "hpp" | "hxx" => Some("cpp"),
        "cs" => Some("csharp"),
        "css" => Some("css"),
        "cue" => Some("cue"),
        "ex" | "exs" => Some("elixir"),
        "elm" => Some("elm"),
        "go" => Some("go"),
        "groovy" | "gradle" => Some("groovy"),
        "hcl" | "tf" | "tfvars" => Some("hcl"),
        "html" | "htm" => Some("html"),
        "java" => Some("java"),
        "js" | "jsx" | "mjs" | "cjs" => Some("javascript"),
        "json" => Some("json"),
        "kt" | "kts" => Some("kotlin"),
        "lua" => Some("lua"),
        "md" | "markdown" => Some("markdown"),
        "ml" | "mli" => Some("ocaml"),
        "php" | "phtml" => Some("php"),
        "proto" => Some("protobuf"),
        "py" | "pyi" | "pyw" => Some("python"),
        "rb" | "rake" | "gemspec" => Some("ruby"),
        "rs" => Some("rust"),
        "scala" | "sc" => Some("scala"),
        "sh" | "bash" | "zsh" => Some("bash"),
        "sql" => Some("sql"),
        "svelte" => Some("svelte"),
        "swift" => Some("swift"),
        "toml" => Some("toml"),
        "ts" => Some("typescript"),
        "tsx" => Some("tsx"),
        "yaml" | "yml" => Some("yaml"),
        _ => None,
    }
}

fn average_line_length(probe: &[u8]) -> usize {
    let lines = probe
        .iter()
        .fold(0_usize, |count, byte| count + usize::from(*byte == b'\n'))
        .max(1);
    probe.len() / lines
}

fn document_id(
    path: &NormalizedPath,
    content_hash: &[u8],
    fingerprint: &[u8],
    ordinal: usize,
    core_start: usize,
    core_end: usize,
) -> Result<DocumentId, SearchError> {
    let ordinal = u64::try_from(ordinal)
        .map_err(|_| SearchError::Corpus("chunk ordinal exceeds u64".into()))?;
    let core_start = u64::try_from(core_start)
        .map_err(|_| SearchError::Corpus("chunk start exceeds u64".into()))?;
    let core_end =
        u64::try_from(core_end).map_err(|_| SearchError::Corpus("chunk end exceeds u64".into()))?;
    let mut identity = DigestContext::new(&SHA256);
    identity.update(b"hay-document-id-v1\0");
    identity.update(path.as_str().as_bytes());
    identity.update(b"\0");
    identity.update(content_hash);
    identity.update(fingerprint);
    identity.update(&ordinal.to_be_bytes());
    identity.update(&core_start.to_be_bytes());
    identity.update(&core_end.to_be_bytes());
    let digest = identity.finish();
    let mut encoded = String::with_capacity(digest.as_ref().len() * 2);
    for byte in digest.as_ref() {
        use std::fmt::Write as _;
        write!(encoded, "{byte:02x}").map_err(|error| SearchError::Corpus(error.to_string()))?;
    }
    DocumentId::new(encoded).map_err(|error| SearchError::Corpus(error.to_string()))
}

fn corpus_error(path: &Path, operation: &str, error: impl std::fmt::Display) -> SearchError {
    SearchError::Corpus(format!("{operation} {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn non_git_stream_filters_and_chunks_without_loading_the_repository() {
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("main.rs"), "fn main() {}\n").unwrap();
        fs::write(directory.path().join("unknown.bin"), "not source").unwrap();
        fs::create_dir(directory.path().join(".hidden")).unwrap();
        fs::write(
            directory.path().join(".hidden/secret.rs"),
            "fn secret() {}\n",
        )
        .unwrap();

        let (stream, progress) =
            RepositoryChunkStream::open(directory.path(), &hay_search::IndexManifest::lexical_v1())
                .unwrap();
        let documents = stream.collect::<Result<Vec<_>, _>>().unwrap();
        let statistics = progress.snapshot();

        assert_eq!(documents.len(), 1);
        assert_eq!(documents[0].path.as_str(), "main.rs");
        assert_eq!(documents[0].doc_id.as_str().len(), 64);
        assert_eq!(statistics.files_indexed, 1);
        assert_eq!(statistics.chunks_indexed, 1);
        assert_eq!(statistics.source_bytes, 13);
        assert_eq!(statistics.blank_chunks_skipped, 0);
        assert_eq!(statistics.skipped.get("unsupported_language"), Some(&1));
    }

    #[test]
    fn binary_probe_and_config_limit_are_observable() {
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("binary.rs"), b"fn main() {}\0tail").unwrap();
        let mut config = File::create(directory.path().join("large.json")).unwrap();
        config.write_all(&vec![b'x'; 50 * 1024 + 1]).unwrap();

        let (stream, progress) =
            RepositoryChunkStream::open(directory.path(), &hay_search::IndexManifest::lexical_v1())
                .unwrap();
        let documents = stream.collect::<Result<Vec<_>, _>>().unwrap();
        let statistics = progress.snapshot();

        assert!(documents.is_empty());
        assert_eq!(statistics.skipped.get("binary"), Some(&1));
        assert_eq!(statistics.skipped.get("oversized_config"), Some(&1));
    }

    #[test]
    fn git_enumeration_honors_standard_ignores() {
        let directory = tempdir().unwrap();
        let status = Command::new("git")
            .arg("init")
            .arg("--quiet")
            .arg(directory.path())
            .status()
            .unwrap();
        assert!(status.success());
        fs::write(directory.path().join(".gitignore"), "ignored.rs\n").unwrap();
        fs::write(directory.path().join("ignored.rs"), "fn ignored() {}\n").unwrap();
        fs::write(directory.path().join("visible.rs"), "fn visible() {}\n").unwrap();

        let (stream, _) =
            RepositoryChunkStream::open(directory.path(), &hay_search::IndexManifest::lexical_v1())
                .unwrap();
        let documents = stream.collect::<Result<Vec<_>, _>>().unwrap();

        assert_eq!(documents.len(), 1);
        assert_eq!(documents[0].path.as_str(), "visible.rs");
    }

    #[test]
    fn document_identity_changes_with_content_and_manifest() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("main.rs");
        let manifest = hay_search::IndexManifest::lexical_v1();
        fs::write(&path, "fn first() {}\n").unwrap();
        let (stream, _) = RepositoryChunkStream::open(directory.path(), &manifest).unwrap();
        let first = stream.collect::<Result<Vec<_>, _>>().unwrap()[0]
            .doc_id
            .clone();

        fs::write(&path, "fn second() {}\n").unwrap();
        let (stream, _) = RepositoryChunkStream::open(directory.path(), &manifest).unwrap();
        let changed_content = stream.collect::<Result<Vec<_>, _>>().unwrap()[0]
            .doc_id
            .clone();
        let changed_manifest = hay_search::IndexManifest {
            model_revision: "lexical-bm25-v3-test".into(),
            ..manifest
        };
        let (stream, changed_progress) =
            RepositoryChunkStream::open(directory.path(), &changed_manifest).unwrap();
        let changed_fingerprint = stream.collect::<Result<Vec<_>, _>>().unwrap()[0]
            .doc_id
            .clone();

        assert_ne!(first, changed_content);
        assert_ne!(changed_content, changed_fingerprint);

        let storage_only_manifest = hay_search::IndexManifest {
            quantization: hay_search::Quantization::ElasticBbq,
            ..changed_manifest
        };
        let (stream, _) =
            RepositoryChunkStream::open(directory.path(), &storage_only_manifest).unwrap();
        let storage_only = stream.collect::<Result<Vec<_>, _>>().unwrap()[0]
            .doc_id
            .clone();
        assert_eq!(changed_fingerprint, storage_only);
        assert!(
            !changed_progress
                .checkpoint()
                .unwrap()
                .matches(directory.path(), &storage_only_manifest)
                .unwrap()
        );
    }

    #[test]
    fn repository_stream_rejects_a_manifest_for_another_chunker() {
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("main.rs"), "fn main() {}\n").unwrap();
        let manifest = hay_search::IndexManifest {
            chunker_version: "cast-v1".into(),
            ..hay_search::IndexManifest::lexical_v1()
        };

        let error = RepositoryChunkStream::open(directory.path(), &manifest)
            .err()
            .expect("mismatched chunker must fail before indexing");
        assert!(error.to_string().contains("executable profile"));
    }

    #[test]
    fn checkpoint_reuses_unchanged_chunks_and_reports_changes_and_deletes() {
        let directory = tempdir().unwrap();
        let manifest = hay_search::IndexManifest::lexical_v1();
        let main = directory.path().join("main.rs");
        fs::write(&main, "fn first() {}\n").unwrap();

        let (stream, first_progress) =
            RepositoryChunkStream::open_incremental(directory.path(), &manifest, None).unwrap();
        let first = stream.collect::<Result<Vec<_>, _>>().unwrap();
        let first_id = first[0].doc_id.clone();
        let first_checkpoint = first_progress.checkpoint().unwrap();

        let (stream, unchanged_progress) = RepositoryChunkStream::open_incremental(
            directory.path(),
            &manifest,
            Some(first_checkpoint.clone()),
        )
        .unwrap();
        assert!(stream.collect::<Result<Vec<_>, _>>().unwrap().is_empty());
        assert!(unchanged_progress.deletions().unwrap().is_empty());
        assert_eq!(unchanged_progress.snapshot().files_unchanged, 1);
        assert_eq!(unchanged_progress.snapshot().chunks_reused, 1);

        fs::write(&main, "fn second() {}\n").unwrap();
        let (stream, changed_progress) = RepositoryChunkStream::open_incremental(
            directory.path(),
            &manifest,
            Some(first_checkpoint),
        )
        .unwrap();
        let changed = stream.collect::<Result<Vec<_>, _>>().unwrap();
        assert_ne!(changed[0].doc_id, first_id);
        assert_eq!(changed_progress.deletions().unwrap(), vec![first_id]);

        fs::remove_file(&main).unwrap();
        let (stream, deleted_progress) = RepositoryChunkStream::open_incremental(
            directory.path(),
            &manifest,
            Some(changed_progress.checkpoint().unwrap()),
        )
        .unwrap();
        assert!(stream.collect::<Result<Vec<_>, _>>().unwrap().is_empty());
        assert_eq!(deleted_progress.deletions().unwrap().len(), 1);
        assert_eq!(deleted_progress.snapshot().chunks_deleted, 1);
    }
}
