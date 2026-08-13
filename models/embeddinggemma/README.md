# Offline EmbeddingGemma ONNX bundle

Hay never downloads model files at runtime. Prepare this directory once while
online, then query and index with networking disabled.

1. Accept Google's Gemma license and acquire the exact
   `google/embeddinggemma-300m` revision through an approved channel.
2. Export a float32 ONNX graph which includes mean pooling, both projection
   layers, and final normalization. The configured output must be the final
   `[batch, 768]` `sentence_embedding`, not `[batch, tokens, hidden]` states.
3. Copy the matching Hugging Face `tokenizer.json` beside the graph. If the
   ONNX graph uses external data files, add every file to `artifacts`.
4. Copy `bundle.example.json` to `bundle.json`, pin the upstream and export
   revisions, and replace each checksum with `shasum -a 256 <file>` output.
5. Set `COTH_HAY_SEEKER_LOCAL_MODEL_DIR` to this directory.

The loader rejects unknown manifest keys, missing checksums, uppercase or
malformed digests, path traversal, symlink escapes, incompatible tensor names,
raw/unpooled output shapes, non-768 base output, and prompt drift. On macOS it
attempts the ONNX Runtime Core ML execution provider first and falls back to
CPU if model compilation fails. Artifact verification and inference perform no
network requests.

DuckDB stores the first 256 MRL dimensions after re-normalization.
Elasticsearch uses the same bundle at 768 dimensions; BBQ remains its only
representation-specific difference.

Verify the complete local contract using only built-in synthetic strings:

```bash
COTH_HAY_SEEKER_LOCAL_MODEL_DIR=/absolute/path/to/bundle \
  cargo run -p cast-embeddings --example local_onnx_smoke
```

The JSON output reports artifact/session startup time, selected Core ML or CPU
provider, any Core ML fallback reason, first and warm query latency, batch
throughput, dimensions, and query-vector norm. No repository content or
network request is involved.

Measure warm encode throughput with Criterion:

```bash
COTH_HAY_SEEKER_LOCAL_MODEL_DIR=/absolute/path/to/bundle \
  cargo bench -p cast-embeddings --bench local_encode -- --noplot
```
