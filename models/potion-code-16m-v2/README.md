# Potion Code 16M v2 local bundle

Hay's `local-static` provider pins `minishlab/potion-code-16M-v2` at revision
`e9d2a44ca6a05ac6685f3b23709ea57eb7352d5b`. The model is code-trained,
MIT-licensed, 256-dimensional, and runs from a static token-embedding table.
Its license lineage is permissive throughout: the checkpoint is MIT, distilled
from the MIT-licensed `nomic-ai/CodeRankEmbed`, itself built on Apache-2.0
`Snowflake/snowflake-arctic-embed-m-long`. Redistribution and automatic
provisioning are permitted with attribution.

Loading is offline and checksum-verified. By default Hay provisions this bundle
into a per-user cache on first use; see
[Automatic model provisioning](../../README.md#automatic-model-provisioning).
The rest of this document describes staging the bundle by hand, which
`HAY_LOCAL_STATIC_MODEL_DIR` selects and which disables provisioning entirely.

Create a private bundle directory containing the three unmodified upstream
artifacts plus the checked-in manifest template:

```text
static-bundle.json
model.safetensors
tokenizer.json
config.json
```

Copy `static-bundle.example.json` to `static-bundle.json`. The runtime verifies
the exact revision metadata, lowercase SHA-256 of every artifact, tokenizer
unknown-token ID, 16,384-token ceiling, F16 tensor shape, and inference profile
before opening the index. It rejects symlinks that escape the bundle.

```bash
export HAY_LOCAL_STATIC_MODEL_DIR=/absolute/path/to/potion-code-16m-v2
cargo run -p hay-cli -- index --backend duckdb --embeddings local-static \
  --database .hay-seeker/code.duckdb --repository /path/to/repository
cargo run -p hay-cli -- search --backend duckdb --embeddings local-static \
  --database .hay-seeker/code.duckdb "where is request validation handled?"
```

Inference uses no special tokens, drops `[UNK]`, caps known tokens at 16,384,
mean-pools the table rows, and L2-normalizes the result. That exact string is
part of the persisted index identity. DuckDB stores per-vector int8 values;
Elasticsearch stores BBQ vectors. Both use the model's native trained 256d
width, so this is an explicit alternative to the original 256d/768d ONNX
representation contract rather than a silent truncation.
