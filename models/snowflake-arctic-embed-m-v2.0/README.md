# Offline Snowflake Arctic Embed ONNX bundle

This Apache-2.0, 305M-parameter profile is the preferred candidate for Hay's
256d DuckDB / 768d Elasticsearch acceptance gate. Snowflake trains the model
with Matryoshka Representation Learning at 256 dimensions and reports a 768d
base representation. The checkpoint is ungated and permitted for commercial
use.

The manifest pins official revision
`95c2741480856aa9666782eb4afe11959938017f`, the published int8 ONNX graph,
the matching tokenizer, and both SHA-256 digests. Download `onnx/model_int8.onnx`
and `tokenizer.json` from that exact revision, copy `bundle.example.json` to
`bundle.json`, and point `HAY_LOCAL_MODEL_DIR` at the resulting directory.

The graph already exposes its final 768-dimensional `sentence_embedding`.
Hay applies Snowflake's documented `query: ` prefix only to queries. Queries
are capped at 256 model tokens. Documents use complete 256-token windows with
32 tokens of source overlap; every window is L2-normalized, then combined at
768 dimensions with a non-padding-token-weighted mean. Only after that shared
aggregation does Hay select the leading 256 or 768 MRL dimensions and
L2-normalize the result. This preserves the proven 1,500-token CAST boundaries
without silently discarding a long chunk's tail, and both storage backends use
the identical aggregation contract.

On the 194-chunk Hay repository smoke corpus, the windowed profile reduced a
clean DuckDB dense build from 47.2 seconds with the legacy 2,048-token
first-window profile to 33.2 seconds. A 512-token window experiment took 42.0
seconds. The 256-token profile retained the exact seed parity result: 0.009678
nDCG@10 delta and zero recall delta between DuckDB and Elasticsearch.

The dynamic int8 graph is CPU-selected because the equivalent Nomic dynamic
graph demonstrated severe Core ML performance regressions on long code
batches; a future static Core ML export must be separately profiled and pinned
before changing this flag.
