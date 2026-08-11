#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
asset_dir="$repo_root/.bench-assets"
asset_path="$asset_dir/o200k_base.tiktoken"
expected_sha="446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
nightly="nightly-2026-08-06"
gigatoken_revision="fac0114b37120ec8a76362e9ee8e1c742aaafaef"
gigatoken_dir="$repo_root/.bench-tools/gigatoken"
gigatoken_patch="$repo_root/benchmarks/gigatoken-rust/patches/profile-rustflags.patch"

mkdir -p "$asset_dir"
if [[ ! -f "$asset_path" ]]; then
    curl --fail --location --output "$asset_path" \
        "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
fi

actual_sha="$(shasum -a 256 "$asset_path" | awk '{print $1}')"
if [[ "$actual_sha" != "$expected_sha" ]]; then
    echo "o200k_base checksum mismatch: expected $expected_sha, got $actual_sha" >&2
    exit 1
fi

if [[ ! -d "$repo_root/.bench-repos/wordpress" ]]; then
    "$repo_root/scripts/fetch-bench-repos.sh"
fi

if ! rustup toolchain list | grep -q "^$nightly"; then
    rustup toolchain install "$nightly" --profile minimal
fi

if [[ "${1:-}" != "--giga-only" ]]; then
    cd "$repo_root"
    cargo bench -p cast-tokenizers --bench sizing -- --sample-size 10
fi

mkdir -p "$repo_root/.bench-tools"
if [[ ! -d "$gigatoken_dir/.git" ]]; then
    git clone --no-checkout https://github.com/marcelroed/gigatoken.git "$gigatoken_dir"
    git -C "$gigatoken_dir" checkout --detach "$gigatoken_revision"
fi

actual_revision="$(git -C "$gigatoken_dir" rev-parse HEAD)"
if [[ "$actual_revision" != "$gigatoken_revision" ]]; then
    echo "GigaToken revision mismatch: expected $gigatoken_revision, got $actual_revision" >&2
    exit 1
fi

if ! grep -q '^cargo-features = \["profile-rustflags"\]' "$gigatoken_dir/Cargo.toml"; then
    git -C "$gigatoken_dir" apply "$gigatoken_patch"
fi

cd "$repo_root/benchmarks/gigatoken-rust"
cargo "+$nightly" bench --bench compare -- --sample-size 10
