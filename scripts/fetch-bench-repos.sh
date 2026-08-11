#!/bin/sh
set -eu

repo_root=${1:-.bench-repos}
mkdir -p "$repo_root"

fetch_repo() {
    repo_name=$1
    repo_url=$2
    repo_revision=$3
    repo_dir="$repo_root/$repo_name"
    new_checkout=false

    if [ ! -d "$repo_dir/.git" ]; then
        git clone --depth 1 --no-checkout "$repo_url" "$repo_dir"
        new_checkout=true
    fi

    if [ "$new_checkout" = true ]; then
        git -C "$repo_dir" fetch --depth 1 origin "$repo_revision"
        git -C "$repo_dir" checkout --detach "$repo_revision"
        echo "$repo_name $repo_revision"
        return
    fi

    if ! git -C "$repo_dir" diff --quiet || ! git -C "$repo_dir" diff --cached --quiet; then
        echo "refusing to change dirty benchmark checkout: $repo_dir" >&2
        exit 1
    fi

    current_revision=$(git -C "$repo_dir" rev-parse HEAD 2>/dev/null || true)
    if [ "$current_revision" != "$repo_revision" ]; then
        git -C "$repo_dir" fetch --depth 1 origin "$repo_revision"
        git -C "$repo_dir" checkout --detach "$repo_revision"
    fi

    echo "$repo_name $repo_revision"
}

fetch_repo wordpress https://github.com/WordPress/wordpress-develop.git 7b887ba4820e0ee87bbf3f14a0e8385b33f1a6fd
fetch_repo django https://github.com/django/django.git dfc52e53f1d19a2730854d68b602fb4dba8bf0c5
fetch_repo kubernetes https://github.com/kubernetes/kubernetes.git 4f5591ab57b75c0b8cabbff3031c9b956075c1ed
fetch_repo ollama https://github.com/ollama/ollama.git 144893850fa778c8c81ff931f26614d62e6689c1
