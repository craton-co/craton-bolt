#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

expected="$(mktemp)"
actual="$(mktemp)"
trap 'rm -f "$expected" "$actual"' EXIT

# Normalize Windows checkouts and the optional UTF-8 BOM before comparing.
sed -e '1s/^\xEF\xBB\xBF//' -e 's/\r$//' \
  docs/PUBLIC_API_SNAPSHOT.txt > "$expected"

cargo +nightly-2026-04-03 public-api -sss --color never \
  --no-default-features --features cuda-stub > "$actual"

if ! diff -u "$expected" "$actual"; then
  echo >&2
  echo "Public API drift detected." >&2
  echo "Regenerate docs/PUBLIC_API_SNAPSHOT.txt with cargo-public-api 0.52.0," >&2
  echo "review the diff, and update docs/API_SURFACE.md in the same change." >&2
  exit 1
fi
