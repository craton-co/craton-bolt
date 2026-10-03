#!/usr/bin/env bash
set -euo pipefail

# Change to the script's directory so it runs correctly regardless of where it's called
cd "$(dirname "$0")"

echo "========================================"
echo " Preparing Docker Environments..."
echo "========================================"

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

# --- Dockerfile 1: Tests Container ---
cat << 'EOF' > "$TMP_DIR/Dockerfile.tests"
FROM rust:latest
WORKDIR /workspace
EOF

# --- Dockerfile 2: All Other Jobs Container ---
cat << 'EOF' > "$TMP_DIR/Dockerfile.others"
FROM rust:latest
WORKDIR /workspace

# Install protoc (required by substrait feature-build)
RUN apt-get update && apt-get install -y protobuf-compiler && rm -rf /var/lib/apt/lists/*

# Add Rust components
RUN rustup component add rustfmt clippy llvm-tools-preview
RUN rustup toolchain install nightly-2026-04-03 --profile minimal
# MSRV leg — must match `rust-version` in Cargo.toml and the hosted matrix.
RUN rustup toolchain install 1.85.0 --profile minimal

# Build the audited tools through Cargo rather than executing a network-fetched
# shell installer in the CI image.
RUN cargo install --locked cargo-llvm-cov --version 0.6.16 \
 && cargo install --locked cargo-deny --version 0.20.2 \
 && cargo install --locked cargo-public-api --version 0.52.0
EOF

# Build images in parallel
echo "Building Docker images (in parallel)..."
docker build -t local-ci-tests -f "$TMP_DIR/Dockerfile.tests" "$TMP_DIR" &
BUILD_TESTS_PID=$!
docker build -t local-ci-others -f "$TMP_DIR/Dockerfile.others" "$TMP_DIR" &
BUILD_OTHERS_PID=$!

BUILD_TESTS_EXIT=0; BUILD_OTHERS_EXIT=0
wait $BUILD_TESTS_PID || BUILD_TESTS_EXIT=$?
wait $BUILD_OTHERS_PID || BUILD_OTHERS_EXIT=$?

if [[ $BUILD_TESTS_EXIT -ne 0 || $BUILD_OTHERS_EXIT -ne 0 ]]; then
    echo "ERROR: Docker image build failed (tests=$BUILD_TESTS_EXIT, others=$BUILD_OTHERS_EXIT)."
    exit 1
fi

# --- Fix for Windows Git Bash path translation ---
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    # Use native Windows path (e.g., C:/craton/bolt)
    HOST_DIR="$(pwd -W)"
    HOST_TMP_DIR="$(cd "$TMP_DIR" && pwd -W)"
    # Prevent Git Bash from changing /workspace to C:\Program Files\Git\workspace
    export MSYS_NO_PATHCONV=1
else
    HOST_DIR="$PWD"
    HOST_TMP_DIR="$TMP_DIR"
fi

# Common volume mounts (shared source + cargo registry).
# Each container gets its OWN target volume to avoid parallel build conflicts.
CACHE_COMMON=(
    -v "$HOST_DIR:/workspace"
    -v "local-ci-cargo-registry:/usr/local/cargo/registry"
)

echo ""
echo "========================================"
echo " Running both containers in parallel..."
echo "========================================"

# Container 1: Tests — run in background subshell.
# set +e inside so the subshell never dies before writing the exit-code file.
# PIPESTATUS[0] captures docker's exit code after the sed pipe.
# --cpus 2 caps each container so two parallel builds don't OOM Docker Desktop.
(
    set +e
    docker run --rm --cpus 2 \
        "${CACHE_COMMON[@]}" \
        -v "local-ci-target-tests:/target" \
        -v "$HOST_TMP_DIR:/ci_tmp" \
        -e CARGO_TARGET_DIR=/target \
        local-ci-tests bash -c "
  set -ex
  _step() { echo \"\$1\" > /ci_tmp/tests_failed_step; echo \">>> Running \$1\"; }

  _step 'cargo test (lib + integration)'
  cargo test --lib --tests --features cuda-stub --no-default-features

  _step 'cargo test (doctests)'
  cargo test --doc --features cuda-stub --no-default-features

  rm -f /ci_tmp/tests_failed_step
" 2>&1 | sed 's/^/[TESTS] /'
    echo "${PIPESTATUS[0]}" > "$TMP_DIR/tests.exit"
) &
TESTS_PID=$!

# Container 2: All other jobs — run in background subshell.
(
    set +e
    docker run --rm --cpus 2 \
        "${CACHE_COMMON[@]}" \
        -v "local-ci-target-others:/target" \
        -v "$HOST_TMP_DIR:/ci_tmp" \
        -e CARGO_TARGET_DIR=/target \
        local-ci-others bash -c "
  set -ex
  _step() { echo \"\$1\" > /ci_tmp/others_failed_step; echo \">>> Running \$1\"; }

  _step 'rustfmt check'
  cargo fmt --all -- --check

  _step 'clippy (blocking)'
  cargo clippy --lib --tests --features cuda-stub --no-default-features -- -D warnings

  _step 'cargo check (lib, strict)'
  RUSTFLAGS='-D warnings' cargo check --lib --features cuda-stub --no-default-features

  _step 'cargo check --features cudarc'
  cargo check --lib --features cudarc --no-default-features

  _step 'MSRV gate (1.85, mirrors the hosted matrix leg)'
  cargo +1.85.0 check --lib --features cuda-stub --no-default-features
  cargo +1.85.0 test --lib --tests --features cuda-stub --no-default-features

  _step 'feature tests (flight + substrait)'
  cargo test --lib --tests --no-default-features --features cuda-stub,flight
  cargo test --lib --tests --no-default-features --features cuda-stub,substrait

  _step 'public API snapshot gate'
  bash scripts/check_public_api.sh

  _step 'cargo doc'
  cargo doc --no-default-features --features cuda-stub --no-deps

  _step 'package (cargo publish --dry-run)'
  cargo publish --dry-run --allow-dirty --no-default-features --features cuda-stub

  _step 'coverage (host, >=50% lines)'
  cargo llvm-cov --no-default-features --features cuda-stub --lib --tests --ignore-filename-regex 'src/cuda/' --lcov --output-path lcov.info --fail-under-lines 50
  cargo llvm-cov --no-default-features --features cuda-stub --lib --tests --ignore-filename-regex 'src/cuda/' --summary-only --fail-under-lines 50

  _step 'cargo deny (licenses + bans)'
  cargo deny check licenses bans

  _step 'cargo deny (advisories, blocking)'
  cargo deny check advisories

  _step 'cargo deny (all-features, blocking)'
  cargo deny --all-features check advisories licenses bans

  rm -f /ci_tmp/others_failed_step
" 2>&1 | sed 's/^/[OTHERS] /'
    echo "${PIPESTATUS[0]}" > "$TMP_DIR/others.exit"
) &
OTHERS_PID=$!

echo "(output is prefixed: [TESTS] and [OTHERS] — lines may interleave)"
echo ""

wait $TESTS_PID
wait $OTHERS_PID

TESTS_EXIT=$(cat "$TMP_DIR/tests.exit" 2>/dev/null || echo 1)
OTHERS_EXIT=$(cat "$TMP_DIR/others.exit" 2>/dev/null || echo 1)

echo ""
echo "========================================"
echo " Results"
echo "========================================"
if [[ $TESTS_EXIT -eq 0 ]]; then
    echo " CONTAINER 1 (TESTS):  PASSED"
else
    TESTS_FAILED_STEP=$(cat "$TMP_DIR/tests_failed_step" 2>/dev/null || echo "unknown step")
    echo " CONTAINER 1 (TESTS):  FAILED (exit $TESTS_EXIT) — step: $TESTS_FAILED_STEP"
fi
if [[ $OTHERS_EXIT -eq 0 ]]; then
    echo " CONTAINER 2 (OTHERS): PASSED"
else
    OTHERS_FAILED_STEP=$(cat "$TMP_DIR/others_failed_step" 2>/dev/null || echo "unknown step")
    echo " CONTAINER 2 (OTHERS): FAILED (exit $OTHERS_EXIT) — step: $OTHERS_FAILED_STEP"
fi

if [[ $TESTS_EXIT -ne 0 || $OTHERS_EXIT -ne 0 ]]; then
    echo ""
    echo " LOCAL CI FAILED."
    exit 1
fi

if [[ "${BOLT_LOCAL_GPU:-0}" == "1" ]]; then
    echo ">>> Running blocking native GPU lane"
    # `--lib --tests` scopes this to the test binaries. A bare `cargo test ...
    # -- --ignored` also runs the doctest phase with `--ignored`, compiling the
    # ```ignore``` examples that are illustrative by construction and cannot
    # compile — that phase would keep the lane permanently red. Doctests run in
    # the tests container above.
    BOLT_BENCH_GPU=1 cargo test --lib --tests --no-default-features --features cudarc -- --ignored --test-threads=1
    BOLT_BENCH_GPU=1 cargo test --no-default-features --features cudarc,reference-tests \
        --test diff_duckdb --test diff_duckdb_semantics --test sql_proptest \
        -- --ignored --test-threads=1
    # The optional-subsystem e2e fixtures need a CUDA context AND their own
    # feature, so neither the hosted feature lane nor the commands above reach
    # their #[ignore]-gated tier.
    BOLT_BENCH_GPU=1 cargo test --no-default-features --features cudarc,flight \
        --test flight_e2e -- --ignored --test-threads=1
    BOLT_BENCH_GPU=1 cargo test --no-default-features --features cudarc,substrait \
        --test substrait_e2e -- --ignored --test-threads=1
else
    echo "GPU lane skipped. Set BOLT_LOCAL_GPU=1 on a CUDA host for full CI parity."
fi

echo ""
echo "========================================"
echo " LOCAL CI COMPLETED SUCCESSFULLY!"
echo "========================================"
