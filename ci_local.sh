#!/usr/bin/env bash
set -e

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

# Install cargo-binstall (A tool to download pre-compiled cargo binaries directly)
RUN curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash

# Fast-install cargo-llvm-cov and cargo-deny using pre-compiled binaries
RUN cargo binstall -y cargo-llvm-cov cargo-deny
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

  _step 'clippy (advisory)'
  cargo clippy --lib --tests --features cuda-stub --no-default-features || echo '⚠️ Clippy failed but is non-blocking'

  _step 'cargo check (lib, strict)'
  RUSTFLAGS='-D warnings' cargo check --lib --features cuda-stub --no-default-features

  _step 'cargo check --features cudarc'
  cargo check --lib --features cudarc --no-default-features

  _step 'feature build (cuda-stub,flight)'
  cargo check --lib --tests --no-default-features --features cuda-stub,flight

  _step 'feature build (cuda-stub,substrait)'
  cargo check --lib --tests --no-default-features --features cuda-stub,substrait

  _step 'cargo doc'
  cargo doc --no-default-features --features cuda-stub --no-deps

  _step 'cargo publish --dry-run'
  cargo publish --dry-run --allow-dirty --no-default-features --features cuda-stub

  _step 'cargo llvm-cov'
  cargo llvm-cov --no-default-features --features cuda-stub --lib --lcov --output-path lcov.info || echo '⚠️ Coverage failed but is non-blocking'
  cargo llvm-cov --no-default-features --features cuda-stub --lib --summary-only || true

  _step 'cargo deny (licenses + bans)'
  cargo deny check licenses bans

  _step 'cargo deny (advisories)'
  cargo deny check advisories || echo '⚠️ cargo-deny advisories failed but is non-blocking'

  _step 'cargo deny (all-features)'
  cargo deny --all-features check advisories licenses bans || echo '⚠️ cargo-deny all-features failed but is non-blocking'

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

echo ""
echo "========================================"
echo " LOCAL CI COMPLETED SUCCESSFULLY!"
echo "========================================"
echo "(Note: gpu-integration tests were skipped as they require a self-hosted physical NVIDIA GPU)"
