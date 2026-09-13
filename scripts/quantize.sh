#!/usr/bin/env bash
# Convert merged weights and build three GGUF variants sequentially.
set -euo pipefail
cd "$(dirname "$0")/.."
STAGING="${1:-out/merged}"
OUTPUT="${2:-out/gguf}"
NAME="${3:-xobsessed-1.7b}"
LLAMA_DIR="${LLAMA_CPP_DIR:-llama.cpp}"
BUILD_JOBS="${BUILD_JOBS:-1}"
[[ "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]]
test -f "$STAGING/config.json"
if [ ! -d "$LLAMA_DIR/.git" ]; then
    echo "Clone a reviewed llama.cpp revision before conversion." >&2
    exit 1
fi
mkdir -p "$OUTPUT"
git -C "$LLAMA_DIR" rev-parse HEAD > "$OUTPUT/llama-cpp-revision.txt"
if [ ! -x "$LLAMA_DIR/build/bin/llama-quantize" ]; then
    cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_CUDA=OFF
    cmake --build "$LLAMA_DIR/build" --target llama-quantize -j "$BUILD_JOBS"
fi
uv run --with gguf --with sentencepiece --with protobuf python "$LLAMA_DIR/convert_hf_to_gguf.py" \
    "$STAGING" --outfile "$OUTPUT/$NAME-BF16.gguf" --outtype bf16
for QUANT in Q8_0 Q6_K Q4_K_M; do
    "$LLAMA_DIR/build/bin/llama-quantize" \
        "$OUTPUT/$NAME-BF16.gguf" "$OUTPUT/$NAME-$QUANT.gguf" "$QUANT" "$BUILD_JOBS"
done
echo "Review the GGUF files before a separate upload."
