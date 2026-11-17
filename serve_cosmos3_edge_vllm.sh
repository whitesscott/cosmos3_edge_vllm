#!/bin/bash
# Serve the Cosmos3-Edge reasoner with vLLM (OpenAI API), optionally with a Gradio frontend.
#
# Usage:
#   ./serve_cosmos3_edge_vllm.sh [OPTIONS]
#
# Options:
#   --port PORT          vLLM port (default 8000)
#   --gpu-mem FRAC       --gpu-memory-utilization (default 0.30)
#   --max-len N          --max-model-len (default 16384)
#   --gradio             Also start cosmos3_vllm_app.py (ask anything + PPE check) when vLLM is up
#   --lab                Also start ppe_lab.py (PPE wordings, label review, sweep reports)
#   --gradio-port PORT   Gradio port (default 7860 for --gradio, 7861 for --lab)
#   --gradio-ip IP       Gradio bind address (default 127.0.0.1)
#   --auth USER:PASS     Gradio login (use when binding beyond localhost or with --tunnel)
#   --tunnel             Expose Gradio via a remotely-managed cloudflared tunnel (implies
#                        --gradio unless --lab is given; public hostname -> http://localhost:PORT
#                        is set in the Cloudflare Zero Trust dashboard)
#   --tunnel-url URL     Public URL to print (optional; auto-detected from the ingress
#                        config the dashboard pushes to cloudflared)
#   --help
#
# Environment (tunnel):
#   TUNNEL_TOKEN            cloudflared tunnel token (overrides the token file)
#   CLOUDFLARED_TOKEN_FILE  Token file (default: ~/.config/cloudflared/gradio.token)
#                           Create it once with:
#                             install -m 600 /dev/null ~/.config/cloudflared/gradio.token
#                             echo 'eyJhIjoi...' > ~/.config/cloudflared/gradio.token
#   CLOUDFLARED_URL         Same as --tunnel-url
#
# Settings follow the forum write-up (Cosmos 3 Edge on Thor):
#   * structured outputs need an explicit backend and no free whitespace, otherwise
#     JSON answers run away to the token cap:
#       --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'
#   * thinking is per request (chat_template_kwargs.enable_thinking); no reasoning
#     parser, so clients see and strip </think> themselves.
#   * min/max pixels per the model card (images go in at ~native resolution).
#   * ViT attention on TORCH_SDPA: the bundled vllm_fa2 ViT kernel is PTX from nvcc 13.4
#     and the Thor driver (CUDA 13.2) rejects it (cudaErrorUnsupportedPtxVersion). Same for
#     the LM's FLASH_ATTN backend, so the LM uses FlashInfer (AOT sm110a cubins).
#   * FlashInfer AOT needs the flashinfer-jit-cache shim AND flashinfer-jit-cache-sm110a at
#     the same version; a missing/stale shim fails engine start with
#     "module 'flashinfer_jit_cache' has no attribute '__version__'". The pre-flight below
#     installs the shim matching the installed sm110a provider.
# If startup dies in memory profiling with "Initial free memory ... current free memory"
# drift, run: sync && echo 3 | sudo tee /proc/sys/vm/drop_caches   and restart.

set -e
VENV="$HOME/.git/uv/cosmos-cu134"
PY="$VENV/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HUB="$HOME/.cache/huggingface/hub/models--nvidia--Cosmos3-Edge"
# refs/main can point at a metadata-only snapshot (6afc125 here: no weights pulled, but its
# weights/template/processor files are byte-identical to a9d944 on the Hub), so prefer it
# only if its first transformer shard exists, else take any snapshot that has weights.
SHARD="transformer/diffusion_pytorch_model-00001-of-00002.safetensors"
MODEL="$HUB/snapshots/$(cat "$HUB/refs/main")"
if [ ! -e "$MODEL/$SHARD" ]; then
    MODEL=$(dirname "$(dirname "$(ls -d "$HUB"/snapshots/*/"$SHARD" | head -1)")")
fi
PORT=8000
GPU_MEM=0.30
MAX_LEN=16384
FRONTEND=""
GRADIO_PORT=""
GRADIO_IP="127.0.0.1"
AUTH=""
USE_TUNNEL=false
TUNNEL_URL="${CLOUDFLARED_URL:-}"
CLOUDFLARED_TOKEN_FILE="${CLOUDFLARED_TOKEN_FILE:-$HOME/.config/cloudflared/gradio.token}"
CLOUDFLARED_LOG="/tmp/cloudflared_gradio.log"

while [[ $# -gt 0 ]]; do
    case $1 in
        --port)        PORT="$2";        shift 2 ;;
        --gpu-mem)     GPU_MEM="$2";     shift 2 ;;
        --max-len)     MAX_LEN="$2";     shift 2 ;;
        --gradio)      FRONTEND="app";   shift ;;
        --lab)         FRONTEND="lab";   shift ;;
        --gradio-port) GRADIO_PORT="$2"; shift 2 ;;
        --gradio-ip)   GRADIO_IP="$2";   shift 2 ;;
        --auth)        AUTH="$2";        shift 2 ;;
        --tunnel)      USE_TUNNEL=true;  shift ;;
        --tunnel-url)  TUNNEL_URL="$2";  shift 2 ;;
        --help) sed -n '2,/^$/{ s/^# \?//; p }' "$0"; exit 0 ;;
        *) echo "Unknown option: $1  (use --help)"; exit 1 ;;
    esac
done

# Fail fast (before loading the model) if the tunnel was requested without a token
if [ "$USE_TUNNEL" = true ]; then
    [ -z "$FRONTEND" ] && FRONTEND="app"   # the tunnel exposes Gradio, so one must run
    command -v cloudflared >/dev/null 2>&1 || {
        echo "ERROR: --tunnel given but cloudflared is not installed" >&2; exit 1; }
    if [ -z "${TUNNEL_TOKEN:-}" ] && [ ! -s "$CLOUDFLARED_TOKEN_FILE" ]; then
        echo "ERROR: --tunnel needs \$TUNNEL_TOKEN or a token in $CLOUDFLARED_TOKEN_FILE" >&2
        exit 1
    fi
    if [ -z "$AUTH" ]; then
        echo "WARNING: public tunnel without --auth — anyone with the URL can use this GPU."
        [ "$FRONTEND" = "lab" ] && echo "         (ppe_lab.py has no login; its Review tab can edit labels.json)"
    fi
fi

# ── Pre-flight: FlashInfer AOT shim must match the sm110a provider ───────────
fi_version() {  # installed version of a distribution, or empty
    "$PY" -c "import importlib.metadata as m, sys
try: print(m.version(sys.argv[1]))
except m.PackageNotFoundError: pass" "$1"
}
PROVIDER=$(fi_version flashinfer-jit-cache-sm110a)
SHIM=$(fi_version flashinfer-jit-cache)
if [ -n "$PROVIDER" ] && [ "$SHIM" != "$PROVIDER" ]; then
    BASE="${PROVIDER%%+*}"   # 0.7.4+cu134 -> 0.7.4
    WHEEL="https://github.com/flashinfer-ai/flashinfer/releases/download/v${BASE}/flashinfer_jit_cache-${PROVIDER}-cp39-abi3-manylinux_2_28_aarch64.whl"
    echo "FlashInfer: shim '${SHIM:-missing}' != provider '$PROVIDER'; installing matching shim"
    echo "  $WHEEL"
    uv pip install --python "$PY" --no-deps "flashinfer-jit-cache @ $WHEEL"
fi

export HF_HUB_OFFLINE=1
echo "Model : $MODEL"
echo "vLLM  : http://127.0.0.1:$PORT/v1  ($("$PY" -c 'import vllm; print(vllm.__version__)'))"

VLLM_CMD=("$VENV/bin/vllm" serve "$MODEL"
    --served-model-name nvidia/Cosmos3-Edge
    --host 127.0.0.1 --port "$PORT"
    --gpu-memory-utilization "$GPU_MEM"
    --max-model-len "$MAX_LEN"
    --limit-mm-per-prompt '{"image": 1, "video": 0}'
    --mm-encoder-attn-backend TORCH_SDPA
    --attention-backend FLASHINFER
    --mm-processor-kwargs '{"do_resize": true, "min_pixels": 4096, "max_pixels": 16777216}'
    --structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}')

if [ -z "$FRONTEND" ]; then
    exec "${VLLM_CMD[@]}"
fi

# ── vLLM + Gradio frontend (+ cloudflared) ───────────────────────────────────
VLLM_LOG="/tmp/cosmos3_edge_vllm_${PORT}.log"
VLLM_PID=""
GRADIO_PID=""
TUNNEL_PID=""
if [ "$FRONTEND" = "lab" ]; then GRADIO_PORT="${GRADIO_PORT:-7861}"; else GRADIO_PORT="${GRADIO_PORT:-7860}"; fi
cleanup() {
    trap '' INT TERM              # a second Ctrl+C must not interrupt shutdown
    echo ""
    echo "Shutting down..."
    for pid in $TUNNEL_PID $GRADIO_PID $VLLM_PID; do
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    exit 0
}
trap cleanup SIGINT SIGTERM

"${VLLM_CMD[@]}" > >(tee "$VLLM_LOG") 2>&1 &
VLLM_PID=$!
echo "Waiting for vLLM (log: $VLLM_LOG)..."
for _ in $(seq 1 300); do
    curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models" && break
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "vLLM exited — check $VLLM_LOG"; exit 1; }
    sleep 2
done
curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models" || { echo "vLLM not ready after 600s"; cleanup; }

if [ "$FRONTEND" = "lab" ]; then
    APP=(ppe_lab.py)
else
    APP=(cosmos3_vllm_app.py)
fi
APP+=(--port "$GRADIO_PORT" --endpoint "http://127.0.0.1:$PORT/v1" --host "$GRADIO_IP")
[ -n "$AUTH" ] && APP+=(--auth "$AUTH")
cd "$SCRIPT_DIR"
"$PY" "${APP[@]}" &
GRADIO_PID=$!
for _ in $(seq 1 60); do
    curl -s -o /dev/null "http://127.0.0.1:$GRADIO_PORT/" && break
    kill -0 "$GRADIO_PID" 2>/dev/null || { echo "Gradio exited"; cleanup; }
    sleep 1
done

# ── cloudflared ──────────────────────────────────────────────────────────────
if [ "$USE_TUNNEL" = true ]; then
    echo ""
    echo "Starting cloudflared tunnel..."

    # Kill any leftover tunnel from a previous run of this script
    pkill -f "cloudflared tunnel --no-autoupdate --logfile $CLOUDFLARED_LOG" 2>/dev/null || true
    sleep 1

    # Token goes via env var or --token-file, never argv, so it stays out of `ps`.
    TUNNEL_CMD=(cloudflared tunnel --no-autoupdate --logfile "$CLOUDFLARED_LOG" run)
    if [ -n "${TUNNEL_TOKEN:-}" ]; then
        export TUNNEL_TOKEN
    else
        TUNNEL_CMD+=(--token-file "$CLOUDFLARED_TOKEN_FILE")
    fi

    rm -f "$CLOUDFLARED_LOG"
    "${TUNNEL_CMD[@]}" > /dev/null 2>&1 &
    TUNNEL_PID=$!

    TUNNEL_OK=false
    for _ in $(seq 1 20); do
        sleep 1
        if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
            echo "Warning: cloudflared exited early — log:"
            cat "$CLOUDFLARED_LOG" 2>/dev/null || true
            TUNNEL_PID=""
            break
        fi
        if grep -q "Registered tunnel connection" "$CLOUDFLARED_LOG" 2>/dev/null; then
            TUNNEL_OK=true
            break
        fi
    done

    if [ "$TUNNEL_OK" = true ]; then
        # The ingress config is pushed from the dashboard shortly after the first
        # connection registers: {"hostname":"chat.example.com","service":"http://127.0.0.1:7860"}
        TUNNEL_SERVICE=""
        for _ in $(seq 1 10); do
            if grep -q "Updated to new configuration" "$CLOUDFLARED_LOG" 2>/dev/null; then
                cfg=$(grep "Updated to new configuration" "$CLOUDFLARED_LOG" | tail -1)
                host=$(grep -oP 'hostname\\*"\s*:\s*\\*"\K[^"\\]+' <<< "$cfg" | head -1 || true)
                TUNNEL_SERVICE=$(grep -oP 'service\\*"\s*:\s*\\*"\Khttps?://[^"\\]+' <<< "$cfg" | head -1 || true)
                [ -z "$TUNNEL_URL" ] && [ -n "$host" ] && TUNNEL_URL="https://$host"
                break
            fi
            sleep 1
        done
        if [ -n "$TUNNEL_SERVICE" ] && [[ "$TUNNEL_SERVICE" != *":$GRADIO_PORT" ]]; then
            echo "Warning: dashboard routes the tunnel to $TUNNEL_SERVICE, but Gradio is on port $GRADIO_PORT"
        fi
        if [ -n "$TUNNEL_URL" ]; then
            echo "Tunnel public URL: $TUNNEL_URL"
        else
            echo "cloudflared tunnel connected (public hostname is set in the Cloudflare dashboard)"
        fi
    else
        echo "Warning: cloudflared tunnel did not connect within 20s"
        echo "  Log: $CLOUDFLARED_LOG"
        grep -v "^$" "$CLOUDFLARED_LOG" 2>/dev/null | tail -5 || true
    fi
fi

echo ""
echo "=========================================="
echo " Running — Ctrl+C stops everything"
echo "=========================================="
echo " vLLM   : http://127.0.0.1:$PORT/v1   (log: $VLLM_LOG)"
echo " Gradio : http://$GRADIO_IP:$GRADIO_PORT"
[ "$USE_TUNNEL" = true ] && [ -n "$TUNNEL_URL" ] && echo " Public : $TUNNEL_URL"
[ "$USE_TUNNEL" = true ] && echo " Tunnel : $CLOUDFLARED_LOG"
echo "=========================================="
wait "$GRADIO_PID" || true
cleanup
