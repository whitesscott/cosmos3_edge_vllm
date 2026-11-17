#!/bin/bash
# Launch the Cosmos3 Gradio front end (cosmos-framework, no Ray) and optionally an ngrok tunnel
#
# Usage:
#   ./run_cosmos_gradio.sh [OPTIONS]
#
# Options:
#   --gradio-port PORT    Port for Gradio (default: 7860)
#   --gradio-ip IP        Bind IP for Gradio (default: 127.0.0.1)
#   --checkpoint PATH     Checkpoint dir or HF name (default: local Cosmos3-Edge snapshot)
#   --output-dir DIR      Where generations are saved (default: ~/cosmos-out/gradio)
#   --auth USER:PASS      Require a Gradio login (strongly recommended with --ngrok / --share)
#   --guardrails          Enable cosmos guardrails (downloads extra models; needs --online first time)
#   --online              Allow Hugging Face downloads (default: offline, cache only)
#   --share               Share Gradio via its built-in public tunnel
#   --ngrok               Expose Gradio via the ngrok named tunnel (default: 'chat')
#   --ngrok-tunnel NAME   ngrok tunnel name from ~/.config/ngrok/ngrok.yml (default: chat,
#                         https://chat.whitesscott.com.ngrok.dev -> 7860)
#   --help                Show this help

set -e

# ── Paths ────────────────────────────────────────────────────────────────────
VENV="$HOME/.git/uv/cosmos-cu134"
REPO="$HOME/.git/cosmos/cosmos-framework"
CUDNN_DIR="$VENV/lib/python3.12/site-packages/nvidia/cudnn"
PATH="$VENV/bin:$PATH"
# Pin TE/torch to the venv's cuDNN (JetPack's system cuDNN 9.27 would otherwise also get loaded).
CUDNN_PATH="$CUDNN_DIR"
LD_LIBRARY_PATH="$CUDNN_DIR/lib"
PYTORCH_ALLOC_CONF="expandable_segments:True"
export PATH CUDNN_PATH LD_LIBRARY_PATH PYTORCH_ALLOC_CONF

NGROK_CONFIG="$HOME/.config/ngrok/ngrok.yml"

# ── Defaults ─────────────────────────────────────────────────────────────────
GRADIO_PORT=7860
GRADIO_IP="127.0.0.1"
CHECKPOINT="$HOME/.cache/huggingface/hub/models--nvidia--Cosmos3-Edge/snapshots/a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba"
OUTPUT_DIR="$HOME/cosmos-out/gradio"
AUTH=""
GUARDRAILS=false
ONLINE=false
GRADIO_SHARE=false
USE_NGROK=false
NGROK_TUNNEL="chat"

# ── Arg parse ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --gradio-port)  GRADIO_PORT="$2";  shift 2 ;;
        --gradio-ip)    GRADIO_IP="$2";    shift 2 ;;
        --checkpoint)   CHECKPOINT="$2";   shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";   shift 2 ;;
        --auth)         AUTH="$2";         shift 2 ;;
        --guardrails)   GUARDRAILS=true;   shift ;;
        --online)       ONLINE=true;       shift ;;
        --share)        GRADIO_SHARE=true; shift ;;
        --ngrok)        USE_NGROK=true;    shift ;;
        --ngrok-tunnel) NGROK_TUNNEL="$2"; USE_NGROK=true; shift 2 ;;
        --help)
            sed -n '/^# Usage/,/^[^#]/{ /^[^#]/d; s/^# \?//; p }' "$0"
            exit 0
            ;;
        *) echo "Unknown option: $1  (use --help)"; exit 1 ;;
    esac
done

# Offline by default: use only the HF cache (Cosmos3-Edge + Wan2.2 VAE are already cached).
if [ "$ONLINE" = true ]; then unset HF_HUB_OFFLINE; else export HF_HUB_OFFLINE=1; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRADIO_PID=""
NGROK_PID=""
NGROK_URL=""
NGROK_UPSTREAM=""
# Resolve the public URL and upstream port for the named tunnel from the ngrok config
if [ -f "$NGROK_CONFIG" ]; then
    NGROK_URL=$(awk -v n="$NGROK_TUNNEL" '$0 ~ "- name: "n"$" {found=1; next} found && /- name:/ {exit} found && /^[[:space:]]+url: https/ {print $2; exit}' "$NGROK_CONFIG")
    NGROK_UPSTREAM=$(awk -v n="$NGROK_TUNNEL" '$0 ~ "- name: "n"$" {found=1; next} found && /- name:/ {exit} found && /upstream:/ {up=1; next} up && /url:/ {print $2; exit}' "$NGROK_CONFIG")
fi

echo "=========================================="
echo " Cosmos3 Studio (cosmos-framework)"
echo "=========================================="
echo " Model  : $CHECKPOINT"
echo " Output : $OUTPUT_DIR"
echo " Gradio : http://${GRADIO_IP}:${GRADIO_PORT}"
echo " HF hub : $( [ "$ONLINE" = true ] && echo online || echo "offline (cache only)" )"
echo " ngrok  : $( [ "$USE_NGROK" = true ] && echo "enabled (named tunnel '$NGROK_TUNNEL' -> ${NGROK_URL:-?})" || echo disabled )"
echo "=========================================="

if [ "$USE_NGROK" = true ] && [ -n "$NGROK_UPSTREAM" ] && [ "${NGROK_UPSTREAM##*:}" != "$GRADIO_PORT" ]; then
    echo "WARNING: ngrok tunnel '$NGROK_TUNNEL' forwards to $NGROK_UPSTREAM but Gradio uses port $GRADIO_PORT."
fi
if { [ "$USE_NGROK" = true ] || [ "$GRADIO_SHARE" = true ]; } && [ -z "$AUTH" ]; then
    echo "WARNING: public tunnel without --auth — anyone with the URL can run generations on this GPU."
fi

# ── Cleanup ──────────────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "Shutting down..."
    for entry in "ngrok:$NGROK_PID" "Gradio:$GRADIO_PID"; do
        svc="${entry%%:*}"
        pid="${entry##*:}"
        if [ -n "$pid" ]; then
            echo "  Stopping $svc (PID $pid)..."
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    echo "Done."
    exit 0
}
trap cleanup SIGINT SIGTERM

# ── Clear stale occupant on Gradio port ──────────────────────────────────────
STALE_PID=$(lsof -ti tcp:"$GRADIO_PORT" 2>/dev/null || true)
if [ -n "$STALE_PID" ]; then
    echo "Port $GRADIO_PORT in use by PID(s) $STALE_PID — killing..."
    kill $STALE_PID 2>/dev/null || true
    sleep 2
fi

# ── Gradio (loads the model in-process) ──────────────────────────────────────
echo ""
echo "Starting Gradio + loading model..."
LOG_FILE="/tmp/cosmos_gradio_${GRADIO_PORT}.log"
# Run from the repo so relative paths in cosmos_framework configs resolve as with the CLI.
cd "$REPO"
GRADIO_CMD=(python "$SCRIPT_DIR/app.py"
    --checkpoint-path "$CHECKPOINT" --output-dir "$OUTPUT_DIR" --repo "$REPO"
    --host "$GRADIO_IP" --port "$GRADIO_PORT")
[ "$GRADIO_SHARE" = true ] && GRADIO_CMD+=(--share)
[ "$GUARDRAILS" = true ] && GRADIO_CMD+=(--guardrails)
[ -n "$AUTH" ] && GRADIO_CMD+=(--auth "$AUTH")
"${GRADIO_CMD[@]}" > >(tee "$LOG_FILE") 2>&1 &
GRADIO_PID=$!

echo "Waiting for model load + Gradio (up to 600s)..."
MAX_WAIT=600
ELAPSED=0
until curl -s -o /dev/null "http://${GRADIO_IP}:${GRADIO_PORT}/"; do
    if ! kill -0 "$GRADIO_PID" 2>/dev/null; then
        echo "Error: app exited unexpectedly — check $LOG_FILE"
        exit 1
    fi
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "Error: startup timed out after ${MAX_WAIT}s — check $LOG_FILE"
        kill "$GRADIO_PID" 2>/dev/null || true
        exit 1
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
done
echo "Gradio is ready (${ELAPSED}s)."

# ── ngrok ────────────────────────────────────────────────────────────────────
if [ "$USE_NGROK" = true ]; then
    echo ""
    echo "Starting ngrok tunnel (config: $NGROK_CONFIG)..."

    # Kill only a leftover ngrok process for this tunnel — leave ssh/other tunnels alone
    pkill -f "ngrok start $NGROK_TUNNEL " 2>/dev/null || true
    sleep 1

    # Stop ghost sessions that survived a previous unclean exit. This stops *all* sessions on the
    # account, so skip it while another ngrok agent (e.g. the 'ssh' tunnel) is running here.
    sessions=""
    if pgrep -x ngrok >/dev/null; then
        echo "Other ngrok agent(s) running — skipping ghost-session cleanup."
    else
        sessions=$(ngrok api tunnel-sessions list 2>/dev/null \
            | grep -oP '"id":\s*"\Kts_[^"]+' || true)
    fi
    if [[ -n "$sessions" ]]; then
        while IFS= read -r sid; do
            ngrok api tunnel-sessions stop "$sid" 2>/dev/null || true
        done <<< "$sessions"
        sleep 1
    fi

    NGROK_LOG="/tmp/ngrok_${NGROK_TUNNEL}.log"
    ngrok start "$NGROK_TUNNEL" --config "$NGROK_CONFIG" --log=stdout \
        > "$NGROK_LOG" 2>&1 &
    NGROK_PID=$!

    # Wait up to 10s for ngrok to establish the tunnel
    for _ in $(seq 1 10); do
        sleep 1
        kill -0 "$NGROK_PID" 2>/dev/null || { echo "Warning: ngrok exited — check $NGROK_LOG"; break; }
        grep -q "started tunnel" "$NGROK_LOG" 2>/dev/null && break
    done

    if [ -n "$NGROK_URL" ]; then
        echo "ngrok public URL: $NGROK_URL"
    else
        echo "Warning: no '$NGROK_TUNNEL' tunnel URL found in $NGROK_CONFIG"
    fi
fi

echo ""
echo "=========================================="
echo " All services running — Ctrl+C to stop"
echo "=========================================="
echo " Gradio : http://${GRADIO_IP}:${GRADIO_PORT}"
[ "$USE_NGROK" = true ] && [ -n "$NGROK_URL" ] && echo " Public : $NGROK_URL"
echo " Log    : $LOG_FILE"
echo "=========================================="
echo ""

wait $GRADIO_PID || true
cleanup
