#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="$HOME/.local/bin"
BIN_PATH="$BIN_DIR/image-cull"

echo "==> Building container image 'image-cull:latest'..."
if command -v podman >/dev/null 2>&1; then
    CONTAINER_ENGINE="podman"
elif command -v docker >/dev/null 2>&1; then
    CONTAINER_ENGINE="docker"
else
    echo "Error: Neither podman nor docker was found on your PATH."
    exit 1
fi

$CONTAINER_ENGINE build -t image-cull:latest "$PROJECT_DIR"

echo "==> Installing executable wrapper script to $BIN_PATH..."
mkdir -p "$BIN_DIR"

cat << 'EOF' > "$BIN_PATH"
#!/usr/bin/env bash
TARGET_DIR="."
FILTER_HOST_DIR=""
IS_DRY_RUN=false
ARGS=()

skip_next=false
for ((i=1; i<=$#; i++)); do
    if [ "$skip_next" = true ]; then
        skip_next=false
        continue
    fi
    arg="${!i}"
    next_index=$((i+1))
    next_arg="${!next_index}"

    if [ "$arg" == "--dry-run" ]; then
        IS_DRY_RUN=true
        ARGS+=("$arg")
    elif [ "$arg" == "--filter-dir" ]; then
        if [ -z "$next_arg" ] || [[ "$next_arg" == -* ]]; then
            echo "Error: --filter-dir requires a directory path argument." >&2
            exit 1
        fi
        FILTER_HOST_DIR="$next_arg"
        skip_next=true
    elif [[ "$arg" == --filter-dir=* ]]; then
        FILTER_HOST_DIR="${arg#*=}"
        if [ -z "$FILTER_HOST_DIR" ]; then
            echo "Error: --filter-dir requires a non-empty directory path." >&2
            exit 1
        fi
    elif [[ "$arg" != -* ]] && [ -d "$arg" ]; then
        TARGET_DIR="$arg"
    else
        ARGS+=("$arg")
    fi
done

REAL_HOST_DIR="$(realpath "$TARGET_DIR")"
MOUNTS=("-v" "$REAL_HOST_DIR:/photos:z")

CONTAINER_FLAGS=()
if [ -n "$FILTER_HOST_DIR" ]; then
    if [ "$IS_DRY_RUN" = false ]; then
        mkdir -p "$FILTER_HOST_DIR"
    fi
    if [ -d "$FILTER_HOST_DIR" ]; then
        REAL_FILTER_DIR="$(realpath "$FILTER_HOST_DIR")"
        MOUNTS+=("-v" "$REAL_FILTER_DIR:/filtered:z")
        CONTAINER_FLAGS+=("--filter-dir" "/filtered")
    fi
fi

CONTAINER_ENGINE="podman"
if ! command -v podman >/dev/null 2>&1; then
    CONTAINER_ENGINE="docker"
fi

exec $CONTAINER_ENGINE run --rm --network host \
    "${MOUNTS[@]}" \
    image-cull:latest --dir /photos --report-path-display "$REAL_HOST_DIR/cull_report.json" "${CONTAINER_FLAGS[@]}" "${ARGS[@]}"
EOF

chmod +x "$BIN_PATH"

echo ""
echo "=========================================================="
echo " Setup complete!"
echo " Executable binary installed to: $BIN_PATH"
echo "=========================================================="
echo " You can now run image-cull from anywhere using:"
echo "   image-cull ~/Downloads --filter-dir ~/Desktop/Filtered_Photos --dry-run"
echo ""
