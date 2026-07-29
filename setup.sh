#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="$HOME/.local/bin"
BIN_PATH="$BIN_DIR/audit-realism"

echo "==> Building container image 'image-auditor:latest'..."
if command -v podman >/dev/null 2>&1; then
    CONTAINER_ENGINE="podman"
elif command -v docker >/dev/null 2>&1; then
    CONTAINER_ENGINE="docker"
else
    echo "Error: Neither podman nor docker was found on your PATH."
    exit 1
fi

$CONTAINER_ENGINE build -t image-auditor:latest "$PROJECT_DIR"

echo "==> Installing executable wrapper script to $BIN_PATH..."
mkdir -p "$BIN_DIR"

cat << 'EOF' > "$BIN_PATH"
#!/usr/bin/env bash
TARGET_DIR="."
TRASH_HOST_DIR=""
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

    if [ "$arg" == "--trash-dir" ]; then
        TRASH_HOST_DIR="$next_arg"
        skip_next=true
    elif [[ "$arg" != -* ]] && [ -d "$arg" ]; then
        TARGET_DIR="$arg"
    else
        ARGS+=("$arg")
    fi
done

REAL_HOST_DIR="$(realpath "$TARGET_DIR")"
MOUNTS=("-v" "$REAL_HOST_DIR:/photos:z")

CONTAINER_FLAGS=()
if [ -n "$TRASH_HOST_DIR" ]; then
    mkdir -p "$TRASH_HOST_DIR"
    REAL_TRASH_DIR="$(realpath "$TRASH_HOST_DIR")"
    MOUNTS+=("-v" "$REAL_TRASH_DIR:/trash:z")
    CONTAINER_FLAGS+=("--trash-dir" "/trash")
fi

CONTAINER_ENGINE="podman"
if ! command -v podman >/dev/null 2>&1; then
    CONTAINER_ENGINE="docker"
fi

exec $CONTAINER_ENGINE run --rm --network host \
    "${MOUNTS[@]}" \
    image-auditor:latest --dir /photos --report-path-display "$REAL_HOST_DIR/realism_audit_report.json" "${CONTAINER_FLAGS[@]}" "${ARGS[@]}"
EOF

chmod +x "$BIN_PATH"

echo ""
echo "=========================================================="
echo " Setup complete!"
echo " Executable binary installed to: $BIN_PATH"
echo "=========================================================="
echo " You can now run the auditor from anywhere using:"
echo "   audit-realism ~/Downloads --trash-dir ~/Desktop/Trash --dry-run"
echo ""
