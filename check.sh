#!/usr/bin/env bash
# Local + CI validation. Keep in lockstep with .github/workflows/ci.yml.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Pin matches CI/local; bump here when upgrading ruff.
RUFF_VERSION="0.16.2"

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "error: missing '$1' on PATH" >&2
    exit 1
  }
}

need python
need shellcheck

# Prefer a working docker (CI); fall back to podman (common on Fedora).
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  CONTAINER_ENGINE=docker
elif command -v podman >/dev/null 2>&1; then
  CONTAINER_ENGINE=podman
else
  echo "error: need a working docker or podman on PATH" >&2
  exit 1
fi

if ! command -v ruff >/dev/null 2>&1 || [[ "$(ruff --version)" != "ruff ${RUFF_VERSION}" ]]; then
  echo "==> pip install ruff==${RUFF_VERSION}"
  pip install -q "ruff==${RUFF_VERSION}"
fi

echo "==> python image_cull.py --self-check"
python image_cull.py --self-check

echo "==> ruff check image_cull.py"
ruff check image_cull.py

echo "==> shellcheck setup.sh check.sh"
shellcheck setup.sh check.sh

echo "==> ${CONTAINER_ENGINE} build -t image-cull:local ."
"${CONTAINER_ENGINE}" build -t image-cull:local .

echo "==> ${CONTAINER_ENGINE} run --rm image-cull:local --self-check"
"${CONTAINER_ENGINE}" run --rm image-cull:local --self-check

echo "OK: all checks passed"
