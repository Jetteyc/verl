#!/usr/bin/env bash
set -euo pipefail

# Sync local verl repo to a remote machine.
# Respects .gitignore, plus a few extra IDE/VCS paths.
#
# Usage:
#   bash scripts/sync_to_remote.sh
#   REMOTE_HOST=10.156.154.36 bash scripts/sync_to_remote.sh
#   DRY_RUN=1 bash scripts/sync_to_remote.sh          # preview only
#   SYNC_DELETE=1 bash scripts/sync_to_remote.sh      # delete remote files not present locally
#
# Environment variables:
#   REMOTE_USER   default: rjy
#   REMOTE_HOST   default: 10.156.154.36
#   REMOTE_PATH   default: ~/projects/verl

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

REMOTE_USER=${REMOTE_USER:-rjy}
REMOTE_HOST=${REMOTE_HOST:-10.156.154.36}
REMOTE_PATH=${REMOTE_PATH:-~/projects/verl}
DRY_RUN=${DRY_RUN:-0}
SYNC_DELETE=${SYNC_DELETE:-0}

if [[ ! -f "${REPO_ROOT}/.gitignore" ]]; then
  echo "error: .gitignore not found under ${REPO_ROOT}" >&2
  exit 1
fi

RSYNC_OPTS=(
  -avz
  --human-readable
  --progress
  # Apply repository .gitignore rules (outputs/, *.log, checkpoints/, wandb/, etc.)
  --filter=':- .gitignore'
  # Extra excludes not always covered or worth always skipping
  --exclude='.git/'
  --exclude='.cursor/'
  --exclude='.idea/'
  --exclude='.claude/'
)

if [[ "${DRY_RUN}" == "1" ]]; then
  RSYNC_OPTS+=(--dry-run)
fi

if [[ "${SYNC_DELETE}" == "1" ]]; then
  RSYNC_OPTS+=(--delete)
fi

DEST="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/"

echo "Syncing ${REPO_ROOT} -> ${DEST}"
echo "  REMOTE_USER=${REMOTE_USER}"
echo "  REMOTE_HOST=${REMOTE_HOST}"
echo "  REMOTE_PATH=${REMOTE_PATH}"
echo "  DRY_RUN=${DRY_RUN}"
echo "  SYNC_DELETE=${SYNC_DELETE}"
echo

ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p ${REMOTE_PATH}"

cd "${REPO_ROOT}"
rsync "${RSYNC_OPTS[@]}" ./ "${DEST}"

echo
echo "Done."
