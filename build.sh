#!/usr/bin/env bash
# Build the lyra2-dgx-spark image and populate the host assets directory.
#
#   ./build.sh [extra `docker compose build` arguments, e.g. --no-cache]
#
# 1. Creates the host directories that compose bind-mounts (assets, and outputs/checkpoints if the
#    defaults point at missing paths), owned by the current user rather than root.
# 2. Builds the image (the Dockerfile clones the Lyra source itself).
# 3. Runs the image once with `sync-assets`, which copies every file and directory of the cloned
#    Lyra-2/assets that does not yet exist in the host assets directory. Nothing is overwritten, so
#    re-running after editing or adding files on the host is safe; new upstream assets after a
#    rebuild are added the same way.
#
# The host assets directory (LYRA_ASSETS_DIR, default ./assets) is what inference reads.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[[ -f .env ]] && set -a && source .env && set +a
export UID GID="${GID:-$(id -g)}"

ASSETS="${LYRA_ASSETS_DIR:-./assets}"
mkdir -p "$ASSETS" "${LYRA_OUTPUT_DIR:-./outputs}"
[[ -d "${LYRA_CHECKPOINTS_DIR:-./checkpoints}" ]] || echo "note: checkpoints directory '${LYRA_CHECKPOINTS_DIR:-./checkpoints}' does not exist yet; inference will fall back to the HF cache" >&2

echo "== building lyra2-dgx-spark ==" >&2
docker compose build "$@"

echo "== syncing cloned assets into $ASSETS ==" >&2
docker compose run --rm -T lyra2 sync-assets

echo "== done. assets on host: $(find "$ASSETS" -type f | wc -l) files under $ASSETS ==" >&2
