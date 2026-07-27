#!/usr/bin/env bash
# Compatibility entrypoint for the 64-worker member of the unified matrix.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
echo "run_64env_serial.sh now delegates to the matched SAC sampling-scale runner." >&2
exec env EXPERIMENTS=parallel_64env "$SCRIPT_DIR/run_all_experiments.sh" "$@"
