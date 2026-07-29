#!/usr/bin/env bash
# Download Unitree Go2 MuJoCo model assets from mujoco_menagerie
set -euo pipefail

REPO="google-deepmind/mujoco_menagerie"
BRANCH="main"
ASSET_DIR="$(cd "$(dirname "$0")/.." && pwd)/assets/unitree_go2"
RAW="https://raw.githubusercontent.com/${REPO}/${BRANCH}/unitree_go2"

MESHES=(
  base_0.obj base_1.obj base_2.obj base_3.obj base_4.obj
  calf_0.obj calf_1.obj calf_mirror_0.obj calf_mirror_1.obj
  foot.obj
  hip_0.obj hip_1.obj
  thigh_0.obj thigh_1.obj thigh_mirror_0.obj thigh_mirror_1.obj
)

echo "=== Downloading Go2 model assets ==="

# XML files
for f in go2.xml scene.xml; do
  if [ ! -f "${ASSET_DIR}/${f}" ]; then
    echo "  -> ${f}"
    curl -fL "${RAW}/${f}" -o "${ASSET_DIR}/${f}"
  fi
done

# Mesh files — need meshdir subdirectory
mkdir -p "${ASSET_DIR}/assets"
echo "  -> meshes (${#MESHES[@]} files)"
for mesh in "${MESHES[@]}"; do
  if [ ! -f "${ASSET_DIR}/assets/${mesh}" ]; then
    echo "      ${mesh}"
    curl -fL "${RAW}/assets/${mesh}" -o "${ASSET_DIR}/assets/${mesh}"
  fi
done

echo "=== Done! Assets at ${ASSET_DIR} ==="
