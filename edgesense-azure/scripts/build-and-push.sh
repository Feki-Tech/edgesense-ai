#!/usr/bin/env bash
# Build EdgeSense images and push them to Azure Container Registry.
# Uses `az acr build` so the build runs in Azure (no local Docker needed)
# and never leaves the credit-friendly path.
#
# Usage:
#   ./build-and-push.sh <acr-name> [tag] [repo-root]
# Example:
#   ./build-and-push.sh edgesenseacrab123 latest ../edgesense-ai
set -euo pipefail

ACR_NAME="${1:?Usage: build-and-push.sh <acr-name> [tag] [repo-root]}"
TAG="${2:-latest}"
REPO_ROOT="${3:-.}"

# image -> "dockerfile:context" (paths relative to REPO_ROOT). Inference
# builds from the repo root because it bakes ml/ at build time; the other
# services expect their own directory as context (mirrors docker-compose).
declare -A SERVICES=(
  ["edgesense-inference"]="inference/Dockerfile:."
  ["edgesense-agent"]="edge-agent/Dockerfile:edge-agent"
  ["edgesense-simulator"]="simulator/Dockerfile:simulator"
  ["edgesense-dashboard"]="dashboard/Dockerfile:dashboard"
)

for image in "${!SERVICES[@]}"; do
  IFS=: read -r dockerfile context <<< "${SERVICES[$image]}"
  echo ">> Building ${image}:${TAG} (context: ${REPO_ROOT}/${context}, file: ${dockerfile})"
  az acr build \
    --registry "${ACR_NAME}" \
    --image "${image}:${TAG}" \
    --file "${REPO_ROOT}/${dockerfile}" \
    "${REPO_ROOT}/${context}"
done

echo "All images pushed to ${ACR_NAME} with tag ${TAG}."
