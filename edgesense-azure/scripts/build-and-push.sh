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

# NOTE: context is the REPO ROOT for every service — the inference image
# bakes a trained model at build time and needs ml/ from the root; keeping
# one context for all four matches docker-compose behavior.
declare -A SERVICES=(
  ["edgesense-inference"]="inference/Dockerfile"
  ["edgesense-agent"]="edge-agent/Dockerfile"
  ["edgesense-simulator"]="simulator/Dockerfile"
  ["edgesense-dashboard"]="dashboard/Dockerfile"
)

for image in "${!SERVICES[@]}"; do
  dockerfile="${SERVICES[$image]}"
  echo ">> Building ${image}:${TAG} (context: ${REPO_ROOT}, file: ${dockerfile})"
  az acr build \
    --registry "${ACR_NAME}" \
    --image "${image}:${TAG}" \
    --file "${REPO_ROOT}/${dockerfile}" \
    "${REPO_ROOT}"
done

echo "All images pushed to ${ACR_NAME} with tag ${TAG}."
