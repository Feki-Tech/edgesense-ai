#!/usr/bin/env bash
# Deploy the champion model to an Azure ML managed online endpoint.
#
# !! COST WARNING !!
# A managed online endpoint bills for its VM 24/7 while it exists
# (Standard_DS2_v2 ~ EUR 90/month). Create it for a demo/screenshot,
# then DELETE it. Your Container Apps inference sidecar remains the
# cheap always-available serving path.
#
# Usage: ./deploy_endpoint.sh <workspace> <resource-group> [instance-type]
set -euo pipefail

WS="${1:?usage: deploy_endpoint.sh <workspace> <resource-group> [instance-type]}"
RG="${2:?usage: deploy_endpoint.sh <workspace> <resource-group> [instance-type]}"
SKU="${3:-Standard_DS2_v2}"
ENDPOINT="edgesense-anomaly-ep"

az ml online-endpoint create --name "$ENDPOINT" -w "$WS" -g "$RG" \
  --auth-mode key || echo "(endpoint may already exist)"

# Resolve the champion version from the registry tag (no alias support in AML).
CHAMPION=$(az ml model show -n edgesense-anomaly -w "$WS" -g "$RG" \
  --query "tags.champion_version" -o tsv 2>/dev/null || true)
MODEL_REF="azureml:edgesense-anomaly@latest"
[ -n "$CHAMPION" ] && MODEL_REF="azureml:edgesense-anomaly:$CHAMPION"
echo "Deploying model: $MODEL_REF"

az ml online-deployment create --name blue \
  --endpoint-name "$ENDPOINT" -w "$WS" -g "$RG" \
  --file "$(dirname "$0")/deployment.yml" \
  --set model="$MODEL_REF" \
  --set instance_type="$SKU" \
  --all-traffic

echo
echo "Score a reading:"
echo "  KEY=\$(az ml online-endpoint get-credentials -n $ENDPOINT -w $WS -g $RG --query primaryKey -o tsv)"
echo "  URI=\$(az ml online-endpoint show -n $ENDPOINT -w $WS -g $RG --query scoring_uri -o tsv)"
echo "  curl -X POST \$URI -H \"Authorization: Bearer \$KEY\" -H 'Content-Type: application/json' \\"
echo "       -d '{\"vibration\": 3.2, \"temperature\": 88.0, \"current\": 12.5}'"
echo
echo "!! When done demoing: az ml online-endpoint delete -n $ENDPOINT -w $WS -g $RG --yes"
