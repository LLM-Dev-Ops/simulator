#!/bin/bash
# Phase 7 - Intelligence & Expansion (Layer 2) - Cloud Run Deployment
#
# INFRASTRUCTURE CONTEXT:
# - Google Cloud Run
# - RuvVector REQUIRED
# - Secrets in Google Secret Manager
#
# ROLE CLARITY:
# - Agents MAY: reason, simulate, explore
# - Agents MUST: emit signals, avoid final decisions
#
# DECISION EVENT RULES (Signal Emission):
# - hypothesis_signal
# - simulation_outcome_signal
# - confidence_delta_signal
#
# PERFORMANCE BUDGETS:
# - MAX_TOKENS=2500
# - MAX_LATENCY_MS=5000

set -euo pipefail

# ============================================================================
# Configuration Variables
# ============================================================================

PROJECT_ID="${PROJECT_ID:-your-gcp-project}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-llm-simulator}"
IMAGE="${IMAGE:-gcr.io/${PROJECT_ID}/${SERVICE_NAME}:phase7}"

# Secret names in Google Secret Manager
RUVVECTOR_URL_SECRET="${RUVVECTOR_URL_SECRET:-ruvvector-service-url}"
API_KEY_SECRET="${API_KEY_SECRET:-llm-simulator-api-key}"
ADMIN_KEY_SECRET="${ADMIN_KEY_SECRET:-llm-simulator-admin-key}"

# Performance budgets (Phase 7)
INTELLIGENCE_MAX_TOKENS="${INTELLIGENCE_MAX_TOKENS:-2500}"
INTELLIGENCE_MAX_LATENCY_MS="${INTELLIGENCE_MAX_LATENCY_MS:-5000}"

# ============================================================================
# Deploy Command
# ============================================================================

echo "Deploying Phase 7 - Intelligence & Expansion (Layer 2)"
echo "======================================================="
echo "Project:  ${PROJECT_ID}"
echo "Region:   ${REGION}"
echo "Service:  ${SERVICE_NAME}"
echo "Image:    ${IMAGE}"
echo ""
echo "Performance Budgets:"
echo "  MAX_TOKENS:     ${INTELLIGENCE_MAX_TOKENS}"
echo "  MAX_LATENCY_MS: ${INTELLIGENCE_MAX_LATENCY_MS}"
echo ""

gcloud run deploy "${SERVICE_NAME}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --image="${IMAGE}" \
    --platform=managed \
    --allow-unauthenticated \
    --memory=2Gi \
    --cpu=2 \
    --timeout=60s \
    --concurrency=100 \
    --min-instances=1 \
    --max-instances=10 \
    --set-env-vars="RUST_LOG=info" \
    --set-env-vars="LLM_SIMULATOR_PORT=8080" \
    --set-env-vars="LLM_SIMULATOR_HOST=0.0.0.0" \
    --set-env-vars="RUVVECTOR_ENABLED=true" \
    --set-env-vars="RUVVECTOR_REQUIRE=true" \
    --set-env-vars="RUVVECTOR_FALLBACK_TO_MOCK=false" \
    --set-env-vars="RUVVECTOR_ALLOW_MOCKS=false" \
    --set-env-vars="INTELLIGENCE_ENABLED=true" \
    --set-env-vars="INTELLIGENCE_MAX_TOKENS=${INTELLIGENCE_MAX_TOKENS}" \
    --set-env-vars="INTELLIGENCE_MAX_LATENCY_MS=${INTELLIGENCE_MAX_LATENCY_MS}" \
    --set-secrets="RUVVECTOR_SERVICE_URL=${RUVVECTOR_URL_SECRET}:latest" \
    --set-secrets="LLM_SIMULATOR_API_KEY=${API_KEY_SECRET}:latest" \
    --set-secrets="LLM_SIMULATOR_ADMIN_KEY=${ADMIN_KEY_SECRET}:latest"

echo ""
echo "Deployment complete!"
echo ""

# Get service URL
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --format='value(status.url)')

echo "Service URL: ${SERVICE_URL}"
echo ""
echo "Health check: curl ${SERVICE_URL}/health"
echo "Metrics:      curl ${SERVICE_URL}/metrics"
