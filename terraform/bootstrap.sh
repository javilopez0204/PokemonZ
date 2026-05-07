#!/usr/bin/env bash
# Bootstrap previo al primer `terraform apply`.
# Crea proyecto (opcional), habilita APIs, y crea el bucket de tfstate.
#
# Uso:
#   PROJECT_ID=mi-proyecto REGION=us-central1 ./bootstrap.sh
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Define PROJECT_ID}"
REGION="${REGION:-us-central1}"
APP="${APP:-pokez}"
TFSTATE_BUCKET="${TFSTATE_BUCKET:-${PROJECT_ID}-${APP}-tfstate}"

echo "→ Configurando gcloud para proyecto $PROJECT_ID"
gcloud config set project "$PROJECT_ID" >/dev/null

echo "→ Habilitando APIs (puede tardar 1-2 min)"
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com \
  secretmanager.googleapis.com \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  cloudresourcemanager.googleapis.com \
  aiplatform.googleapis.com \
  generativelanguage.googleapis.com

echo "→ Creando bucket de tfstate gs://$TFSTATE_BUCKET (si no existe)"
if ! gcloud storage buckets describe "gs://$TFSTATE_BUCKET" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://$TFSTATE_BUCKET" \
    --location="$REGION" \
    --uniform-bucket-level-access \
    --public-access-prevention
  gcloud storage buckets update "gs://$TFSTATE_BUCKET" --versioning
fi

echo
echo "✓ Bootstrap listo."
echo
echo "Edita terraform/envs/dev/backend.tf y pon:"
echo "  bucket = \"$TFSTATE_BUCKET\""
echo
echo "Después: cd terraform/envs/dev && terraform init && terraform apply"
