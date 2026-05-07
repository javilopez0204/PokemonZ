variable "project_id"         { type = string }
variable "runtime_sa_id"      { type = string }
variable "ci_sa_id"           { type = string }
variable "bucket_name"        { type = string }
variable "artifact_repo_name" { type = string }
variable "region"             { type = string }

# ── Service account del runtime de Cloud Run ────────────────────────────────
resource "google_service_account" "runtime" {
  account_id   = var.runtime_sa_id
  display_name = "Pokédex Z Cloud Run runtime"
  project      = var.project_id
}

resource "google_storage_bucket_iam_member" "runtime_bucket_reader" {
  bucket = var.bucket_name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.runtime.email}"
}

# Permiso para llamar a la API de Generative Language / Vertex (si se usa
# auth via SA en lugar de API key, lo dejamos preparado).
resource "google_project_iam_member" "runtime_aiplatform_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.runtime.email}"
}

# ── Service account de CI ────────────────────────────────────────────────────
resource "google_service_account" "ci" {
  account_id   = var.ci_sa_id
  display_name = "Pokédex Z GitHub Actions CI"
  project      = var.project_id
}

resource "google_project_iam_member" "ci_run_admin" {
  project = var.project_id
  role    = "roles/run.admin"
  member  = "serviceAccount:${google_service_account.ci.email}"
}

resource "google_project_iam_member" "ci_artifact_writer" {
  project = var.project_id
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${google_service_account.ci.email}"
}

resource "google_storage_bucket_iam_member" "ci_bucket_admin" {
  bucket = var.bucket_name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.ci.email}"
}

# Para que CI pueda hacer terraform apply (necesita actuar sobre IAM, secret manager, etc.)
resource "google_project_iam_member" "ci_secret_admin" {
  project = var.project_id
  role    = "roles/secretmanager.admin"
  member  = "serviceAccount:${google_service_account.ci.email}"
}

resource "google_project_iam_member" "ci_iam_admin" {
  project = var.project_id
  role    = "roles/iam.serviceAccountAdmin"
  member  = "serviceAccount:${google_service_account.ci.email}"
}

# CI necesita poder asignar la SA del runtime a Cloud Run
resource "google_service_account_iam_member" "ci_actas_runtime" {
  service_account_id = google_service_account.runtime.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.ci.email}"
}

output "runtime_sa_email" {
  value = google_service_account.runtime.email
}

output "ci_sa_email" {
  value = google_service_account.ci.email
}
