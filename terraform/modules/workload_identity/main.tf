variable "project_id"        { type = string }
variable "pool_id"           { type = string }
variable "provider_id"       { type = string }
variable "github_repository" {
  type        = string
  description = "owner/repo"
}
variable "ci_sa_email" { type = string }

resource "google_iam_workload_identity_pool" "gha" {
  project                   = var.project_id
  workload_identity_pool_id = var.pool_id
  display_name              = "GitHub Actions"
}

resource "google_iam_workload_identity_pool_provider" "github" {
  project                            = var.project_id
  workload_identity_pool_id          = google_iam_workload_identity_pool.gha.workload_identity_pool_id
  workload_identity_pool_provider_id = var.provider_id
  display_name                       = "GitHub OIDC"

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.actor"      = "assertion.actor"
    "attribute.repository" = "assertion.repository"
    "attribute.ref"        = "assertion.ref"
  }

  # Sólo permitimos tokens de este repo
  attribute_condition = "assertion.repository == \"${var.github_repository}\""
}

# La SA de CI confía en identidades de este pool restringidas al repo
resource "google_service_account_iam_member" "ci_wif" {
  service_account_id = "projects/${var.project_id}/serviceAccounts/${var.ci_sa_email}"
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.gha.name}/attribute.repository/${var.github_repository}"
}

output "provider_resource" {
  description = "Valor para `workload_identity_provider` en google-github-actions/auth"
  value       = google_iam_workload_identity_pool_provider.github.name
}

output "pool_name" {
  value = google_iam_workload_identity_pool.gha.name
}
