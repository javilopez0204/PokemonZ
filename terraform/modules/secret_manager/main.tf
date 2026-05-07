variable "project_id"   { type = string }
variable "secret_name"  { type = string }
variable "secret_value" {
  type      = string
  sensitive = true
}
variable "accessor_sas" {
  type        = list(string)
  description = "Emails de service accounts que pueden leer el secreto"
  default     = []
}

resource "google_secret_manager_secret" "this" {
  project   = var.project_id
  secret_id = var.secret_name

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "this" {
  secret      = google_secret_manager_secret.this.id
  secret_data = var.secret_value
}

resource "google_secret_manager_secret_iam_member" "accessors" {
  for_each = toset(var.accessor_sas)

  project   = var.project_id
  secret_id = google_secret_manager_secret.this.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${each.key}"
}

output "secret_id" {
  value = google_secret_manager_secret.this.secret_id
}
