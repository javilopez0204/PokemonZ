variable "project_id" { type = string }
variable "region"     { type = string }
variable "repo_id"    { type = string }

resource "google_artifact_registry_repository" "images" {
  project       = var.project_id
  location      = var.region
  repository_id = var.repo_id
  format        = "DOCKER"
  description   = "Imágenes Docker de Pokédex Z"
}

output "repository_id" {
  value = google_artifact_registry_repository.images.repository_id
}

output "repository_url" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}
