output "service_url" {
  description = "URL pública del servicio Cloud Run"
  value       = module.cloud_run.url
}

output "bucket_name" {
  description = "Bucket de GCS con el PDF y el índice"
  value       = module.gcs.bucket_name
}

output "artifact_repo" {
  description = "Repositorio Artifact Registry para las imágenes"
  value       = module.artifact_registry.repository_url
}

output "runtime_service_account" {
  description = "Service account del runtime de Cloud Run"
  value       = module.iam.runtime_sa_email
}

output "ci_service_account" {
  description = "Service account usada por GitHub Actions"
  value       = module.iam.ci_sa_email
}

output "workload_identity_provider" {
  description = "Recurso WIF que GitHub Actions debe usar como `workload_identity_provider`"
  value       = module.workload_identity.provider_resource
}
