locals {
  bucket_name             = "${var.project_id}-${var.app_name}-data"
  artifact_repo           = "${var.app_name}-images"
  cloud_run_service       = "${var.app_name}-api"
  service_account         = "${var.app_name}-runtime"
  ci_service_account      = "${var.app_name}-ci"
  workload_identity_pool  = "${var.app_name}-gha"
  workload_identity_prov  = "github"
  api_key_secret_name     = "${var.app_name}-google-api-key"
}

# APIs necesarias
resource "google_project_service" "services" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "storage.googleapis.com",
    "secretmanager.googleapis.com",
    "iamcredentials.googleapis.com",
    "iam.googleapis.com",
    "aiplatform.googleapis.com",
    "generativelanguage.googleapis.com",
    "cloudresourcemanager.googleapis.com",
  ])
  project            = var.project_id
  service            = each.key
  disable_on_destroy = false
}

module "gcs" {
  source      = "../../modules/gcs"
  bucket_name = local.bucket_name
  region      = var.region
  depends_on  = [google_project_service.services]
}

module "artifact_registry" {
  source     = "../../modules/artifact_registry"
  project_id = var.project_id
  region     = var.region
  repo_id    = local.artifact_repo
  depends_on = [google_project_service.services]
}

module "iam" {
  source             = "../../modules/iam"
  project_id         = var.project_id
  runtime_sa_id      = local.service_account
  ci_sa_id           = local.ci_service_account
  bucket_name        = module.gcs.bucket_name
  artifact_repo_name = module.artifact_registry.repository_id
  region             = var.region
  depends_on         = [google_project_service.services]
}

module "secret_manager" {
  source        = "../../modules/secret_manager"
  project_id    = var.project_id
  secret_name   = local.api_key_secret_name
  secret_value  = var.google_api_key
  accessor_sas  = [module.iam.runtime_sa_email]
  depends_on    = [google_project_service.services]
}

module "workload_identity" {
  source             = "../../modules/workload_identity"
  project_id         = var.project_id
  pool_id            = local.workload_identity_pool
  provider_id        = local.workload_identity_prov
  github_repository  = var.github_repository
  ci_sa_email        = module.iam.ci_sa_email
  depends_on         = [google_project_service.services]
}

module "cloud_run" {
  source            = "../../modules/cloud_run"
  project_id        = var.project_id
  region            = var.region
  service_name      = local.cloud_run_service
  image             = var.container_image
  service_account   = module.iam.runtime_sa_email
  min_instances     = var.min_instances
  max_instances     = var.max_instances
  cpu               = var.cpu
  memory            = var.memory
  bucket_name       = module.gcs.bucket_name
  api_key_secret_id = module.secret_manager.secret_id
  depends_on        = [module.iam, module.secret_manager]
}
