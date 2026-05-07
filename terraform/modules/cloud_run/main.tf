variable "project_id"        { type = string }
variable "region"            { type = string }
variable "service_name"      { type = string }
variable "image"             { type = string }
variable "service_account"   { type = string }
variable "min_instances"     { type = number }
variable "max_instances"     { type = number }
variable "cpu"               { type = string }
variable "memory"            { type = string }
variable "bucket_name"       { type = string }
variable "api_key_secret_id" { type = string }

resource "google_cloud_run_v2_service" "api" {
  project  = var.project_id
  name     = var.service_name
  location = var.region

  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    service_account = var.service_account

    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    containers {
      image = var.image

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = var.cpu
          memory = var.memory
        }
        cpu_idle          = true
        startup_cpu_boost = true
      }

      env {
        name  = "GCP_PROJECT"
        value = var.project_id
      }
      env {
        name  = "GCP_REGION"
        value = var.region
      }
      env {
        name  = "GCS_BUCKET"
        value = var.bucket_name
      }
      env {
        name = "GOOGLE_API_KEY"
        value_source {
          secret_key_ref {
            secret  = var.api_key_secret_id
            version = "latest"
          }
        }
      }

      startup_probe {
        initial_delay_seconds = 5
        period_seconds        = 10
        timeout_seconds       = 5
        failure_threshold     = 30
        http_get {
          path = "/api/health"
        }
      }
    }

    timeout = "300s"
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }
}

# Acceso público
resource "google_cloud_run_v2_service_iam_member" "public" {
  project  = var.project_id
  location = google_cloud_run_v2_service.api.location
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

output "url" {
  value = google_cloud_run_v2_service.api.uri
}

output "name" {
  value = google_cloud_run_v2_service.api.name
}
