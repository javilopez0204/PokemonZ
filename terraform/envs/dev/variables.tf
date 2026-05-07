variable "project_id" {
  type        = string
  description = "ID del proyecto GCP"
}

variable "region" {
  type        = string
  default     = "us-central1"
  description = "Región para Cloud Run, Artifact Registry y GCS"
}

variable "app_name" {
  type        = string
  default     = "pokez"
  description = "Prefijo para los recursos"
}

variable "container_image" {
  type        = string
  description = "Imagen Docker completa (host/proyecto/repo/imagen:tag) que Cloud Run desplegará"
}

variable "github_repository" {
  type        = string
  description = "Repositorio GitHub en formato owner/repo (para Workload Identity Federation)"
}

variable "google_api_key" {
  type        = string
  description = "Clave de Google AI Studio para Gemini. Se almacena en Secret Manager."
  sensitive   = true
}

variable "min_instances" {
  type    = number
  default = 0
}

variable "max_instances" {
  type    = number
  default = 2
}

variable "cpu" {
  type    = string
  default = "2"
}

variable "memory" {
  type    = string
  default = "2Gi"
}
