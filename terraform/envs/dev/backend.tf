terraform {
  required_version = ">= 1.6.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }

  # El bucket de tfstate se crea en bootstrap (ver README).
  # Cambia el "bucket" para usar el tuyo.
  backend "gcs" {
    bucket = "pruebaedem-pokez-tfstate"
    prefix = "envs/dev"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
