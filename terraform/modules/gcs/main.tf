variable "bucket_name" { type = string }
variable "region"      { type = string }

resource "google_storage_bucket" "data" {
  name                        = var.bucket_name
  location                    = var.region
  force_destroy               = false
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition { num_newer_versions = 5 }
    action    { type = "Delete" }
  }
}

output "bucket_name" {
  value = google_storage_bucket.data.name
}

output "bucket_url" {
  value = google_storage_bucket.data.url
}
