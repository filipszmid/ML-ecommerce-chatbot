terraform {
  required_version = ">= 1.7.0"

  required_providers {
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.7"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 6.15"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
