terraform {
  required_version = ">= 1.6.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = ">= 4.2"
    }
    random = {
      source  = "hashicorp/random"
      version = ">= 3.6"
    }
  }

  # For a portfolio project local state is fine. For team use, switch to an
  # azurerm remote backend (uncomment and create the storage account first):
  # backend "azurerm" {
  #   resource_group_name  = "tfstate-rg"
  #   storage_account_name = "edgesensetfstate"
  #   container_name       = "tfstate"
  #   key                  = "edgesense.tfstate"
  # }
}

provider "azurerm" {
  features {}
}
