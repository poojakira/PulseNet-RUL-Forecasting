variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "production"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "pulsenet-eks"
}

variable "kubernetes_version" {
  description = "Kubernetes version for EKS"
  type        = string
  default     = "1.31"
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnet_cidrs" {
  description = "Private subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnet_cidrs" {
  description = "Public subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

variable "node_group_instance_types" {
  description = "Instance types for EKS node group"
  type        = list(string)
  default     = ["t3.medium", "t3a.medium"]
}

variable "node_group_desired_size" {
  description = "Desired node count"
  type        = number
  default     = 3
}

variable "node_group_min_size" {
  description = "Minimum node count"
  type        = number
  default     = 2
}

variable "node_group_max_size" {
  description = "Maximum node count"
  type        = number
  default     = 10
}

variable "node_group_disk_size" {
  description = "Node group disk size in GB"
  type        = number
  default     = 50
}

variable "tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default = {
    Project     = "PulseNet-RUL-Forecasting"
    Environment = "production"
    ManagedBy   = "Terraform"
  }
}
