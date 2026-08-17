# PulseNet Terraform Deployment

## Overview

This Terraform configuration deploys the PulseNet-RUL-Forecasting infrastructure on AWS:
- **EKS** cluster with managed node groups
- **VPC** with public/private subnets across 3 AZs
- **ECR** repository for container images
- **Secrets Manager** for application secrets
- **S3** backend with DynamoDB state locking
- **CloudWatch** log groups

## Prerequisites

- Terraform >= 1.6
- AWS CLI configured with appropriate credentials
- `kubectl` installed
- Route53 hosted zone (for ingress, if desired)

## Quick Start

```bash
# 1. Initialize Terraform
cd deploy/terraform
terraform init

# 2. Review the plan
terraform plan

# 3. Apply
terraform apply -auto-approve

# 4. Configure kubectl
$(terraform output -raw configure_kubectl)

# 5. Deploy application
kubectl apply -f ../k8s/namespace.yaml
kubectl apply -f ../k8s/serviceaccount.yaml
kubectl apply -f ../k8s/configmap.yaml
kubectl apply -f ../k8s/secret.yaml
kubectl apply -f ../k8s/deployment.yaml
kubectl apply -f ../k8s/service.yaml
kubectl apply -f ../k8s/hpa.yaml
kubectl apply -f ../k8s/network-policy.yaml
kubectl apply -f ../k8s/pdb.yaml
```

## Infrastructure Architecture

```
Internet
   |
   └── Internet Gateway
         |
         └── Public Subnets (3 AZs)
               |── NAT Gateway
               |── Load Balancers
               |
               └── Private Subnets (3 AZs)
                     |── EKS Cluster (control plane)
                     |── EKS Node Groups (worker nodes)
                     |── ECR (container registry)
                     └── Secrets Manager
```

## Variables

See `variables.tf` for all configurable parameters. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `region` | `us-east-1` | AWS region |
| `cluster_name` | `pulsenet-eks` | EKS cluster name |
| `kubernetes_version` | `1.31` | K8s version |
| `node_group_desired_size` | `3` | Initial node count |
| `node_group_min_size` | `2` | Minimum nodes |
| `node_group_max_size` | `10` | Maximum nodes |

## Security

- Cluster endpoint is **private only** (no public API server access)
- ECR images are **immutable** and scanned on push
- ECR uses **KMS** encryption
- Terraform state stored in **encrypted S3** with **DynamoDB locking**
- Secrets stored in **AWS Secrets Manager** with automatic rotation

## State Management

Terraform state is stored in S3 with DynamoDB locking:

```bash
bucket: pulsenet-terraform-state
dynamodb: pulsenet-terraform-locks
```
