# PulseNet Kubernetes Deployment

## Prerequisites

- Kubernetes 1.28+
- kubectl configured with cluster access
- [external-secrets](https://external-secrets.io/) operator installed
- AWS Secrets Manager configured (for secret management)

## Quick Start

```bash
# 1. Create namespace and resources
kubectl apply -f deploy/k8s/namespace.yaml

# 2. Deploy secrets via ExternalSecrets
kubectl apply -f deploy/k8s/secret.yaml

# 3. Configure RBAC
kubectl apply -f deploy/k8s/serviceaccount.yaml

# 4. Deploy application config
kubectl apply -f deploy/k8s/configmap.yaml

# 5. Deploy the application
kubectl apply -f deploy/k8s/deployment.yaml

# 6. Expose the service
kubectl apply -f deploy/k8s/service.yaml

# 7. Configure autoscaling
kubectl apply -f deploy/k8s/hpa.yaml

# 8. Apply network policies
kubectl apply -f deploy/k8s/network-policy.yaml

# 9. Apply PodDisruptionBudget
kubectl apply -f deploy/k8s/pdb.yaml
```

## Verify Deployment

```bash
kubectl get all -n pulsenet
kubectl get pods -n pulsenet -w
kubectl get hpa -n pulsenet
```

## Architecture

| Resource | Purpose |
|----------|---------|
| Namespace | Isolates PulseNet resources |
| ConfigMap | Non-sensitive configuration |
| ExternalSecret | AWS Secrets Manager integration for secrets |
| ServiceAccount | Minimal RBAC for pod operations |
| Deployment | 3 replicas with HA pod anti-affinity |
| Service | Internal ClusterIP on port 8000 |
| HPA | Auto-scaling (CPU 70%, 2-10 pods) |
| NetworkPolicy | Zero-trust default-deny |
| PodDisruptionBudget | 1 pod minimum during disruptions |

## Security Features

- **runAsNonRoot**: Containers run as non-root user
- **readOnlyRootFilesystem**: Writable only through emptyDir volumes
- **Capabilities drop all**: No Linux capabilities
- **Seccomp**: RuntimeDefault profile
- **PodAntiAffinity**: Pods spread across nodes
- **NetworkPolicies**: Default deny with minimal allow rules
- **ExternalSecrets**: No secrets in Git

## Monitoring

- Metrics exposed at `/metrics` (Prometheus format)
- Liveness probe: `GET /healthz`
- Readiness probe: `GET /readyz`
- Prometheus auto-discovery via annotations

## Graceful Shutdown

The deployment includes a `preStop` hook that sends SIGTERM and waits 30 seconds for clean connection draining.
