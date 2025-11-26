# LLM-Simulator Deployment Quick Start

**Get LLM-Simulator running in production in under 15 minutes**

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start Options](#quick-start-options)
3. [Local Development](#local-development)
4. [Docker Deployment](#docker-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Cloud Platform Deployments](#cloud-platform-deployments)
7. [Verification and Testing](#verification-and-testing)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Tools

| Tool | Version | Purpose | Installation |
|------|---------|---------|--------------|
| **kubectl** | 1.24+ | Kubernetes CLI | [Install](https://kubernetes.io/docs/tasks/tools/) |
| **helm** | 3.10+ | Package manager | [Install](https://helm.sh/docs/intro/install/) |
| **docker** | 20.10+ | Container runtime | [Install](https://docs.docker.com/get-docker/) |

### Optional Tools

| Tool | Purpose |
|------|---------|
| **make** | Simplified command execution |
| **curl** | API testing |
| **jq** | JSON parsing |

### Access Requirements

- Kubernetes cluster (local or cloud)
- Container registry access (for custom builds)
- kubectl configured with cluster access

---

## Quick Start Options

### Option 1: Single Command (Helm - Recommended)

```bash
# Add Helm repository (when published)
helm repo add llm-devops https://llm-devops.github.io/charts
helm repo update

# Install with defaults
helm install llm-simulator llm-devops/llm-simulator \
  --namespace llm-devops \
  --create-namespace

# Verify installation
kubectl get pods -n llm-devops -l app.kubernetes.io/name=llm-simulator
```

**Time to deployment: ~2 minutes**

### Option 2: Local Repository (Development)

```bash
# Clone repository
git clone https://github.com/llm-devops/llm-simulator.git
cd llm-simulator

# Deploy with Makefile
make helm-install

# Or manually
helm install llm-simulator deploy/helm/llm-simulator \
  --namespace llm-devops \
  --create-namespace
```

**Time to deployment: ~3 minutes**

### Option 3: Docker Compose (Local Testing)

```bash
# Clone repository
git clone https://github.com/llm-devops/llm-simulator.git
cd llm-simulator

# Start all services
docker-compose up -d

# Access services
# - Simulator: http://localhost:8080
# - Prometheus: http://localhost:9091
# - Grafana: http://localhost:3000
```

**Time to deployment: ~1 minute**

---

## Local Development

### Single Binary

**Download and Run**:

```bash
# Download latest release
VERSION=1.0.0
wget https://github.com/llm-devops/llm-simulator/releases/download/v${VERSION}/llm-simulator-${VERSION}-linux-x86_64.tar.gz

# Extract
tar xzf llm-simulator-${VERSION}-linux-x86_64.tar.gz

# Run with default configuration
./llm-simulator serve

# Run with custom configuration
./llm-simulator serve --config simulator.yaml --port 8080
```

**Build from Source**:

```bash
# Install Rust (if not installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone and build
git clone https://github.com/llm-devops/llm-simulator.git
cd llm-simulator
cargo build --release

# Run
./target/release/llm-simulator serve
```

### Configuration

Create `simulator.yaml`:

```yaml
version: "1.0"

server:
  host: "127.0.0.1"
  port: 8080
  max_connections: 1000

simulation:
  max_concurrent_sessions: 100

telemetry:
  enabled: true
  logging:
    level: "info"
  metrics:
    enabled: true

features:
  streaming: true
```

---

## Docker Deployment

### Pre-built Image

```bash
# Pull latest image
docker pull ghcr.io/llm-devops/llm-simulator:latest

# Run container
docker run -d \
  --name llm-simulator \
  -p 8080:8080 \
  -p 9090:9090 \
  ghcr.io/llm-devops/llm-simulator:latest

# Check health
curl http://localhost:8080/health
```

### Custom Configuration

```bash
# Create config directory
mkdir -p config profiles

# Copy example configuration
cp simulator.example.yaml config/simulator.yaml

# Run with mounted config
docker run -d \
  --name llm-simulator \
  -p 8080:8080 \
  -p 9090:9090 \
  -v $(pwd)/config:/app/config:ro \
  -v $(pwd)/profiles:/app/profiles:ro \
  ghcr.io/llm-devops/llm-simulator:latest
```

### Docker Compose

```bash
# Clone repository
git clone https://github.com/llm-devops/llm-simulator.git
cd llm-simulator

# Start stack (simulator + observability)
docker-compose up -d

# View logs
docker-compose logs -f llm-simulator

# Stop stack
docker-compose down
```

---

## Kubernetes Deployment

### Method 1: Helm (Recommended)

**Basic Installation**:

```bash
# Install with defaults
helm install llm-simulator deploy/helm/llm-simulator \
  --namespace llm-devops \
  --create-namespace

# Wait for pods to be ready
kubectl wait --for=condition=ready pod \
  -l app.kubernetes.io/name=llm-simulator \
  -n llm-devops \
  --timeout=300s
```

**Production Installation**:

```bash
# Create custom values file
cat > values-prod.yaml <<EOF
replicaCount: 5

resources:
  requests:
    cpu: 1000m
    memory: 2Gi
  limits:
    cpu: 4000m
    memory: 8Gi

autoscaling:
  enabled: true
  minReplicas: 5
  maxReplicas: 50

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: llm-simulator.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: llm-simulator-tls
      hosts:
        - llm-simulator.example.com

podDisruptionBudget:
  enabled: true
  minAvailable: 3
EOF

# Install with production values
helm install llm-simulator deploy/helm/llm-simulator \
  --namespace llm-devops \
  --create-namespace \
  --values values-prod.yaml

# Verify deployment
helm status llm-simulator -n llm-devops
kubectl get all -n llm-devops -l app.kubernetes.io/name=llm-simulator
```

**Upgrade Release**:

```bash
# Upgrade to new version
helm upgrade llm-simulator deploy/helm/llm-simulator \
  --namespace llm-devops \
  --set image.tag=1.1.0 \
  --values values-prod.yaml

# Rollback if needed
helm rollback llm-simulator -n llm-devops
```

### Method 2: kubectl (Manual)

```bash
# Create namespace
kubectl create namespace llm-devops

# Apply ConfigMaps
kubectl apply -f deploy/kubernetes/configmap.yaml

# Apply Deployment
kubectl apply -f deploy/kubernetes/deployment.yaml

# Apply Service
kubectl apply -f deploy/kubernetes/service.yaml

# Apply HPA
kubectl apply -f deploy/kubernetes/hpa.yaml

# Apply Ingress (if needed)
kubectl apply -f deploy/kubernetes/ingress.yaml

# Check status
kubectl get pods -n llm-devops -w
```

### Method 3: Makefile

```bash
# Clone repository
git clone https://github.com/llm-devops/llm-simulator.git
cd llm-simulator

# Install with Helm
make helm-install

# Or apply raw manifests
make k8s-apply

# Check status
make k8s-status

# View logs
make k8s-logs
```

---

## Cloud Platform Deployments

### AWS (EKS)

**Prerequisites**:
- EKS cluster
- AWS CLI configured
- kubectl configured for EKS

**Deploy**:

```bash
# Update kubeconfig
aws eks update-kubeconfig --name llm-devops-prod --region us-east-1

# Create values file for AWS
cat > values-aws.yaml <<EOF
ingress:
  enabled: true
  className: alb
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/listen-ports: '[{"HTTP": 80}, {"HTTPS": 443}]'
    alb.ingress.kubernetes.io/ssl-redirect: '443'

persistence:
  enabled: true
  storageClass: gp3
  size: 50Gi

serviceAccount:
  create: true
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::ACCOUNT_ID:role/llm-simulator-role
EOF

# Deploy
helm install llm-simulator deploy/helm/llm-simulator \
  --namespace llm-devops \
  --create-namespace \
  --values values-aws.yaml
```

### Azure (AKS)

**Prerequisites**:
- AKS cluster
- Azure CLI configured
- kubectl configured for AKS

**Deploy**:

```bash
# Get AKS credentials
az aks get-credentials --resource-group llm-devops-rg --name llm-devops-prod

# Create values file for Azure
cat > values-azure.yaml <<EOF
ingress:
  enabled: true
  className: azure-application-gateway
  annotations:
    appgw.ingress.kubernetes.io/ssl-redirect: "true"

persistence:
  enabled: true
  storageClass: managed-premium
  size: 50Gi

serviceAccount:
  create: true
  annotations:
    azure.workload.identity/client-id: CLIENT_ID
EOF

# Deploy
helm install llm-simulator deploy/helm/llm-simulator \
  --namespace llm-devops \
  --create-namespace \
  --values values-azure.yaml
```

### GCP (GKE)

**Prerequisites**:
- GKE cluster
- gcloud CLI configured
- kubectl configured for GKE

**Deploy**:

```bash
# Get GKE credentials
gcloud container clusters get-credentials llm-devops-prod --region us-central1

# Create values file for GCP
cat > values-gcp.yaml <<EOF
ingress:
  enabled: true
  className: gce
  annotations:
    kubernetes.io/ingress.global-static-ip-name: llm-simulator-ip
    networking.gke.io/managed-certificates: llm-simulator-cert

persistence:
  enabled: true
  storageClass: standard-rwo
  size: 50Gi

serviceAccount:
  create: true
  annotations:
    iam.gke.io/gcp-service-account: llm-simulator@PROJECT_ID.iam.gserviceaccount.com
EOF

# Deploy
helm install llm-simulator deploy/helm/llm-simulator \
  --namespace llm-devops \
  --create-namespace \
  --values values-gcp.yaml
```

---

## Verification and Testing

### Health Checks

```bash
# Direct pod access
kubectl port-forward -n llm-devops svc/llm-simulator 8080:8080

# Health endpoint
curl http://localhost:8080/health

# Expected response: {"status":"healthy"}

# Readiness endpoint
curl http://localhost:8080/ready

# Expected response: {"status":"ready"}
```

### Functional Testing

```bash
# Test OpenAI-compatible endpoint
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4-turbo",
    "messages": [
      {"role": "user", "content": "Hello, world!"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'

# Expected: Simulated chat completion response
```

### Streaming Test

```bash
# Test streaming endpoint
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4-turbo",
    "messages": [{"role": "user", "content": "Count to 5"}],
    "stream": true
  }'

# Expected: Server-sent events (SSE) stream
```

### Metrics Verification

```bash
# Port forward Prometheus metrics
kubectl port-forward -n llm-devops svc/llm-simulator 9090:9090

# Fetch metrics
curl http://localhost:9090/metrics

# Check specific metrics
curl -s http://localhost:9090/metrics | grep llm_simulator_requests_total
```

### Load Testing

```bash
# Simple load test (100 concurrent requests)
for i in {1..100}; do
  curl -X POST http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-4-turbo","messages":[{"role":"user","content":"test"}]}' &
done
wait

# Monitor pod metrics during load
kubectl top pods -n llm-devops -l app.kubernetes.io/name=llm-simulator
```

---

## Troubleshooting

### Pods Not Starting

**Check pod status**:
```bash
kubectl get pods -n llm-devops
kubectl describe pod <pod-name> -n llm-devops
```

**Common issues**:
1. **ImagePullBackOff**: Registry authentication issue
   ```bash
   # Check image pull secret
   kubectl get secrets -n llm-devops
   ```

2. **CrashLoopBackOff**: Configuration or resource issue
   ```bash
   # Check logs
   kubectl logs <pod-name> -n llm-devops --previous
   ```

3. **Pending**: Resource constraints
   ```bash
   # Check node resources
   kubectl describe nodes | grep -A 5 "Allocated resources"
   ```

### High Latency

**Check pod resources**:
```bash
kubectl top pods -n llm-devops
```

**Check HPA status**:
```bash
kubectl get hpa -n llm-devops
kubectl describe hpa llm-simulator-hpa -n llm-devops
```

**Scale manually if needed**:
```bash
kubectl scale deployment llm-simulator -n llm-devops --replicas=10
```

### Connection Issues

**Check service**:
```bash
kubectl get svc llm-simulator -n llm-devops
kubectl describe svc llm-simulator -n llm-devops
```

**Check endpoints**:
```bash
kubectl get endpoints llm-simulator -n llm-devops
```

**Test from within cluster**:
```bash
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- \
  curl http://llm-simulator.llm-devops:8080/health
```

### Configuration Issues

**View current config**:
```bash
kubectl get configmap llm-simulator-config -n llm-devops -o yaml
```

**Update config**:
```bash
# Edit ConfigMap
kubectl edit configmap llm-simulator-config -n llm-devops

# Restart pods to pick up changes
kubectl rollout restart deployment llm-simulator -n llm-devops
```

### Ingress Not Working

**Check ingress status**:
```bash
kubectl get ingress -n llm-devops
kubectl describe ingress llm-simulator-ingress -n llm-devops
```

**Check ingress controller**:
```bash
kubectl get pods -n ingress-nginx
kubectl logs -n ingress-nginx <ingress-controller-pod>
```

### Viewing Logs

**Real-time logs**:
```bash
kubectl logs -n llm-devops -l app.kubernetes.io/name=llm-simulator -f
```

**Specific pod**:
```bash
kubectl logs <pod-name> -n llm-devops
```

**Previous crash logs**:
```bash
kubectl logs <pod-name> -n llm-devops --previous
```

### Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `Connection refused` | Service not ready | Wait for pods to be ready |
| `404 Not Found` | Wrong endpoint | Check API path |
| `429 Too Many Requests` | Rate limit hit | Increase rate limits in config |
| `500 Internal Server Error` | Configuration error | Check pod logs |
| `OOMKilled` | Memory limit too low | Increase memory limits |

---

## Next Steps

After successful deployment:

1. **Configure Monitoring**:
   - Set up Grafana dashboards
   - Configure Prometheus alerts
   - Enable distributed tracing

2. **Customize Configuration**:
   - Add custom provider profiles
   - Adjust latency distributions
   - Configure error injection

3. **Performance Tuning**:
   - Optimize resource allocation
   - Fine-tune auto-scaling
   - Configure caching

4. **Security Hardening**:
   - Enable TLS/mTLS
   - Set up authentication
   - Configure network policies

5. **Integration**:
   - Connect to LLM-Gateway
   - Configure LLM-Orchestrator
   - Set up LLM-Analytics-Hub

---

## Support and Resources

- **Documentation**: [Full Deployment Guide](/workspaces/llm-simulator/DEPLOYMENT_ARCHITECTURE.md)
- **GitHub Issues**: [Report bugs or request features](https://github.com/llm-devops/llm-simulator/issues)
- **Community**: [Join discussions](https://github.com/llm-devops/llm-simulator/discussions)
- **Examples**: [/workspaces/llm-simulator/examples/](examples/)

---

**Quick Command Reference**:

```bash
# Deployment
make helm-install              # Install with Helm
make k8s-apply                 # Apply Kubernetes manifests
make compose-up                # Start Docker Compose stack

# Verification
make health                    # Check health
make k8s-status                # Check Kubernetes status
make metrics                   # View metrics

# Troubleshooting
make k8s-logs                  # View logs
make k8s-describe              # Describe deployment

# Cleanup
make helm-uninstall            # Uninstall Helm release
make k8s-delete                # Delete Kubernetes resources
make compose-down              # Stop Docker Compose
```

---

**Time to Production**: 15 minutes from zero to fully deployed and tested LLM-Simulator instance!
