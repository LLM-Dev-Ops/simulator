# LLM-Simulator Deployment Artifacts Index

**Complete reference for all deployment-related files and documentation**

---

## Documentation

### Primary Guides

| Document | Description | Path |
|----------|-------------|------|
| **Deployment Architecture** | Complete enterprise deployment architecture and patterns | [DEPLOYMENT_ARCHITECTURE.md](/workspaces/llm-simulator/DEPLOYMENT_ARCHITECTURE.md) |
| **Quick Start Guide** | Get deployed in 15 minutes | [DEPLOYMENT_QUICKSTART.md](/workspaces/llm-simulator/DEPLOYMENT_QUICKSTART.md) |
| **This Index** | Navigation for all deployment files | [DEPLOYMENT_INDEX.md](/workspaces/llm-simulator/DEPLOYMENT_INDEX.md) |

---

## Container Configuration

### Docker

| File | Purpose | Path |
|------|---------|------|
| **Dockerfile** | Multi-stage production container build | [Dockerfile](/workspaces/llm-simulator/Dockerfile) |
| **.dockerignore** | Files excluded from Docker build context | [.dockerignore](/workspaces/llm-simulator/.dockerignore) |
| **docker-compose.yml** | Multi-service local deployment | [docker-compose.yml](/workspaces/llm-simulator/docker-compose.yml) |

**Key Features**:
- Multi-stage build (builder + runtime)
- Non-root user (UID 1000)
- Health checks included
- Multi-architecture support (amd64, arm64)
- Final image size: ~95MB

---

## Kubernetes Manifests

### Core Resources

| File | Resource Type | Purpose | Path |
|------|--------------|---------|------|
| **deployment.yaml** | Deployment | Standard stateless deployment pattern | [deploy/kubernetes/deployment.yaml](/workspaces/llm-simulator/deploy/kubernetes/deployment.yaml) |
| **statefulset.yaml** | StatefulSet | Stateful deployment with persistence | [deploy/kubernetes/statefulset.yaml](/workspaces/llm-simulator/deploy/kubernetes/statefulset.yaml) |
| **daemonset.yaml** | DaemonSet | Node-local deployment for edge | [deploy/kubernetes/daemonset.yaml](/workspaces/llm-simulator/deploy/kubernetes/daemonset.yaml) |

### Networking

| File | Resource Type | Purpose | Path |
|------|--------------|---------|------|
| **service.yaml** | Service | ClusterIP, LoadBalancer, NodePort, Headless services | [deploy/kubernetes/service.yaml](/workspaces/llm-simulator/deploy/kubernetes/service.yaml) |
| **ingress.yaml** | Ingress | Nginx, AWS ALB, GCP GLB, Azure App Gateway, Istio | [deploy/kubernetes/ingress.yaml](/workspaces/llm-simulator/deploy/kubernetes/ingress.yaml) |

### Configuration

| File | Resource Type | Purpose | Path |
|------|--------------|---------|------|
| **configmap.yaml** | ConfigMap | Application and provider profile configuration | [deploy/kubernetes/configmap.yaml](/workspaces/llm-simulator/deploy/kubernetes/configmap.yaml) |

### Auto-Scaling

| File | Resource Type | Purpose | Path |
|------|--------------|---------|------|
| **hpa.yaml** | HorizontalPodAutoscaler | Standard HPA, custom metrics HPA, VPA, KEDA | [deploy/kubernetes/hpa.yaml](/workspaces/llm-simulator/deploy/kubernetes/hpa.yaml) |

**Scaling Features**:
- Standard HPA (CPU/Memory)
- Custom metrics HPA (RPS, queue depth, latency)
- Vertical Pod Autoscaler (VPA)
- KEDA for event-driven scaling
- Scale-to-zero support

---

## Helm Chart

### Chart Structure

```
deploy/helm/llm-simulator/
├── Chart.yaml              # Chart metadata
├── values.yaml             # Default configuration
├── templates/              # Kubernetes templates
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   ├── hpa.yaml
│   ├── pdb.yaml
│   ├── serviceaccount.yaml
│   ├── servicemonitor.yaml
│   └── NOTES.txt
└── README.md
```

| File | Purpose | Path |
|------|---------|------|
| **Chart.yaml** | Helm chart metadata and dependencies | [deploy/helm/llm-simulator/Chart.yaml](/workspaces/llm-simulator/deploy/helm/llm-simulator/Chart.yaml) |
| **values.yaml** | Default Helm values with comprehensive options | [deploy/helm/llm-simulator/values.yaml](/workspaces/llm-simulator/deploy/helm/llm-simulator/values.yaml) |

**Values Configuration Categories**:
- Deployment type (Deployment/StatefulSet/DaemonSet)
- Image and pull policy
- Resource limits and requests
- Auto-scaling configuration
- Ingress and service settings
- Security contexts
- Observability integration
- Provider profiles

---

## CI/CD Pipelines

### GitHub Actions

| File | Purpose | Triggers | Path |
|------|---------|----------|------|
| **ci.yml** | Continuous Integration | Push, PR, Manual | [.github/workflows/ci.yml](/workspaces/llm-simulator/.github/workflows/ci.yml) |
| **cd.yml** | Continuous Deployment | Tag push, Manual | [.github/workflows/cd.yml](/workspaces/llm-simulator/.github/workflows/cd.yml) |

**CI Pipeline Stages**:
1. Code quality (lint, format check)
2. Security audit (cargo-audit, cargo-deny)
3. Build and test (multi-platform)
4. Code coverage
5. Docker build (multi-arch)
6. Integration tests
7. Performance benchmarks
8. Helm chart validation
9. Kubernetes manifest validation

**CD Pipeline Stages**:
1. Release build and publish
2. Deploy to staging
3. Integration tests on staging
4. Deploy to production (canary)
5. Canary validation
6. Full production rollout
7. Multi-cloud deployment (AWS, Azure, GCP)
8. Post-deployment validation

---

## Monitoring and Observability

### Prometheus

| File | Purpose | Path |
|------|---------|------|
| **prometheus.yml** | Prometheus scrape configuration | [deploy/prometheus/prometheus.yml](/workspaces/llm-simulator/deploy/prometheus/prometheus.yml) |
| **alerts.yml** | Alerting rules | [deploy/prometheus/rules/alerts.yml](/workspaces/llm-simulator/deploy/prometheus/rules/alerts.yml) |

**Alert Categories**:
- Availability (uptime, restarts, replicas)
- Performance (latency, TTFT)
- Error rates
- Resource usage (CPU, memory, throttling)
- Queue depth and concurrency
- SLO compliance
- Capacity planning

**Metrics Collected**:
- Request rate and duration
- Error counts by type
- Token generation metrics
- Queue depth
- Resource utilization
- Pod health status

---

## Build and Automation

### Makefile

| File | Purpose | Targets | Path |
|------|---------|---------|------|
| **Makefile** | Build, test, deploy automation | 50+ targets | [Makefile](/workspaces/llm-simulator/Makefile) |

**Target Categories**:
- Development (setup, dev, fmt, lint)
- Building (build, clean, dist)
- Testing (test, bench, coverage)
- Security (audit, deny)
- Docker (build, push, run, scan)
- Docker Compose (up, down, logs)
- Kubernetes (apply, delete, status, logs)
- Helm (lint, install, upgrade, uninstall)
- CI/CD helpers (ci, release-build, release-publish)
- Monitoring (metrics, health, load-test)

**Quick Command Examples**:
```bash
make help           # Show all available targets
make ci             # Run all CI checks
make docker-build   # Build Docker image
make helm-install   # Install with Helm
make k8s-status     # Check deployment status
```

---

## Configuration Examples

### Application Configuration

| File | Purpose | Path |
|------|---------|------|
| **simulator.example.yaml** | Complete example configuration | [simulator.example.yaml](/workspaces/llm-simulator/simulator.example.yaml) |

**Configuration Sections**:
- Server settings (host, port, connections)
- Provider profiles (GPT-4, Claude, etc.)
- Simulation parameters (concurrency, sessions)
- Telemetry (logging, metrics, tracing)
- Scenarios (load patterns, assertions)
- Security (auth, API keys, IP filtering)
- Feature flags

---

## Deployment Patterns Summary

### Pattern Selection Guide

| Pattern | Use Case | State | Scaling | Complexity |
|---------|----------|-------|---------|------------|
| **Deployment** | General production use | Stateless | Horizontal (HPA) | Low |
| **StatefulSet** | Session persistence, deterministic testing | Stateful | Horizontal (ordered) | Medium |
| **DaemonSet** | Edge computing, node-local | Stateless | Fixed (1 per node) | Low |

### Cloud Platform Support

| Platform | Ingress | Storage | Identity | Guide |
|----------|---------|---------|----------|-------|
| **AWS** | ALB | EBS (gp3) | IRSA | [Values](#) |
| **Azure** | App Gateway | Azure Disk | Managed Identity | [Values](#) |
| **GCP** | GLB | Persistent Disk | Workload Identity | [Values](#) |
| **On-Prem** | Nginx | Local/NFS | Service Account | [Values](#) |

---

## Resource Sizing Guidelines

### Pod Resource Recommendations

| Environment | CPU Request | CPU Limit | Memory Request | Memory Limit | Replicas |
|-------------|-------------|-----------|----------------|--------------|----------|
| **Development** | 250m | 1000m | 512Mi | 2Gi | 1 |
| **Staging** | 500m | 2000m | 1Gi | 4Gi | 2 |
| **Production (Light)** | 1000m | 4000m | 2Gi | 8Gi | 3-10 |
| **Production (Heavy)** | 2000m | 8000m | 4Gi | 16Gi | 5-20 |
| **High-Throughput** | 4000m | 16000m | 8Gi | 32Gi | 10-50 |

### Capacity Planning

**Per-Pod Capacity**:
- Light workload: ~100 RPS
- Medium workload: ~500 RPS
- Heavy workload: ~1,000 RPS
- Optimized: ~2,000 RPS

**Example Calculation**:
```
Target: 10,000 RPS
Per-pod capacity: 500 RPS
Required pods: 10,000 / 500 = 20 pods
With 25% headroom: 25 pods
HPA config: min=25, max=50
```

---

## Security Configuration

### Security Layers

1. **Container Security**:
   - Non-root user (UID 1000)
   - Read-only root filesystem
   - No privilege escalation
   - Dropped capabilities
   - Seccomp profile

2. **Kubernetes Security**:
   - Pod Security Standards (Restricted)
   - Network Policies
   - RBAC
   - Service Account
   - Pod Disruption Budget

3. **Network Security**:
   - TLS termination at ingress
   - mTLS with service mesh (optional)
   - Network segmentation
   - Egress filtering

4. **Application Security**:
   - Input validation
   - Rate limiting
   - Request size limits
   - Timeout enforcement
   - Audit logging

---

## Deployment Checklist

### Pre-Deployment

- [ ] Review configuration changes
- [ ] Validate resource quotas
- [ ] Check cluster capacity
- [ ] Run CI pipeline (all checks pass)
- [ ] Backup current configuration
- [ ] Notify stakeholders

### Deployment

- [ ] Deploy to staging first
- [ ] Run integration tests
- [ ] Monitor metrics during rollout
- [ ] Verify health checks pass
- [ ] Check pod events

### Post-Deployment

- [ ] Validate endpoints
- [ ] Run smoke tests
- [ ] Check dashboards
- [ ] Verify logs streaming
- [ ] Update documentation
- [ ] Send notification

---

## Troubleshooting Resources

### Common Issues

| Issue | Check | Fix |
|-------|-------|-----|
| Pods not starting | `kubectl describe pod` | Check image, resources, config |
| High latency | `kubectl top pods` | Scale up or increase resources |
| Connection issues | `kubectl get endpoints` | Check service and network policy |
| Config not applied | `kubectl get configmap` | Restart pods after config change |
| Ingress not working | `kubectl describe ingress` | Check ingress controller logs |

### Useful Commands

```bash
# Status
kubectl get all -n llm-devops
kubectl get events -n llm-devops --sort-by='.lastTimestamp'

# Logs
kubectl logs -n llm-devops -l app=llm-simulator -f

# Debug
kubectl describe pod <pod-name> -n llm-devops
kubectl exec -it <pod-name> -n llm-devops -- sh

# Metrics
kubectl top pods -n llm-devops
kubectl top nodes

# Health
curl http://<pod-ip>:8080/health
curl http://<pod-ip>:9090/metrics
```

---

## Quick Reference

### Essential Files

```
llm-simulator/
├── Dockerfile                              # Container build
├── docker-compose.yml                      # Local stack
├── Makefile                                # Build automation
├── simulator.example.yaml                  # Config example
├── DEPLOYMENT_ARCHITECTURE.md              # Architecture guide
├── DEPLOYMENT_QUICKSTART.md                # Quick start
├── DEPLOYMENT_INDEX.md                     # This file
├── .github/workflows/
│   ├── ci.yml                              # CI pipeline
│   └── cd.yml                              # CD pipeline
├── deploy/
│   ├── kubernetes/
│   │   ├── deployment.yaml                 # K8s Deployment
│   │   ├── statefulset.yaml                # K8s StatefulSet
│   │   ├── daemonset.yaml                  # K8s DaemonSet
│   │   ├── service.yaml                    # K8s Services
│   │   ├── ingress.yaml                    # K8s Ingress
│   │   ├── configmap.yaml                  # K8s ConfigMaps
│   │   └── hpa.yaml                        # Auto-scaling
│   ├── helm/llm-simulator/
│   │   ├── Chart.yaml                      # Helm metadata
│   │   └── values.yaml                     # Helm values
│   └── prometheus/
│       ├── prometheus.yml                  # Prometheus config
│       └── rules/alerts.yml                # Alert rules
```

### Essential Commands

```bash
# Build and Test
make ci                                     # Run all CI checks
make build                                  # Build binary
make docker-build                           # Build container

# Local Development
make dev                                    # Run with auto-reload
docker-compose up -d                        # Start local stack

# Kubernetes Deployment
make helm-install                           # Deploy with Helm
make k8s-status                             # Check status
make k8s-logs                               # View logs

# Troubleshooting
make health                                 # Health check
make metrics                                # View metrics
kubectl describe pod <name> -n llm-devops   # Debug pod
```

---

## Support

- **Documentation**: Complete guides in this repository
- **Issues**: [GitHub Issues](https://github.com/llm-devops/llm-simulator/issues)
- **Discussions**: [GitHub Discussions](https://github.com/llm-devops/llm-simulator/discussions)
- **Examples**: [/examples](examples/) directory

---

**Document Version**: 1.0.0
**Last Updated**: 2025-11-26
**Total Artifacts**: 20+ files covering complete deployment lifecycle
