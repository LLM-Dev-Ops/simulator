# Security Quick Reference Guide
# LLM-Simulator - Security Configuration & Deployment

> **Version**: 1.0.0
> **Date**: 2025-11-26
> **Audience**: DevOps, Security Engineers, SREs

---

## Table of Contents

1. [Security Checklist](#1-security-checklist)
2. [Quick Start: Secure Deployment](#2-quick-start-secure-deployment)
3. [Configuration Templates](#3-configuration-templates)
4. [Security Command Reference](#4-security-command-reference)
5. [Incident Response Quick Actions](#5-incident-response-quick-actions)
6. [Common Security Tasks](#6-common-security-tasks)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Security Checklist

### Pre-Deployment Checklist

```markdown
## Production Deployment Security Checklist

### Authentication & Authorization
- [ ] Admin API key configured and stored in secret manager
- [ ] API key format validation enabled
- [ ] RBAC roles configured correctly
- [ ] Brute force protection enabled (max_failed_attempts: 5)
- [ ] Session timeout configured (default: 3600s)

### Network Security
- [ ] TLS 1.3 enabled
- [ ] Valid TLS certificate installed
- [ ] mTLS configured (if required)
- [ ] Cipher suites restricted to secure algorithms
- [ ] Network policies applied (Kubernetes)
- [ ] Ingress rules configured
- [ ] Egress rules configured
- [ ] IP allowlist configured (if required)

### Data Protection
- [ ] PII detection enabled
- [ ] Auto-redaction configured
- [ ] Log retention policy set
- [ ] Audit retention configured (7 years for compliance)
- [ ] No sensitive data in configuration files

### Audit Logging
- [ ] Audit logging enabled
- [ ] Tamper-evident logging enabled
- [ ] Log signing configured
- [ ] Audit destination configured (file/SIEM)
- [ ] Log shipping to external system configured

### Secret Management
- [ ] Secret manager integration configured
- [ ] All secrets moved out of config files
- [ ] Secret rotation enabled
- [ ] Secrets mounted via CSI driver (Kubernetes)
- [ ] Environment variables secured

### Rate Limiting
- [ ] Per-key rate limits configured
- [ ] Per-IP rate limits configured
- [ ] Burst allowance configured
- [ ] Backpressure threshold set

### Monitoring & Alerts
- [ ] Security metrics exported to Prometheus
- [ ] Alert rules configured
- [ ] SIEM integration configured
- [ ] Security dashboard created
- [ ] On-call rotation configured

### Compliance
- [ ] Required compliance framework identified (SOC2/HIPAA/PCI)
- [ ] Compliance reporting configured
- [ ] Audit log retention meets compliance requirements
- [ ] Security documentation completed

### Testing
- [ ] Security tests passing (unit, integration)
- [ ] Penetration testing completed
- [ ] Vulnerability scan completed (trivy/semgrep)
- [ ] Dependency audit passing (cargo audit)
- [ ] Load testing completed (rate limits verified)

### Documentation
- [ ] Security runbooks updated
- [ ] Incident response plan reviewed
- [ ] Team trained on security procedures
- [ ] Emergency contact list updated
```

---

## 2. Quick Start: Secure Deployment

### Minimal Secure Configuration (Development)

```yaml
# config/security-dev.yaml
authentication:
  enabled: true
  methods:
    api_key:
      enabled: true
      format_regex: "^sk-[a-zA-Z0-9]{32,}$"

authorization:
  enabled: true
  default_deny: true

network_security:
  tls:
    enabled: false  # Use reverse proxy for TLS in dev

  rate_limiting:
    enabled: true
    requests_per_minute: 100

audit_logging:
  enabled: true
  level: "standard"
  destination:
    type: "stdout"
  tamper_evident: false
```

### Production Secure Configuration

```yaml
# config/security-prod.yaml
authentication:
  enabled: true
  methods:
    api_key:
      enabled: true
      format_regex: "^sk-[a-zA-Z0-9]{32,}$"
      max_sessions_per_key: 100
    jwt:
      enabled: false
    mtls:
      enabled: false

  brute_force_protection:
    enabled: true
    max_failed_attempts: 5
    lockout_duration_seconds: 900

authorization:
  enabled: true
  default_deny: true

network_security:
  tls:
    enabled: true
    min_version: "1.3"
    cert_path: "${TLS_CERT_PATH}"
    key_path: "${TLS_KEY_PATH}"
    cipher_suites:
      - "TLS_AES_256_GCM_SHA384"
      - "TLS_CHACHA20_POLY1305_SHA256"

  rate_limiting:
    enabled: true
    requests_per_minute: 1000
    burst_size: 100
    per_api_key: true

  ip_allowlist:
    enabled: false

data_protection:
  pii_handling:
    auto_detect: true
    detection_action: "redact"
    log_detection: true

  retention:
    log_retention_days: 90
    audit_retention_days: 2555
    metrics_retention_days: 365

audit_logging:
  enabled: true
  level: "comprehensive"
  destination:
    type: "file"
    path: "/var/log/llm-simulator/audit.log"
  tamper_evident: true
  sign_logs: true
  signing_key: "${AUDIT_SIGNING_KEY}"

secret_management:
  provider: "aws_secrets_manager"
  region: "us-east-1"
  secret_prefix: "llm-simulator/prod"
  caching:
    enabled: true
    ttl_seconds: 3600
  rotation:
    enabled: true
    period_days: 90

security_monitoring:
  enabled: true
  alerts:
    - name: "high_failed_auth_rate"
      condition: "failed_auth_rate > 10 per minute"
      severity: "high"
```

---

## 3. Configuration Templates

### Kubernetes Deployment with Security

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-simulator
  namespace: llm-simulator
  labels:
    app: llm-simulator
    version: v1.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-simulator
  template:
    metadata:
      labels:
        app: llm-simulator
        version: v1.0.0
    spec:
      serviceAccountName: llm-simulator-sa

      # Security Context (Pod-level)
      securityContext:
        runAsNonRoot: true
        runAsUser: 10000
        runAsGroup: 10000
        fsGroup: 10000
        seccompProfile:
          type: RuntimeDefault

      containers:
      - name: llm-simulator
        image: llm-simulator:1.0.0
        imagePullPolicy: Always

        # Security Context (Container-level)
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          runAsNonRoot: true
          runAsUser: 10000
          capabilities:
            drop:
              - ALL

        ports:
        - name: http
          containerPort: 8080
          protocol: TCP
        - name: metrics
          containerPort: 9090
          protocol: TCP

        env:
        - name: RUST_LOG
          value: "info"
        - name: CONFIG_PATH
          value: "/etc/llm-simulator/config.yaml"
        - name: TLS_CERT_PATH
          value: "/run/secrets/tls/tls.crt"
        - name: TLS_KEY_PATH
          value: "/run/secrets/tls/tls.key"
        - name: ADMIN_API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-simulator-secrets
              key: admin-api-key

        volumeMounts:
        - name: config
          mountPath: /etc/llm-simulator
          readOnly: true
        - name: tls-certs
          mountPath: /run/secrets/tls
          readOnly: true
        - name: tmp
          mountPath: /tmp
        - name: logs
          mountPath: /var/log/llm-simulator

        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"

        livenessProbe:
          httpGet:
            path: /health
            port: 8080
            scheme: HTTP
          initialDelaySeconds: 10
          periodSeconds: 30
          timeoutSeconds: 5
          failureThreshold: 3

        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
            scheme: HTTP
          initialDelaySeconds: 5
          periodSeconds: 10
          timeoutSeconds: 3
          failureThreshold: 3

      volumes:
      - name: config
        configMap:
          name: llm-simulator-config
      - name: tls-certs
        secret:
          secretName: llm-simulator-tls
          defaultMode: 0400
      - name: tmp
        emptyDir: {}
      - name: logs
        emptyDir: {}

      # Node affinity for security (optional)
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - llm-simulator
              topologyKey: kubernetes.io/hostname

---
# Service
apiVersion: v1
kind: Service
metadata:
  name: llm-simulator
  namespace: llm-simulator
  labels:
    app: llm-simulator
spec:
  type: ClusterIP
  ports:
  - name: http
    port: 8080
    targetPort: 8080
    protocol: TCP
  - name: metrics
    port: 9090
    targetPort: 9090
    protocol: TCP
  selector:
    app: llm-simulator

---
# ServiceAccount
apiVersion: v1
kind: ServiceAccount
metadata:
  name: llm-simulator-sa
  namespace: llm-simulator
  annotations:
    # For AWS IAM Role for Service Accounts (IRSA)
    eks.amazonaws.com/role-arn: arn:aws:iam::123456789012:role/llm-simulator-secrets-role

---
# NetworkPolicy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: llm-simulator-netpol
  namespace: llm-simulator
spec:
  podSelector:
    matchLabels:
      app: llm-simulator
  policyTypes:
  - Ingress
  - Egress

  ingress:
  # Allow from ingress controller
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8080

  # Allow from monitoring (Prometheus)
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 9090

  egress:
  # Allow DNS
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: UDP
      port: 53

  # Allow HTTPS (for secret manager)
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: TCP
      port: 443

---
# PodSecurityPolicy (deprecated in K8s 1.25+, use PodSecurity admission)
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: llm-simulator-psp
  annotations:
    seccomp.security.alpha.kubernetes.io/allowedProfileNames: 'runtime/default'
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
  hostNetwork: false
  hostIPC: false
  hostPID: false
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
  readOnlyRootFilesystem: true
```

### Docker Compose (Development)

```yaml
# docker-compose.yaml
version: '3.8'

services:
  llm-simulator:
    image: llm-simulator:1.0.0
    container_name: llm-simulator

    # Security options
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    read_only: true

    user: "10000:10000"

    ports:
      - "8080:8080"
      - "9090:9090"

    environment:
      - RUST_LOG=info
      - CONFIG_PATH=/etc/llm-simulator/config.yaml

    volumes:
      - ./config/security-dev.yaml:/etc/llm-simulator/config.yaml:ro
      - tmp-volume:/tmp
      - log-volume:/var/log/llm-simulator

    tmpfs:
      - /run:mode=1777,size=100M

    restart: unless-stopped

    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 10s

volumes:
  tmp-volume:
  log-volume:
```

---

## 4. Security Command Reference

### Secret Management

```bash
# AWS Secrets Manager

# Create admin API key
aws secretsmanager create-secret \
  --name llm-simulator/prod/admin_api_key \
  --description "LLM-Simulator Admin API Key" \
  --secret-string "sk-admin-$(openssl rand -hex 16)"

# Retrieve secret
aws secretsmanager get-secret-value \
  --secret-id llm-simulator/prod/admin_api_key \
  --query SecretString \
  --output text

# Rotate secret
aws secretsmanager rotate-secret \
  --secret-id llm-simulator/prod/admin_api_key

# List secrets
aws secretsmanager list-secrets \
  --filters Key=name,Values=llm-simulator/prod/

# Delete secret (with recovery window)
aws secretsmanager delete-secret \
  --secret-id llm-simulator/prod/admin_api_key \
  --recovery-window-in-days 30
```

### TLS Certificate Management

```bash
# Generate self-signed certificate (development only)
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout tls.key \
  -out tls.crt \
  -days 365 \
  -subj "/CN=llm-simulator.local"

# Generate certificate signing request (production)
openssl req -new -newkey rsa:4096 -nodes \
  -keyout tls.key \
  -out tls.csr \
  -subj "/C=US/ST=CA/L=SF/O=Example/CN=api.llm-simulator.com"

# Verify certificate
openssl x509 -in tls.crt -text -noout

# Test TLS connection
openssl s_client -connect api.llm-simulator.com:443 -tls1_3

# Create Kubernetes secret for TLS
kubectl create secret tls llm-simulator-tls \
  --cert=tls.crt \
  --key=tls.key \
  --namespace=llm-simulator
```

### Security Scanning

```bash
# Dependency vulnerability scanning
cargo audit

# Container image scanning
trivy image llm-simulator:1.0.0 --severity HIGH,CRITICAL

# Static code analysis
cargo clippy -- -D warnings

# Secret scanning
gitleaks detect --source . --verbose

# SBOM generation
cargo-sbom --output-format spdx > sbom.spdx.json
```

### Audit Log Management

```bash
# Query audit logs (file-based)
jq 'select(.event_type == "AuthenticationFailure")' \
  /var/log/llm-simulator/audit.log

# Count events by type
jq -r '.event_type' /var/log/llm-simulator/audit.log | \
  sort | uniq -c | sort -rn

# Find failed admin API access
jq 'select(.resource.path | startswith("/admin")) |
    select(.result.status == "Failure")' \
  /var/log/llm-simulator/audit.log

# Verify audit log integrity
# (Assuming custom tool for chain verification)
./verify-audit-chain /var/log/llm-simulator/audit.log

# Export audit logs to external system
cat /var/log/llm-simulator/audit.log | \
  curl -X POST https://siem.example.com/api/logs \
  -H "Authorization: Bearer $SIEM_TOKEN" \
  -H "Content-Type: application/json" \
  --data-binary @-
```

---

## 5. Incident Response Quick Actions

### Compromised API Key

```bash
#!/bin/bash
# incident-api-key-compromise.sh

KEY_PREFIX="$1"  # e.g., "sk-abc123"

echo "=== API Key Compromise Response ==="
echo "Key Prefix: $KEY_PREFIX"

# 1. Revoke key (would be in key registry)
echo "[1/5] Revoking key..."
# Implementation-specific: Update key status to Revoked

# 2. Extract all usage from audit logs
echo "[2/5] Extracting key usage from audit logs..."
jq "select(.actor.id | startswith(\"$KEY_PREFIX\"))" \
  /var/log/llm-simulator/audit.log > /tmp/compromised-key-audit.json

# 3. Identify suspicious activity
echo "[3/5] Analyzing for suspicious activity..."
jq 'select(.source_ip != "expected.ip.address")' \
  /tmp/compromised-key-audit.json

# 4. Block source IPs
echo "[4/5] Blocking suspicious IPs..."
# Implementation-specific: Add to IP blocklist

# 5. Notify stakeholders
echo "[5/5] Notifying stakeholders..."
# Send alert via PagerDuty/Slack/Email

echo "✅ Incident response completed"
echo "See /tmp/compromised-key-audit.json for details"
```

### Unauthorized Admin Access Attempt

```bash
#!/bin/bash
# incident-unauthorized-admin.sh

echo "=== Unauthorized Admin Access Response ==="

# 1. Extract failed admin API attempts
echo "[1/4] Extracting failed admin API attempts..."
jq 'select(.resource.path | startswith("/admin")) |
    select(.result.status == "Failure") |
    select(.timestamp > (now - 3600 | strftime("%Y-%m-%dT%H:%M:%SZ")))' \
  /var/log/llm-simulator/audit.log > /tmp/admin-failures.json

# 2. Identify attacking IPs
echo "[2/4] Identifying attacking IPs..."
jq -r '.source_ip' /tmp/admin-failures.json | sort | uniq -c | sort -rn

# 3. Auto-block IPs with >5 failed attempts
echo "[3/4] Blocking IPs with excessive failures..."
jq -r '.source_ip' /tmp/admin-failures.json | \
  sort | uniq -c | \
  awk '$1 > 5 {print $2}' | \
  while read ip; do
    # Add to firewall blocklist
    echo "Blocking IP: $ip"
    # iptables -A INPUT -s $ip -j DROP
  done

# 4. Send alert
echo "[4/4] Sending security alert..."
# Implementation-specific: Send to security team

echo "✅ Response completed"
```

### DDoS Attack

```bash
#!/bin/bash
# incident-ddos-response.sh

echo "=== DDoS Attack Response ==="

# 1. Enable aggressive rate limiting
echo "[1/4] Enabling aggressive rate limiting..."
kubectl set env deployment/llm-simulator \
  -n llm-simulator \
  RATE_LIMIT_RPM=100 \
  RATE_LIMIT_BURST=10

# 2. Identify top attacking IPs
echo "[2/4] Identifying attacking sources..."
jq -r '.source_ip' /var/log/llm-simulator/audit.log | \
  tail -10000 | sort | uniq -c | sort -rn | head -20

# 3. Enable DDoS protection (cloud provider)
echo "[3/4] Activating DDoS protection..."
# AWS Shield Advanced
# aws shield create-protection ...

# 4. Scale infrastructure
echo "[4/4] Scaling infrastructure..."
kubectl scale deployment/llm-simulator \
  --replicas=10 \
  -n llm-simulator

echo "✅ DDoS response activated"
```

---

## 6. Common Security Tasks

### Rotate Secrets

```bash
#!/bin/bash
# rotate-secrets.sh

echo "=== Secret Rotation ==="

# 1. Admin API Key
echo "[1/3] Rotating admin API key..."
NEW_ADMIN_KEY="sk-admin-$(openssl rand -hex 16)"

aws secretsmanager update-secret \
  --secret-id llm-simulator/prod/admin_api_key \
  --secret-string "$NEW_ADMIN_KEY"

echo "New admin key: ${NEW_ADMIN_KEY:0:12}..."

# 2. JWT Signing Key
echo "[2/3] Rotating JWT signing key..."
NEW_JWT_SECRET=$(openssl rand -base64 32)

aws secretsmanager update-secret \
  --secret-id llm-simulator/prod/jwt_secret \
  --secret-string "$NEW_JWT_SECRET"

# 3. Audit Log Signing Key
echo "[3/3] Rotating audit signing key..."
NEW_AUDIT_KEY=$(openssl rand -hex 32)

aws secretsmanager update-secret \
  --secret-id llm-simulator/prod/audit_signing_key \
  --secret-string "$NEW_AUDIT_KEY"

# Trigger pod restart to reload secrets
kubectl rollout restart deployment/llm-simulator -n llm-simulator

echo "✅ Secret rotation completed"
echo "Verify with: kubectl rollout status deployment/llm-simulator -n llm-simulator"
```

### Update TLS Certificate

```bash
#!/bin/bash
# update-tls-cert.sh

CERT_FILE="$1"
KEY_FILE="$2"

echo "=== TLS Certificate Update ==="

# 1. Validate certificate
echo "[1/4] Validating certificate..."
openssl x509 -in "$CERT_FILE" -noout -checkend 0 || {
  echo "❌ Certificate is expired or invalid"
  exit 1
}

# 2. Verify private key matches certificate
echo "[2/4] Verifying key matches certificate..."
CERT_MODULUS=$(openssl x509 -noout -modulus -in "$CERT_FILE" | openssl md5)
KEY_MODULUS=$(openssl rsa -noout -modulus -in "$KEY_FILE" | openssl md5)

if [ "$CERT_MODULUS" != "$KEY_MODULUS" ]; then
  echo "❌ Certificate and key do not match"
  exit 1
fi

# 3. Update Kubernetes secret
echo "[3/4] Updating Kubernetes secret..."
kubectl create secret tls llm-simulator-tls \
  --cert="$CERT_FILE" \
  --key="$KEY_FILE" \
  --namespace=llm-simulator \
  --dry-run=client -o yaml | \
  kubectl apply -f -

# 4. Trigger rolling update
echo "[4/4] Triggering rolling update..."
kubectl rollout restart deployment/llm-simulator -n llm-simulator

echo "✅ TLS certificate updated"
```

### Security Audit

```bash
#!/bin/bash
# security-audit.sh

echo "=== Security Audit ==="

# 1. Check TLS configuration
echo "[1/6] Checking TLS configuration..."
curl -I https://api.llm-simulator.com 2>&1 | grep -i "TLS"

# 2. Verify authentication is enabled
echo "[2/6] Verifying authentication..."
curl -s http://localhost:8080/v1/chat/completions | \
  jq -r '.error.code' | grep -q "401" && \
  echo "✓ Authentication required" || \
  echo "✗ Authentication NOT required"

# 3. Check rate limiting
echo "[3/6] Testing rate limiting..."
for i in {1..110}; do
  curl -s -o /dev/null -w "%{http_code}\n" \
    -H "Authorization: Bearer sk-test" \
    http://localhost:8080/v1/chat/completions
done | grep -c "429"

# 4. Verify audit logging
echo "[4/6] Verifying audit logging..."
[ -f /var/log/llm-simulator/audit.log ] && \
  echo "✓ Audit log exists" || \
  echo "✗ Audit log missing"

# 5. Check security headers
echo "[5/6] Checking security headers..."
curl -I https://api.llm-simulator.com 2>&1 | grep -i "strict-transport-security"

# 6. Scan for vulnerabilities
echo "[6/6] Scanning for vulnerabilities..."
trivy image llm-simulator:1.0.0 --severity HIGH,CRITICAL --quiet

echo "✅ Security audit completed"
```

---

## 7. Troubleshooting

### Authentication Issues

```bash
# Problem: Authentication failing with valid API key

# Check 1: Verify key format
echo "sk-abc123def456ghi789" | grep -E '^sk-[a-zA-Z0-9]{32,}$'

# Check 2: Check audit logs for auth failures
jq 'select(.event_type == "AuthenticationFailure") |
    select(.timestamp > (now - 300 | strftime("%Y-%m-%dT%H:%M:%SZ")))' \
  /var/log/llm-simulator/audit.log

# Check 3: Verify key is not locked out
jq 'select(.event_type == "AuthenticationLockout")' \
  /var/log/llm-simulator/audit.log

# Check 4: Test authentication endpoint
curl -v -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-test123" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]}'
```

### TLS Certificate Issues

```bash
# Problem: TLS handshake failing

# Check 1: Verify certificate is valid
openssl x509 -in /run/secrets/tls/tls.crt -noout -dates

# Check 2: Check certificate chain
openssl s_client -connect api.llm-simulator.com:443 -showcerts

# Check 3: Verify cipher suites
openssl s_client -connect api.llm-simulator.com:443 -tls1_3

# Check 4: Test with specific cipher
openssl s_client -connect api.llm-simulator.com:443 \
  -cipher TLS_AES_256_GCM_SHA384
```

### Audit Log Issues

```bash
# Problem: Audit logs not being written

# Check 1: Verify log directory permissions
ls -la /var/log/llm-simulator/

# Check 2: Check if audit logging is enabled
cat /etc/llm-simulator/config.yaml | grep -A5 "audit_logging:"

# Check 3: Check for disk space
df -h /var/log

# Check 4: Verify log rotation is working
ls -lh /var/log/llm-simulator/audit.log*
```

### Rate Limiting Issues

```bash
# Problem: Rate limits not being enforced

# Check 1: Verify rate limiting is enabled
curl -I http://localhost:8080/v1/chat/completions 2>&1 | \
  grep -i "x-ratelimit"

# Check 2: Check rate limit configuration
cat /etc/llm-simulator/config.yaml | grep -A10 "rate_limiting:"

# Check 3: Monitor rate limit metrics
curl -s http://localhost:9090/metrics | grep rate_limit

# Check 4: Test rate limit enforcement
for i in {1..150}; do
  curl -s -o /dev/null -w "%{http_code} " \
    -H "Authorization: Bearer sk-test" \
    http://localhost:8080/v1/chat/completions
  sleep 0.1
done | grep -c "429"
```

### Secret Manager Issues

```bash
# Problem: Unable to retrieve secrets from AWS Secrets Manager

# Check 1: Verify IAM role is attached (EKS)
kubectl describe serviceaccount llm-simulator-sa -n llm-simulator | \
  grep -i "eks.amazonaws.com/role-arn"

# Check 2: Test secret retrieval
aws secretsmanager get-secret-value \
  --secret-id llm-simulator/prod/admin_api_key \
  --query SecretString \
  --output text

# Check 3: Check IAM permissions
aws sts get-caller-identity

# Check 4: Verify CSI driver is installed
kubectl get csidriver secrets-store.csi.k8s.io
```

---

## Quick Reference Tables

### HTTP Status Codes (Security)

| Status Code | Meaning | Security Implication |
|-------------|---------|---------------------|
| 200 OK | Success | Normal operation |
| 400 Bad Request | Invalid input | Input validation working |
| 401 Unauthorized | Missing/invalid auth | Authentication required |
| 403 Forbidden | Permission denied | Authorization working |
| 429 Too Many Requests | Rate limit exceeded | Rate limiting active |
| 500 Internal Server Error | Server error | Check logs |
| 503 Service Unavailable | Overload/maintenance | Load shedding active |

### Security Headers Reference

| Header | Value | Purpose |
|--------|-------|---------|
| Strict-Transport-Security | max-age=31536000; includeSubDomains | Force HTTPS |
| X-Frame-Options | DENY | Prevent clickjacking |
| X-Content-Type-Options | nosniff | Prevent MIME sniffing |
| X-XSS-Protection | 1; mode=block | XSS protection |
| Content-Security-Policy | default-src 'self' | Restrict resource loading |
| Referrer-Policy | strict-origin-when-cross-origin | Control referrer info |

### Common Log Queries

```bash
# Failed authentication in last hour
jq 'select(.event_type == "AuthenticationFailure") |
    select(.timestamp > (now - 3600 | strftime("%Y-%m-%dT%H:%M:%SZ")))' \
  audit.log

# Top 10 IPs by request count
jq -r '.source_ip' audit.log | sort | uniq -c | sort -rn | head -10

# Admin API access attempts
jq 'select(.resource.path | startswith("/admin"))' audit.log

# Rate limit exceeded events
jq 'select(.event_type == "RateLimitExceeded")' audit.log

# Configuration changes
jq 'select(.event_type == "ConfigurationChanged")' audit.log

# Authorization denials
jq 'select(.event_type == "AuthorizationDenied")' audit.log
```

---

## Emergency Contacts

```yaml
# emergency-contacts.yaml

security_team:
  name: "Security Team"
  email: "security@example.com"
  pagerduty: "https://example.pagerduty.com/security"
  slack: "#security-incidents"

operations_team:
  name: "Operations Team"
  email: "ops@example.com"
  pagerduty: "https://example.pagerduty.com/ops"
  slack: "#ops-alerts"

compliance_team:
  name: "Compliance Team"
  email: "compliance@example.com"
  slack: "#compliance"

incident_commander:
  name: "Security Incident Commander"
  phone: "+1-555-0100"
  email: "incident-commander@example.com"
```

---

## Additional Resources

- **Full Security Architecture**: SECURITY_ARCHITECTURE.md
- **Security Diagrams**: SECURITY_DIAGRAMS.md
- **Core Simulation Engine**: CORE_SIMULATION_ENGINE.md
- **HTTP Server Design**: HTTP_SERVER_DESIGN_SUMMARY.md
- **Error Injection Framework**: ERROR_INJECTION_FRAMEWORK.md

---

**Document Control**:
- **Version**: 1.0.0
- **Last Updated**: 2025-11-26
- **Next Review**: 2026-02-26
- **Target Audience**: DevOps Engineers, Security Engineers, SREs
