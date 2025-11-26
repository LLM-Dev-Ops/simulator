# Security Architecture Diagrams
# LLM-Simulator - Visual Security Reference

> **Version**: 1.0.0
> **Date**: 2025-11-26
> **Companion to**: SECURITY_ARCHITECTURE.md

---

## Table of Contents

1. [High-Level Security Architecture](#1-high-level-security-architecture)
2. [Authentication Flow Diagrams](#2-authentication-flow-diagrams)
3. [Authorization Decision Flow](#3-authorization-decision-flow)
4. [Network Security Topology](#4-network-security-topology)
5. [Data Flow Diagrams](#5-data-flow-diagrams)
6. [Threat Model Diagrams](#6-threat-model-diagrams)

---

## 1. High-Level Security Architecture

### 1.1 Security Layers Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     EXTERNAL CLIENTS                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │  Web Client  │  │  Mobile App  │  │  CI/CD Tool  │                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
└────────────────────────────┬────────────────────────────────────────────┘
                             │ HTTPS/TLS 1.3
                             │ Optional: mTLS
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     SECURITY PERIMETER LAYER                             │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  WAF / DDoS Protection / Load Balancer                       │      │
│  │  • IP Filtering        • Rate Limiting                       │      │
│  │  • TLS Termination     • Geographic Filtering                │      │
│  └──────────────────────────┬───────────────────────────────────┘      │
└─────────────────────────────┼───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     AUTHENTICATION LAYER                                 │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  Authentication Middleware                                   │      │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐         │      │
│  │  │  API Key    │  │     JWT     │  │  mTLS Cert   │         │      │
│  │  │ Validation  │  │ Validation  │  │  Validation  │         │      │
│  │  └─────────────┘  └─────────────┘  └──────────────┘         │      │
│  └──────────────────────────┬───────────────────────────────────┘      │
└─────────────────────────────┼───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     AUTHORIZATION LAYER                                  │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  RBAC Authorization Engine                                   │      │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │      │
│  │  │     User     │  │    Admin     │  │   ReadOnly   │       │      │
│  │  │     Role     │  │     Role     │  │     Role     │       │      │
│  │  └──────────────┘  └──────────────┘  └──────────────┘       │      │
│  └──────────────────────────┬───────────────────────────────────┘      │
└─────────────────────────────┼───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     INPUT VALIDATION LAYER                               │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  • Schema Validation    • PII Detection                      │      │
│  │  • Size Limits         • Injection Prevention                │      │
│  │  • Format Validation   • Sanitization                        │      │
│  └──────────────────────────┬───────────────────────────────────┘      │
└─────────────────────────────┼───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                                    │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  LLM-Simulator Core Services                                 │      │
│  │  • Simulation Engine    • Error Injection                    │      │
│  │  • Config Manager       • Latency Model                      │      │
│  └──────────────────────────┬───────────────────────────────────┘      │
└─────────────────────────────┼───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     AUDIT & MONITORING LAYER                             │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  • Comprehensive Audit Logging                               │      │
│  │  • Tamper-Evident Log Chain                                  │      │
│  │  • Security Event Detection                                  │      │
│  │  • Compliance Reporting                                      │      │
│  └──────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Authentication Flow Diagrams

### 2.1 API Key Authentication Flow (Primary Method)

```
┌──────────┐                ┌──────────┐                ┌──────────┐                ┌──────────┐
│  Client  │                │  Axum    │                │   Auth   │                │  Audit   │
│          │                │ Gateway  │                │ Manager  │                │  Logger  │
└─────┬────┘                └─────┬────┘                └─────┬────┘                └─────┬────┘
      │                           │                           │                           │
      │  POST /v1/chat           │                           │                           │
      │  Authorization: Bearer    │                           │                           │
      │  sk-abc123...            │                           │                           │
      ├──────────────────────────>│                           │                           │
      │                           │                           │                           │
      │                           │  extract_credentials()    │                           │
      │                           ├──────────────────────────>│                           │
      │                           │                           │                           │
      │                           │                           │  validate_key_format()   │
      │                           │                           ├──────────────────┐        │
      │                           │                           │                  │        │
      │                           │                           │<─────────────────┘        │
      │                           │                           │                           │
      │                           │                           │  lookup_key_metadata()   │
      │                           │                           ├──────────────────┐        │
      │                           │                           │                  │        │
      │                           │                           │<─────────────────┘        │
      │                           │                           │                           │
      │                           │                           │  check_key_status()      │
      │                           │                           ├──────────────────┐        │
      │                           │                           │                  │        │
      │                           │                           │<─────────────────┘        │
      │                           │                           │                           │
      │                           │                           │  check_ip_allowlist()    │
      │                           │                           ├──────────────────┐        │
      │                           │                           │                  │        │
      │                           │                           │<─────────────────┘        │
      │                           │                           │                           │
      │                           │  AuthContext              │                           │
      │                           │<──────────────────────────┤                           │
      │                           │                           │                           │
      │                           │                           │  log_auth_event()        │
      │                           │                           ├──────────────────────────>│
      │                           │                           │                           │
      │                           │  process_request()        │                           │
      │                           ├──────────────────────────────────────────┐            │
      │                           │                           │              │            │
      │                           │<──────────────────────────────────────────┘            │
      │                           │                           │                           │
      │  200 OK                   │                           │                           │
      │<──────────────────────────┤                           │                           │
      │                           │                           │                           │
```

### 2.2 mTLS Authentication Flow

```
┌──────────┐         ┌──────────┐         ┌──────────┐         ┌──────────┐
│  Client  │         │   TLS    │         │   Auth   │         │  Audit   │
│ (w/cert) │         │Terminator│         │ Manager  │         │  Logger  │
└─────┬────┘         └─────┬────┘         └─────┬────┘         └─────┬────┘
      │                    │                    │                    │
      │  ClientHello       │                    │                    │
      │  + Client Cert     │                    │                    │
      ├───────────────────>│                    │                    │
      │                    │                    │                    │
      │                    │  validate_cert()   │                    │
      │                    ├────────────────────┐                    │
      │                    │  • Check signature │                    │
      │                    │  • Check expiry    │                    │
      │                    │  • Check revocation│                    │
      │                    │  • Verify CA chain │                    │
      │                    │<────────────────────┘                    │
      │                    │                    │                    │
      │  ServerHello       │                    │                    │
      │  + Session Keys    │                    │                    │
      │<───────────────────┤                    │                    │
      │                    │                    │                    │
      │  Application Data  │                    │                    │
      │  POST /v1/chat     │                    │                    │
      ├───────────────────>│                    │                    │
      │                    │                    │                    │
      │                    │  extract_cert_dn() │                    │
      │                    ├───────────────────>│                    │
      │                    │                    │                    │
      │                    │                    │  lookup_principal()│
      │                    │                    ├────────────────┐   │
      │                    │                    │                │   │
      │                    │                    │<────────────────┘   │
      │                    │                    │                    │
      │                    │  AuthContext       │                    │
      │                    │<───────────────────┤                    │
      │                    │                    │                    │
      │                    │                    │  log_mtls_auth()   │
      │                    │                    ├───────────────────>│
      │                    │                    │                    │
      │  200 OK            │                    │                    │
      │<───────────────────┤                    │                    │
      │                    │                    │                    │
```

---

## 3. Authorization Decision Flow

### 3.1 RBAC Authorization Flow

```
┌────────────────────────────────────────────────────────────────────┐
│                     Authorization Decision Tree                    │
└────────────────────────────────────────────────────────────────────┘

                         Request Received
                                │
                                │
                                ▼
                     ┌──────────────────────┐
                     │ Extract AuthContext  │
                     │ • Principal          │
                     │ • Role               │
                     │ • Session            │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ Identify Resource    │
                     │ • Endpoint path      │
                     │ • HTTP method        │
                     │ • Resource ID        │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ Determine Action     │
                     │ • read / write       │
                     │ • execute / delete   │
                     │ • admin              │
                     └──────────┬───────────┘
                                │
                                ▼
                  ┌─────────────────────────┐
                  │  Lookup Role Permissions│
                  └──────────┬──────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │  Role has Permission?        │
              └──────────┬───────────────────┘
                         │
           ┌─────────────┴─────────────┐
           │                           │
          YES                         NO
           │                           │
           ▼                           ▼
    ┌──────────────┐         ┌─────────────────┐
    │   GRANTED    │         │     DENIED      │
    │              │         │                 │
    │ • Allow      │         │ • Return 403    │
    │   request    │         │ • Log denial    │
    │ • Log grant  │         │ • Alert (if     │
    │              │         │   admin API)    │
    └──────────────┘         └─────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  Authorization Matrix                           │
├─────────────┬──────────────────────────────────────────────────┤
│   Resource  │  User │ Admin │ ReadOnly │ System │              │
├─────────────┼───────┼───────┼──────────┼────────┼──────────────┤
│ /v1/chat    │  ✅   │  ✅   │    ❌    │   ✅   │ Execute      │
│ /v1/embed   │  ✅   │  ✅   │    ❌    │   ✅   │ Execute      │
│ /health     │  ✅   │  ✅   │    ✅    │   ✅   │ Read         │
│ /metrics    │  ❌   │  ✅   │    ✅    │   ✅   │ Read         │
│ /admin/cfg  │  ❌   │  ✅   │    ❌    │   ❌   │ Write        │
│ /admin/scn  │  ❌   │  ✅   │    ❌    │   ✅   │ Execute      │
└─────────────┴───────┴───────┴──────────┴────────┴──────────────┘
```

---

## 4. Network Security Topology

### 4.1 Kubernetes Deployment Topology

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        INTERNET                                          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 │ HTTPS (443)
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     CLOUD PROVIDER (AWS/GCP/Azure)                       │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │              Load Balancer / Ingress Controller               │     │
│  │  • TLS Termination (TLS 1.3)                                  │     │
│  │  • SSL Certificates (Let's Encrypt / ACM)                     │     │
│  │  • DDoS Protection                                            │     │
│  │  • WAF Rules                                                  │     │
│  └────────────────────────────┬──────────────────────────────────┘     │
│                                │                                         │
│                                │ HTTP (8080) - Internal                  │
│                                ▼                                         │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │           KUBERNETES CLUSTER (Private VPC)                    │     │
│  │  ┌─────────────────────────────────────────────────────────┐ │     │
│  │  │        Namespace: llm-simulator                        │ │     │
│  │  │  ┌──────────────────────────────────────────────────┐  │ │     │
│  │  │  │  NetworkPolicy: Ingress                         │  │ │     │
│  │  │  │  • Allow from: ingress-nginx namespace          │  │ │     │
│  │  │  │  • Allow from: monitoring namespace             │  │ │     │
│  │  │  │  • Allow to: port 8080 (app)                    │  │ │     │
│  │  │  │  • Allow to: port 9090 (metrics)                │  │ │     │
│  │  │  └──────────────────┬───────────────────────────────┘  │ │     │
│  │  │                     │                                   │ │     │
│  │  │                     ▼                                   │ │     │
│  │  │  ┌──────────────────────────────────────────────────┐  │ │     │
│  │  │  │  Service: llm-simulator-svc                     │  │ │     │
│  │  │  │  • Type: ClusterIP                              │  │ │     │
│  │  │  │  • Port: 8080 → targetPort: 8080                │  │ │     │
│  │  │  │  • Selector: app=llm-simulator                  │  │ │     │
│  │  │  └──────────────────┬───────────────────────────────┘  │ │     │
│  │  │                     │                                   │ │     │
│  │  │                     ▼                                   │ │     │
│  │  │  ┌──────────────────────────────────────────────────┐  │ │     │
│  │  │  │  Pod: llm-simulator                             │  │ │     │
│  │  │  │  ┌────────────────────────────────────────┐     │  │ │     │
│  │  │  │  │  Container: llm-simulator              │     │  │ │     │
│  │  │  │  │  • Image: llm-simulator:1.0.0          │     │  │ │     │
│  │  │  │  │  • Port: 8080                          │     │  │ │     │
│  │  │  │  │  • SecurityContext:                    │     │  │ │     │
│  │  │  │  │    - readOnlyRootFilesystem: true      │     │  │ │     │
│  │  │  │  │    - allowPrivilegeEscalation: false   │     │  │ │     │
│  │  │  │  │    - runAsNonRoot: true                │     │  │ │     │
│  │  │  │  │    - capabilities: drop ALL            │     │  │ │     │
│  │  │  │  │  • Resources:                          │     │  │ │     │
│  │  │  │  │    - requests: cpu=500m, mem=512Mi     │     │  │ │     │
│  │  │  │  │    - limits: cpu=2, mem=2Gi            │     │  │ │     │
│  │  │  │  └────────────────────────────────────────┘     │  │ │     │
│  │  │  │  ┌────────────────────────────────────────┐     │  │ │     │
│  │  │  │  │  Volumes                               │     │  │ │     │
│  │  │  │  │  • config-vol (ConfigMap)              │     │  │ │     │
│  │  │  │  │  • secrets-vol (Secret/CSI)            │     │  │ │     │
│  │  │  │  │  • tmp-vol (emptyDir)                  │     │  │ │     │
│  │  │  │  └────────────────────────────────────────┘     │  │ │     │
│  │  │  └──────────────────────────────────────────────────┘  │ │     │
│  │  └─────────────────────────────────────────────────────────┘ │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │     Supporting Services (Separate Namespaces)                 │     │
│  │  • Prometheus (monitoring)                                    │     │
│  │  • Grafana (visualization)                                    │     │
│  │  • Fluentd (log aggregation)                                  │     │
│  │  • Secret Manager CSI Driver                                  │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

External Dependencies:
┌─────────────────────────────────────────────────────────────────────────┐
│  • AWS Secrets Manager / GCP Secret Manager (HTTPS)                     │
│  • CloudWatch / Stackdriver Logging (HTTPS)                             │
│  • External SIEM (Splunk/Elasticsearch) (HTTPS)                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Network Segmentation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Network Segmentation Model                           │
└─────────────────────────────────────────────────────────────────────────┘

ZONE 1: Public / DMZ (Internet-Facing)
┌──────────────────────────────────────────────────┐
│  • Load Balancer                                 │
│  • WAF                                           │
│  • DDoS Protection                               │
│  Security Level: HIGH                            │
│  Allowed Egress: Zone 2 only                     │
└────────────────┬─────────────────────────────────┘
                 │ Port 8080 (HTTP)
                 │ Firewall: Stateful inspection
                 ▼
ZONE 2: Application Tier (Private Subnet)
┌──────────────────────────────────────────────────┐
│  • LLM-Simulator Pods                            │
│  • Service Mesh (optional)                       │
│  Security Level: MEDIUM-HIGH                     │
│  Allowed Egress:                                 │
│    - Zone 3 (secrets, monitoring)                │
│    - Internet (secret manager APIs)              │
│  NetworkPolicy: Strict ingress/egress rules      │
└────────────────┬─────────────────────────────────┘
                 │ Port 443 (HTTPS)
                 │ Firewall: Strict allow-list
                 ▼
ZONE 3: Infrastructure Services (Isolated)
┌──────────────────────────────────────────────────┐
│  • Prometheus (monitoring)                       │
│  • Secret Manager CSI                            │
│  • Audit Log Aggregator                          │
│  Security Level: HIGH                            │
│  Allowed Egress: Internet (for alerts)           │
│  NetworkPolicy: Minimal access                   │
└──────────────────────────────────────────────────┘
```

---

## 5. Data Flow Diagrams

### 5.1 Secure Request/Response Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Secure Request/Response Data Flow                           │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────┐                                                    ┌──────────┐
│  Client  │                                                    │  Audit   │
│          │                                                    │   Log    │
└─────┬────┘                                                    └─────▲────┘
      │                                                               │
      │ 1. HTTPS Request                                              │
      │    (TLS 1.3 Encrypted)                                        │
      │    Headers: Authorization, Content-Type                       │
      │    Body: JSON Payload                                         │
      ├──────────────────────────────────────────────┐                │
      │                                              │                │
      ▼                                              │                │
┌─────────────────────────────────────────┐         │                │
│  2. TLS Termination                     │         │                │
│     • Decrypt TLS                       │         │                │
│     • Validate Certificate              │         │                │
│     • Extract HTTP Request              │         │                │
└─────┬───────────────────────────────────┘         │                │
      │                                              │                │
      │ 3. Plain HTTP (Internal)                    │                │
      ▼                                              │                │
┌─────────────────────────────────────────┐         │                │
│  4. Authentication Middleware           │         │                │
│     • Extract Bearer Token              │         │                │
│     • Redact Key (log prefix only) ───────────────┼───────────────>│
│     • Validate Format                   │         │                │
│     • Lookup Metadata                   │         │                │
└─────┬───────────────────────────────────┘         │                │
      │                                              │                │
      │ 5. AuthContext                               │                │
      ▼                                              │                │
┌─────────────────────────────────────────┐         │                │
│  6. Authorization Middleware            │         │                │
│     • Check Role Permissions            │         │                │
│     • Log Authorization Decision ──────────────────┼───────────────>│
└─────┬───────────────────────────────────┘         │                │
      │                                              │                │
      │ 7. Authorized Request                        │                │
      ▼                                              │                │
┌─────────────────────────────────────────┐         │                │
│  8. Input Validation Middleware         │         │                │
│     • Schema Validation                 │         │                │
│     • PII Detection & Redaction         │         │                │
│     • Size/Length Checks                │         │                │
│     • Sanitization                      │         │                │
└─────┬───────────────────────────────────┘         │                │
      │                                              │                │
      │ 9. Sanitized Request                         │                │
      ▼                                              │                │
┌─────────────────────────────────────────┐         │                │
│  10. Application Handler                │         │                │
│      • Process Request                  │         │                │
│      • Generate Response                │         │                │
│      • Log Processing ─────────────────────────────┼───────────────>│
└─────┬───────────────────────────────────┘         │                │
      │                                              │                │
      │ 11. JSON Response                            │                │
      ▼                                              │                │
┌─────────────────────────────────────────┐         │                │
│  12. Security Headers Middleware        │         │                │
│      • Add HSTS Header                  │         │                │
│      • Add CSP Header                   │         │                │
│      • Add X-Frame-Options              │         │                │
│      • Add X-Content-Type-Options       │         │                │
└─────┬───────────────────────────────────┘         │                │
      │                                              │                │
      │ 13. Response with Security Headers           │                │
      ▼                                              │                │
┌─────────────────────────────────────────┐         │                │
│  14. TLS Encryption                     │         │                │
│      • Encrypt Response                 │         │                │
│      • Send to Client                   │         │                │
└─────┬───────────────────────────────────┘         │                │
      │                                              │                │
      │ 15. HTTPS Response ◄──────────────────────────┘                │
      │     (TLS 1.3 Encrypted)                                        │
      ▼                                                                │
   Client                                                              │
                                                                       │
   16. Audit Log Written ◄────────────────────────────────────────────┘
       • Request/Response logged
       • Signed with HMAC
       • Added to tamper-evident chain
```

### 5.2 Secret Retrieval Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Secret Retrieval Data Flow                            │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│ LLM-Simulator│         │Secret Manager│         │Cloud Provider│
│     Pod      │         │     (CSI)    │         │  Secret Store│
└──────┬───────┘         └──────┬───────┘         └──────┬───────┘
       │                        │                        │
       │ 1. Pod Startup         │                        │
       ├───────────────────────>│                        │
       │                        │                        │
       │                        │ 2. Mount Request       │
       │                        │   (IAM Role/SA)        │
       │                        ├───────────────────────>│
       │                        │                        │
       │                        │ 3. Authenticate        │
       │                        │    • Verify IAM Role   │
       │                        │    • Check Permissions │
       │                        │<───────────────────────│
       │                        │                        │
       │                        │ 4. Fetch Secrets       │
       │                        │    • admin_api_key     │
       │                        │    • jwt_secret        │
       │                        │    • tls_key           │
       │                        ├───────────────────────>│
       │                        │                        │
       │                        │ 5. Encrypted Secrets   │
       │                        │<───────────────────────│
       │                        │                        │
       │                        │ 6. Decrypt (KMS)       │
       │                        ├────────────────┐       │
       │                        │                │       │
       │                        │<────────────────┘       │
       │                        │                        │
       │ 7. Mount as Volume     │                        │
       │    /run/secrets/       │                        │
       │<───────────────────────│                        │
       │                        │                        │
       │ 8. Read Secrets        │                        │
       │    (In-Memory)         │                        │
       ├────────────────┐       │                        │
       │                │       │                        │
       │<────────────────┘       │                        │
       │                        │                        │
       │ 9. Zeroize on Reload   │                        │
       │    (Memory safety)     │                        │
       │                        │                        │

   ┌───────────────────────────────────────────────────────┐
   │  Secret Lifecycle                                     │
   ├───────────────────────────────────────────────────────┤
   │  1. Creation:    Cloud KMS encrypted                  │
   │  2. Storage:     Secret Manager (versioned)           │
   │  3. Retrieval:   IAM authenticated, KMS decrypted     │
   │  4. In-Transit:  TLS 1.3 encrypted                    │
   │  5. In-Memory:   Zeroized on drop (Rust SecretString) │
   │  6. In-Use:      Never logged or exposed             │
   │  7. Rotation:    Automated every 90 days              │
   └───────────────────────────────────────────────────────┘
```

---

## 6. Threat Model Diagrams

### 6.1 STRIDE Threat Model

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         STRIDE Threat Model                              │
└─────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│ S - Spoofing Identity                                                 │
├───────────────────────────────────────────────────────────────────────┤
│ Threat: Attacker impersonates legitimate user with forged API key    │
│                                                                       │
│   Attacker ──┬─> Forged Key    ──X──> Validation ──> REJECTED       │
│              │   "sk-fake..."                                        │
│              │                                                       │
│              └─> Valid Format   ──✓──> Lookup     ──> NOT FOUND     │
│                  "sk-abc123..."                                      │
│                                                                       │
│ Mitigation:                                                           │
│  ✓ Format validation (regex)                                         │
│  ✓ Prefix verification                                               │
│  ✓ Metadata lookup (simulated)                                       │
│  ✓ Rate limiting on auth attempts                                    │
│  ✓ Brute force protection                                            │
└───────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│ T - Tampering with Data                                               │
├───────────────────────────────────────────────────────────────────────┤
│ Threat: Attacker modifies audit logs to hide malicious activity      │
│                                                                       │
│   Audit Log Chain:                                                    │
│   ┌──────┐     ┌──────┐     ┌──────┐     ┌──────┐                  │
│   │Event1├────>│Event2├────>│Event3├────>│Event4│                  │
│   │Hash1 │     │Hash2 │     │Hash3 │     │Hash4 │                  │
│   └──┬───┘     └──┬───┘     └──┬───┘     └──┬───┘                  │
│      │            │            │            │                       │
│      │            X TAMPER     │            │                       │
│      │            │            │            │                       │
│      └────────────┴────────────┴────────────┘                       │
│                   │                                                  │
│                   ▼                                                  │
│            Integrity Check FAILS                                     │
│            (Hash chain broken)                                       │
│                                                                       │
│ Mitigation:                                                           │
│  ✓ Hash chaining (each event includes previous hash)                 │
│  ✓ HMAC signatures on each event                                     │
│  ✓ Write-once log storage                                            │
│  ✓ External log shipping (SIEM)                                      │
└───────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│ R - Repudiation                                                       │
├───────────────────────────────────────────────────────────────────────┤
│ Threat: User denies making API requests                              │
│                                                                       │
│   User: "I never made those requests!"                               │
│                                                                       │
│   Audit Log Evidence:                                                 │
│   {                                                                   │
│     "event_id": "uuid-123",                                           │
│     "timestamp": "2025-11-26T12:34:56Z",                              │
│     "actor": {                                                        │
│       "id": "sk-abc123...",                                           │
│       "role": "User"                                                  │
│     },                                                                │
│     "resource": "/v1/chat/completions",                               │
│     "source_ip": "192.168.1.100",                                     │
│     "request_id": "req-789",                                          │
│     "signature": "hmac-sha256:abcdef..."                              │
│   }                                                                   │
│                                                                       │
│ Mitigation:                                                           │
│  ✓ Comprehensive audit logging (all API calls)                       │
│  ✓ Signed audit events (non-repudiation)                             │
│  ✓ Immutable log storage                                             │
│  ✓ IP address + user agent logging                                   │
└───────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│ I - Information Disclosure                                            │
├───────────────────────────────────────────────────────────────────────┤
│ Threat: API keys leaked in logs                                      │
│                                                                       │
│   ❌ BEFORE (Vulnerable):                                             │
│   log.info("Auth with key: sk-abc123def456ghi789...")                │
│                                                                       │
│   ✓ AFTER (Secure):                                                  │
│   log.info("Auth with key: sk-abc123...")                            │
│                           ^^^^^^^^                                    │
│                           Prefix only (8 chars)                       │
│                                                                       │
│ Mitigation:                                                           │
│  ✓ Automatic key redaction in logs                                   │
│  ✓ PII detection and masking                                         │
│  ✓ TLS encryption in transit                                         │
│  ✓ No sensitive data storage                                         │
│  ✓ Secure error messages (no stack traces)                           │
└───────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│ D - Denial of Service                                                 │
├───────────────────────────────────────────────────────────────────────┤
│ Threat: Request flooding overwhelms system                            │
│                                                                       │
│   Attacker ──> 10,000 req/s ──┐                                      │
│                               │                                       │
│                               ▼                                       │
│                    ┌───────────────────┐                              │
│                    │  Rate Limiter     │                              │
│                    │  (1000 req/min)   │                              │
│                    └─────┬─────────────┘                              │
│                          │                                            │
│              ┌───────────┴───────────┐                                │
│              │                       │                                │
│           ALLOW (1000)          REJECT (9000)                         │
│              │                       │                                │
│              ▼                       ▼                                │
│        Process Request         Return 429                             │
│                                + Retry-After                          │
│                                                                       │
│ Mitigation:                                                           │
│  ✓ Multi-layer rate limiting (IP, API key, endpoint)                 │
│  ✓ Concurrency limits (semaphore)                                    │
│  ✓ Request queue with backpressure                                   │
│  ✓ Graceful degradation                                              │
│  ✓ DDoS protection (CloudFlare/AWS Shield)                           │
└───────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│ E - Elevation of Privilege                                            │
├───────────────────────────────────────────────────────────────────────┤
│ Threat: User attempts to access admin API                            │
│                                                                       │
│   User (Role: User) ──> POST /admin/config                           │
│                                                                       │
│                         ▼                                             │
│                  ┌──────────────┐                                     │
│                  │Authorization │                                     │
│                  │    Engine    │                                     │
│                  └──────┬───────┘                                     │
│                         │                                             │
│           Check: Does "User" role have                                │
│                  "write" permission on                                │
│                  "admin/config" resource?                             │
│                         │                                             │
│                         ▼                                             │
│                     ┌───────┐                                         │
│                     │  NO   │                                         │
│                     └───┬───┘                                         │
│                         │                                             │
│                         ▼                                             │
│                  Return 403 Forbidden                                 │
│                  + Audit Log (AuthzDenied)                            │
│                  + Alert (if suspicious)                              │
│                                                                       │
│ Mitigation:                                                           │
│  ✓ RBAC enforcement at every endpoint                                │
│  ✓ Separate admin API key                                            │
│  ✓ Principle of least privilege                                      │
│  ✓ Audit all authorization failures                                  │
│  ✓ Alert on repeated admin access attempts                           │
└───────────────────────────────────────────────────────────────────────┘
```

### 6.2 Attack Tree: Unauthorized Admin Access

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Attack Tree: Unauthorized Admin Access               │
└─────────────────────────────────────────────────────────────────────────┘

                      ┌──────────────────────────┐
                      │  Gain Admin Access to    │
                      │    LLM-Simulator         │
                      │  (Change Configuration)  │
                      └───────────┬──────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                │                                   │
                ▼                                   ▼
    ┌───────────────────────┐         ┌───────────────────────┐
    │  Steal Admin API Key  │   OR    │ Exploit Vulnerability │
    └───────────┬───────────┘         └───────────┬───────────┘
                │                                  │
       ┌────────┴────────┐                ┌───────┴───────┐
       │                 │                │               │
       ▼                 ▼                ▼               ▼
┌──────────────┐  ┌─────────────┐  ┌──────────┐  ┌──────────────┐
│ Social       │  │ Log File    │  │ Authz    │  │ Inject       │
│ Engineering  │  │ Scraping    │  │ Bypass   │  │ Malicious    │
│              │  │             │  │          │  │ Config       │
│ Likelihood:  │  │ Likelihood: │  │ Likelihood│  │ Likelihood:  │
│   Medium     │  │   Low       │  │   Low    │  │   Very Low   │
│              │  │             │  │          │  │              │
│ Mitigation:  │  │ Mitigation: │  │ Mitigation│  │ Mitigation:  │
│ • Security   │  │ • Key       │  │ • RBAC   │  │ • Schema     │
│   Training   │  │   Redaction │  │   Tests  │  │   Validation │
│ • Key        │  │ • Audit Log │  │ • Code   │  │ • Input      │
│   Rotation   │  │   Signing   │  │   Review │  │   Sanitize   │
└──────────────┘  └─────────────┘  └──────────┘  └──────────────┘
```

---

## Conclusion

These diagrams provide visual representations of the security architecture described in SECURITY_ARCHITECTURE.md. They illustrate:

1. **Defense in Depth**: Multiple security layers from network to application
2. **Zero Trust**: Authentication and authorization at every request
3. **Secure Data Flow**: TLS encryption, sanitization, redaction
4. **Threat Mitigation**: STRIDE-based threat model with countermeasures
5. **Network Segmentation**: Isolated zones with strict firewall rules
6. **Audit Trail**: Tamper-evident logging with hash chaining

For implementation details, refer to the main security architecture document.

---

**Document Control**:
- **Version**: 1.0.0
- **Last Updated**: 2025-11-26
- **Companion Document**: SECURITY_ARCHITECTURE.md
