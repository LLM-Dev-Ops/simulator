# Security and Compliance Architecture
# LLM-Simulator - Enterprise-Grade Security Design

> **Classification**: Internal - Security Architecture Documentation
> **Version**: 1.0.0
> **Date**: 2025-11-26
> **Author**: Principal Security Architect
> **Status**: Production-Ready Design

---

## Executive Summary

This document defines the comprehensive security architecture for LLM-Simulator, an enterprise-grade system that simulates LLM provider APIs for testing, development, and compliance validation. The architecture implements defense-in-depth principles, zero-trust security model, and compliance-ready audit capabilities suitable for regulated environments including SOC2, HIPAA, and PCI-DSS contexts.

**Key Security Objectives**:
- Authentication simulation without storing real credentials
- Authorization controls for administrative functions
- Network security with TLS/mTLS support
- Comprehensive audit logging for compliance
- Secret management integration
- Supply chain security
- Runtime security and threat detection

---

## Table of Contents

1. [Security Architecture Overview](#1-security-architecture-overview)
2. [Authentication and Authorization](#2-authentication-and-authorization)
3. [Network Security](#3-network-security)
4. [Data Protection](#4-data-protection)
5. [Audit and Compliance](#5-audit-and-compliance)
6. [Secret Management](#6-secret-management)
7. [Supply Chain Security](#7-supply-chain-security)
8. [Threat Model](#8-threat-model)
9. [Security Configuration Reference](#9-security-configuration-reference)
10. [Compliance Control Mappings](#10-compliance-control-mappings)
11. [Incident Response](#11-incident-response)
12. [Security Testing](#12-security-testing)

---

## 1. Security Architecture Overview

### 1.1 Security Principles

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM-Simulator Security Architecture              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │                    External Clients                         │    │
│  │  • Development Teams  • CI/CD Systems  • Test Harnesses    │    │
│  └───────────────────┬────────────────────────────────────────┘    │
│                      │ HTTPS/TLS 1.3                                │
│                      │ mTLS (Optional)                              │
│                      ▼                                               │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │            Security Perimeter (Ingress)                      │  │
│  ├──────────────────────────────────────────────────────────────┤  │
│  │  • TLS Termination         • Rate Limiting                  │  │
│  │  • Certificate Validation  • DDoS Protection                │  │
│  │  • IP Allowlisting         • Request Signing Validation     │  │
│  └───────────────────┬──────────────────────────────────────────┘  │
│                      │                                               │
│                      ▼                                               │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │         Application Layer Security (Middleware)              │  │
│  ├──────────────────────────────────────────────────────────────┤  │
│  │  Layer 1: Request Tracing & Correlation                     │  │
│  │  Layer 2: Authentication (API Key Simulation)               │  │
│  │  Layer 3: Authorization (RBAC for Admin API)                │  │
│  │  Layer 4: Rate Limiting & Throttling                        │  │
│  │  Layer 5: Input Validation & Sanitization                   │  │
│  │  Layer 6: Audit Logging                                     │  │
│  │  Layer 7: Security Headers                                  │  │
│  └───────────────────┬──────────────────────────────────────────┘  │
│                      │                                               │
│                      ▼                                               │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Core Application Services                       │  │
│  ├──────────────────────────────────────────────────────────────┤  │
│  │  • Simulation Engine     • Error Injection                  │  │
│  │  • Config Manager        • Metrics Collector                │  │
│  │  • Session Management    • Latency Model                    │  │
│  └───────────────────┬──────────────────────────────────────────┘  │
│                      │                                               │
│                      ▼                                               │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │          Security Monitoring & Observability                 │  │
│  ├──────────────────────────────────────────────────────────────┤  │
│  │  • Security Event Logging    • Anomaly Detection            │  │
│  │  • Audit Trail Generation    • Threat Intelligence          │  │
│  │  • Compliance Reporting      • SIEM Integration             │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Defense-in-Depth Layers

| Layer | Security Control | Implementation |
|-------|-----------------|----------------|
| **L1: Network** | TLS/mTLS, Firewalls, Network Policies | TLS 1.3, Optional mTLS, K8s NetworkPolicies |
| **L2: Perimeter** | Rate Limiting, DDoS Protection | Token bucket algorithm, IP-based limits |
| **L3: Authentication** | API Key Simulation, Token Validation | Bearer token format validation, key prefix verification |
| **L4: Authorization** | RBAC, Endpoint ACLs | Admin API key separation, role-based access |
| **L5: Application** | Input Validation, Output Encoding | Schema validation, JSON sanitization |
| **L6: Data** | Encryption at Rest/Transit | TLS for transit, no PII storage by design |
| **L7: Audit** | Comprehensive Logging | Structured audit logs, tamper-evident logging |

### 1.3 Security Design Principles

1. **Zero Trust**: Never trust, always verify - all requests authenticated and authorized
2. **Least Privilege**: Minimal permissions required for each operation
3. **Defense in Depth**: Multiple layers of security controls
4. **Fail Secure**: System defaults to deny on error conditions
5. **Separation of Duties**: Admin vs. user API separation
6. **Audit Everything**: Comprehensive logging for compliance
7. **Secure by Default**: Security features enabled in default configuration
8. **Minimal Attack Surface**: Only essential endpoints exposed

---

## 2. Authentication and Authorization

### 2.1 Authentication Architecture

```rust
// File: src/security/authentication.rs

use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};

/// Authentication module for API key simulation and validation
pub struct AuthenticationManager {
    /// Simulated API key registry (prefix-based validation)
    key_registry: Arc<RwLock<ApiKeyRegistry>>,

    /// JWT validator for token-based auth
    jwt_validator: Arc<JwtValidator>,

    /// mTLS certificate validator
    mtls_validator: Option<Arc<MtlsValidator>>,

    /// Authentication audit logger
    audit_logger: Arc<AuthAuditLogger>,

    /// Configuration
    config: AuthConfig,
}

/// Configuration for authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Enable API key authentication
    pub enable_api_key_auth: bool,

    /// Enable JWT authentication
    pub enable_jwt_auth: bool,

    /// Enable mTLS authentication
    pub enable_mtls_auth: bool,

    /// Require authentication for all endpoints
    pub require_auth: bool,

    /// Admin API key (separate from user keys)
    pub admin_api_key: Option<SecretString>,

    /// JWT signing secret
    pub jwt_secret: Option<SecretString>,

    /// JWT token expiration (seconds)
    pub jwt_expiration: u64,

    /// API key format validation regex
    pub key_format_regex: String,

    /// Maximum concurrent sessions per key
    pub max_sessions_per_key: Option<usize>,

    /// Failed auth attempt threshold before lockout
    pub max_failed_attempts: u32,

    /// Lockout duration (seconds)
    pub lockout_duration: u64,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            enable_api_key_auth: true,
            enable_jwt_auth: false,
            enable_mtls_auth: false,
            require_auth: true,
            admin_api_key: None,
            jwt_secret: None,
            jwt_expiration: 3600,
            key_format_regex: "^sk-[a-zA-Z0-9]{32,}$".to_string(),
            max_sessions_per_key: Some(100),
            max_failed_attempts: 5,
            lockout_duration: 900, // 15 minutes
        }
    }
}

/// API Key registry for simulation
pub struct ApiKeyRegistry {
    /// Simulated keys with metadata (no actual credential storage)
    keys: HashMap<KeyId, ApiKeyMetadata>,

    /// Key prefix to ID mapping
    prefix_map: HashMap<String, KeyId>,

    /// Failed attempt tracking
    failed_attempts: HashMap<String, FailedAttemptTracker>,
}

#[derive(Debug, Clone)]
pub struct ApiKeyMetadata {
    /// Unique key identifier
    pub id: KeyId,

    /// Key prefix (first 8 chars) for logging
    pub prefix: String,

    /// Associated role
    pub role: Role,

    /// Key status
    pub status: KeyStatus,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Last used timestamp
    pub last_used: Option<DateTime<Utc>>,

    /// Rate limit tier
    pub rate_limit_tier: RateLimitTier,

    /// Allowed endpoints
    pub allowed_endpoints: Vec<String>,

    /// IP allowlist (if configured)
    pub ip_allowlist: Option<Vec<String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyStatus {
    Active,
    Suspended,
    Revoked,
    Expired,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Regular API user
    User,

    /// Administrator with full access
    Admin,

    /// Read-only monitoring access
    ReadOnly,

    /// CI/CD system access
    System,
}

impl AuthenticationManager {
    pub fn new(config: AuthConfig) -> Self {
        Self {
            key_registry: Arc::new(RwLock::new(ApiKeyRegistry::new())),
            jwt_validator: Arc::new(JwtValidator::new(config.jwt_secret.clone())),
            mtls_validator: config.enable_mtls_auth.then(|| Arc::new(MtlsValidator::new())),
            audit_logger: Arc::new(AuthAuditLogger::new()),
            config,
        }
    }

    /// Authenticate request using multiple methods
    pub async fn authenticate(
        &self,
        request: &HttpRequest,
    ) -> AuthResult<AuthContext> {
        let start = Instant::now();

        // Extract authentication credentials
        let credentials = self.extract_credentials(request)?;

        // Check for brute force protection
        self.check_rate_limiting(&credentials).await?;

        // Authenticate based on credential type
        let auth_context = match credentials {
            Credentials::ApiKey(key) => {
                self.authenticate_api_key(&key, request).await?
            }
            Credentials::Jwt(token) => {
                self.authenticate_jwt(&token, request).await?
            }
            Credentials::MtlsCert(cert) => {
                self.authenticate_mtls(&cert, request).await?
            }
            Credentials::None => {
                if self.config.require_auth {
                    return Err(AuthError::MissingCredentials);
                }
                AuthContext::anonymous()
            }
        };

        // Log authentication event
        self.audit_logger.log_auth_event(AuditEvent {
            event_type: AuditEventType::AuthenticationSuccess,
            principal: auth_context.principal.clone(),
            request_id: request.id,
            timestamp: Utc::now(),
            ip_address: request.client_ip.clone(),
            user_agent: request.user_agent.clone(),
            resource: request.uri.clone(),
            result: "success".to_string(),
            duration_ms: start.elapsed().as_millis() as u64,
        }).await;

        Ok(auth_context)
    }

    /// Authenticate API key (simulation mode)
    async fn authenticate_api_key(
        &self,
        key: &str,
        request: &HttpRequest,
    ) -> AuthResult<AuthContext> {
        // Validate key format
        if !self.validate_key_format(key) {
            self.record_failed_attempt(key, request).await;
            return Err(AuthError::InvalidKeyFormat);
        }

        // Extract key prefix
        let prefix = self.extract_key_prefix(key);

        // In simulation mode, we validate format but don't store actual keys
        // Real implementation would hash and compare against stored hash

        let registry = self.key_registry.read().await;

        // Lookup by prefix for simulation
        let key_id = registry.prefix_map.get(&prefix)
            .ok_or_else(|| {
                self.record_failed_attempt(key, request).await;
                AuthError::InvalidApiKey
            })?;

        let metadata = registry.keys.get(key_id)
            .ok_or(AuthError::InvalidApiKey)?;

        // Check key status
        match metadata.status {
            KeyStatus::Active => {},
            KeyStatus::Suspended => return Err(AuthError::KeySuspended),
            KeyStatus::Revoked => return Err(AuthError::KeyRevoked),
            KeyStatus::Expired => return Err(AuthError::KeyExpired),
        }

        // Check IP allowlist if configured
        if let Some(ref allowlist) = metadata.ip_allowlist {
            if !allowlist.contains(&request.client_ip) {
                return Err(AuthError::IpNotAllowed);
            }
        }

        // Update last used timestamp
        drop(registry);
        let mut registry = self.key_registry.write().await;
        if let Some(meta) = registry.keys.get_mut(key_id) {
            meta.last_used = Some(Utc::now());
        }

        Ok(AuthContext {
            principal: Principal::ApiKey(key_id.clone()),
            role: metadata.role,
            authenticated: true,
            authentication_method: AuthMethod::ApiKey,
            session_id: SessionId::new(),
            metadata: HashMap::new(),
        })
    }

    /// Validate API key format
    fn validate_key_format(&self, key: &str) -> bool {
        let regex = regex::Regex::new(&self.config.key_format_regex)
            .expect("Invalid regex pattern");
        regex.is_match(key)
    }

    /// Extract key prefix for logging (never log full key)
    fn extract_key_prefix(&self, key: &str) -> String {
        key.chars().take(8).collect()
    }

    /// Record failed authentication attempt
    async fn record_failed_attempt(&self, key: &str, request: &HttpRequest) {
        let prefix = self.extract_key_prefix(key);

        self.audit_logger.log_auth_event(AuditEvent {
            event_type: AuditEventType::AuthenticationFailure,
            principal: format!("{}...", prefix),
            request_id: request.id,
            timestamp: Utc::now(),
            ip_address: request.client_ip.clone(),
            user_agent: request.user_agent.clone(),
            resource: request.uri.clone(),
            result: "invalid_credentials".to_string(),
            duration_ms: 0,
        }).await;
    }
}

/// Authentication context
#[derive(Debug, Clone)]
pub struct AuthContext {
    /// Authenticated principal
    pub principal: Principal,

    /// Role assigned to principal
    pub role: Role,

    /// Whether authentication was successful
    pub authenticated: bool,

    /// Authentication method used
    pub authentication_method: AuthMethod,

    /// Session identifier
    pub session_id: SessionId,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub enum Principal {
    ApiKey(KeyId),
    JwtSubject(String),
    Certificate(String),
    Anonymous,
}

#[derive(Debug, Clone, Copy)]
pub enum AuthMethod {
    ApiKey,
    Jwt,
    MtlsCertificate,
    None,
}
```

### 2.2 Authorization Model (RBAC)

```rust
// File: src/security/authorization.rs

/// Role-Based Access Control (RBAC) implementation
pub struct AuthorizationManager {
    /// Role definitions
    roles: Arc<RwLock<RoleRegistry>>,

    /// Permission registry
    permissions: Arc<PermissionRegistry>,

    /// Audit logger
    audit_logger: Arc<AuthzAuditLogger>,
}

/// Role definition
#[derive(Debug, Clone)]
pub struct RoleDefinition {
    pub name: String,
    pub permissions: Vec<Permission>,
    pub inherits_from: Vec<String>,
}

/// Permission definition
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Permission {
    /// Resource type (e.g., "config", "scenarios", "metrics")
    pub resource: String,

    /// Action (e.g., "read", "write", "delete", "execute")
    pub action: Action,

    /// Optional resource identifier
    pub resource_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    Read,
    Write,
    Delete,
    Execute,
    Admin,
}

impl AuthorizationManager {
    /// Check if principal has permission for action
    pub async fn authorize(
        &self,
        context: &AuthContext,
        resource: &str,
        action: Action,
    ) -> AuthzResult<()> {
        let start = Instant::now();

        // Get role permissions
        let roles = self.roles.read().await;
        let role_def = roles.get_role(&context.role)
            .ok_or(AuthzError::RoleNotFound)?;

        // Check if role has required permission
        let required_permission = Permission {
            resource: resource.to_string(),
            action,
            resource_id: None,
        };

        let authorized = self.has_permission(&role_def, &required_permission);

        // Log authorization decision
        self.audit_logger.log_authz_event(AuthzAuditEvent {
            event_type: if authorized {
                AuthzEventType::AuthorizationGranted
            } else {
                AuthzEventType::AuthorizationDenied
            },
            principal: context.principal.clone(),
            resource: resource.to_string(),
            action,
            decision: if authorized { "allow" } else { "deny" }.to_string(),
            timestamp: Utc::now(),
            duration_ms: start.elapsed().as_millis() as u64,
        }).await;

        if authorized {
            Ok(())
        } else {
            Err(AuthzError::PermissionDenied)
        }
    }

    fn has_permission(
        &self,
        role: &RoleDefinition,
        required: &Permission,
    ) -> bool {
        // Check direct permissions
        if role.permissions.contains(required) {
            return true;
        }

        // Check for wildcard permissions
        let wildcard = Permission {
            resource: required.resource.clone(),
            action: Action::Admin,
            resource_id: None,
        };

        role.permissions.contains(&wildcard)
    }
}

/// Authorization matrix for LLM-Simulator
pub fn default_authorization_matrix() -> HashMap<Role, Vec<Permission>> {
    let mut matrix = HashMap::new();

    // User role - standard API access
    matrix.insert(Role::User, vec![
        Permission {
            resource: "completions".to_string(),
            action: Action::Execute,
            resource_id: None,
        },
        Permission {
            resource: "chat".to_string(),
            action: Action::Execute,
            resource_id: None,
        },
        Permission {
            resource: "embeddings".to_string(),
            action: Action::Execute,
            resource_id: None,
        },
        Permission {
            resource: "models".to_string(),
            action: Action::Read,
            resource_id: None,
        },
    ]);

    // Admin role - full access
    matrix.insert(Role::Admin, vec![
        Permission {
            resource: "*".to_string(),
            action: Action::Admin,
            resource_id: None,
        },
    ]);

    // ReadOnly role - monitoring only
    matrix.insert(Role::ReadOnly, vec![
        Permission {
            resource: "metrics".to_string(),
            action: Action::Read,
            resource_id: None,
        },
        Permission {
            resource: "health".to_string(),
            action: Action::Read,
            resource_id: None,
        },
        Permission {
            resource: "stats".to_string(),
            action: Action::Read,
            resource_id: None,
        },
    ]);

    // System role - CI/CD access
    matrix.insert(Role::System, vec![
        Permission {
            resource: "scenarios".to_string(),
            action: Action::Execute,
            resource_id: None,
        },
        Permission {
            resource: "config".to_string(),
            action: Action::Read,
            resource_id: None,
        },
    ]);

    matrix
}
```

### 2.3 Authentication Flows

```
┌─────────────────────────────────────────────────────────────────┐
│           API Key Authentication Flow                           │
└─────────────────────────────────────────────────────────────────┘

Client                  Gateway              Auth Manager         Audit
  │                        │                      │                 │
  │  POST /v1/chat        │                      │                 │
  │  Authorization:       │                      │                 │
  │  Bearer sk-abc123     │                      │                 │
  ├───────────────────────>│                      │                 │
  │                        │  validate_format()   │                 │
  │                        ├─────────────────────>│                 │
  │                        │                      │                 │
  │                        │  check_status()      │                 │
  │                        │<─────────────────────│                 │
  │                        │                      │                 │
  │                        │  check_permissions() │                 │
  │                        ├─────────────────────>│                 │
  │                        │                      │                 │
  │                        │  log_auth_event()    │                 │
  │                        │──────────────────────┼────────────────>│
  │                        │                      │                 │
  │  200 OK + Response     │                      │                 │
  │<───────────────────────│                      │                 │
  │                        │                      │                 │


┌─────────────────────────────────────────────────────────────────┐
│           mTLS Authentication Flow                              │
└─────────────────────────────────────────────────────────────────┘

Client                  TLS Terminator       Auth Manager         Audit
  │                        │                      │                 │
  │  TLS Handshake         │                      │                 │
  │  + Client Cert         │                      │                 │
  ├───────────────────────>│                      │                 │
  │                        │                      │                 │
  │                        │  validate_cert()     │                 │
  │                        ├─────────────────────>│                 │
  │                        │                      │                 │
  │                        │  check_revocation()  │                 │
  │                        │<─────────────────────│                 │
  │                        │                      │                 │
  │  TLS Session           │  extract_subject()   │                 │
  │  Established           ├─────────────────────>│                 │
  │<───────────────────────│                      │                 │
  │                        │                      │                 │
  │  POST /v1/chat         │  log_auth_event()    │                 │
  ├───────────────────────>│──────────────────────┼────────────────>│
  │                        │                      │                 │
```

---

## 3. Network Security

### 3.1 Network Security Architecture

```rust
// File: src/security/network.rs

/// Network security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkSecurityConfig {
    /// TLS configuration
    pub tls: TlsConfig,

    /// Network policies
    pub network_policies: NetworkPolicies,

    /// Firewall rules
    pub firewall_rules: Vec<FirewallRule>,

    /// DDoS protection settings
    pub ddos_protection: DdosProtectionConfig,
}

/// TLS/SSL configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    /// Enable TLS
    pub enabled: bool,

    /// Minimum TLS version
    pub min_version: TlsVersion,

    /// Certificate path
    pub cert_path: PathBuf,

    /// Private key path
    pub key_path: PathBuf,

    /// Certificate chain path (optional)
    pub chain_path: Option<PathBuf>,

    /// Cipher suites (secure defaults)
    pub cipher_suites: Vec<String>,

    /// Enable mTLS (mutual TLS)
    pub mtls_enabled: bool,

    /// Client CA certificate path
    pub client_ca_path: Option<PathBuf>,

    /// Require client certificate
    pub require_client_cert: bool,

    /// OCSP stapling
    pub enable_ocsp_stapling: bool,

    /// Session resumption
    pub enable_session_resumption: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TlsVersion {
    Tls12,
    Tls13,
}

impl Default for TlsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_version: TlsVersion::Tls13,
            cert_path: PathBuf::from("/etc/certs/tls.crt"),
            key_path: PathBuf::from("/etc/certs/tls.key"),
            chain_path: None,
            cipher_suites: vec![
                // TLS 1.3 ciphers (recommended)
                "TLS_AES_256_GCM_SHA384".to_string(),
                "TLS_CHACHA20_POLY1305_SHA256".to_string(),
                "TLS_AES_128_GCM_SHA256".to_string(),
            ],
            mtls_enabled: false,
            client_ca_path: None,
            require_client_cert: false,
            enable_ocsp_stapling: true,
            enable_session_resumption: true,
        }
    }
}

/// Network policies for Kubernetes deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPolicies {
    /// Ingress rules
    pub ingress: Vec<IngressRule>,

    /// Egress rules
    pub egress: Vec<EgressRule>,

    /// Default deny policy
    pub default_deny: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngressRule {
    /// Source CIDR blocks
    pub from_cidrs: Vec<String>,

    /// Source namespaces (K8s)
    pub from_namespaces: Vec<String>,

    /// Allowed ports
    pub ports: Vec<u16>,

    /// Protocol
    pub protocol: Protocol,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EgressRule {
    /// Destination CIDR blocks
    pub to_cidrs: Vec<String>,

    /// Allowed ports
    pub ports: Vec<u16>,

    /// Protocol
    pub protocol: Protocol,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Protocol {
    TCP,
    UDP,
    ICMP,
}
```

### 3.2 Network Security Topology

```yaml
# Kubernetes NetworkPolicy for LLM-Simulator
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: llm-simulator-network-policy
  namespace: llm-simulator
spec:
  podSelector:
    matchLabels:
      app: llm-simulator
  policyTypes:
    - Ingress
    - Egress

  # Ingress Rules
  ingress:
    # Allow from ingress controller only
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
          port: 9090  # Metrics endpoint

  # Egress Rules
  egress:
    # DNS resolution
    - to:
        - namespaceSelector:
            matchLabels:
              name: kube-system
      ports:
        - protocol: UDP
          port: 53

    # Secret manager access (if using external secrets)
    - to:
        - namespaceSelector: {}
      ports:
        - protocol: TCP
          port: 443

    # Deny all other egress
```

### 3.3 TLS Configuration

```nginx
# Example Nginx TLS termination configuration
upstream llm_simulator {
    server llm-simulator:8080;
    keepalive 32;
}

server {
    listen 443 ssl http2;
    server_name api.llm-simulator.example.com;

    # TLS Configuration
    ssl_certificate /etc/nginx/certs/tls.crt;
    ssl_certificate_key /etc/nginx/certs/tls.key;
    ssl_protocols TLSv1.3 TLSv1.2;
    ssl_ciphers 'TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256:ECDHE-RSA-AES256-GCM-SHA384';
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    ssl_session_tickets off;

    # OCSP Stapling
    ssl_stapling on;
    ssl_stapling_verify on;
    ssl_trusted_certificate /etc/nginx/certs/chain.pem;

    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # mTLS Configuration (optional)
    # ssl_client_certificate /etc/nginx/certs/client-ca.pem;
    # ssl_verify_client optional;
    # ssl_verify_depth 2;

    location / {
        proxy_pass http://llm_simulator;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Pass client certificate DN to backend (if mTLS)
        # proxy_set_header X-Client-DN $ssl_client_s_dn;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    # Rate limiting
    limit_req zone=api_limit burst=100 nodelay;
    limit_req_status 429;
}

# Rate limit zone
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/s;
```

---

## 4. Data Protection

### 4.1 Data Classification

```rust
// File: src/security/data_protection.rs

/// Data classification levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
pub enum DataClassification {
    /// Public data - no restrictions
    Public,

    /// Internal data - company confidential
    Internal,

    /// Confidential - restricted access
    Confidential,

    /// Restricted - highly sensitive (PII, PHI, PCI)
    Restricted,
}

/// Data protection configuration
#[derive(Debug, Clone)]
pub struct DataProtectionConfig {
    /// Encryption at rest
    pub encryption_at_rest: EncryptionConfig,

    /// Encryption in transit (covered by TLS)
    pub encryption_in_transit: bool,

    /// PII handling rules
    pub pii_handling: PiiHandlingConfig,

    /// Data retention policies
    pub retention: RetentionPolicy,

    /// Data masking rules
    pub masking_rules: Vec<MaskingRule>,
}

/// Encryption configuration for data at rest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    /// Enable encryption at rest
    pub enabled: bool,

    /// Encryption algorithm
    pub algorithm: EncryptionAlgorithm,

    /// Key management system
    pub kms_provider: KmsProvider,

    /// Key rotation period (days)
    pub key_rotation_days: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    Aes256Gcm,
    ChaCha20Poly1305,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KmsProvider {
    AwsKms { key_id: String },
    GcpKms { key_ring: String, key_name: String },
    AzureKeyVault { vault_url: String, key_name: String },
    HashicorpVault { mount_path: String, key_name: String },
    Local { key_path: PathBuf },
}

/// PII handling configuration
#[derive(Debug, Clone)]
pub struct PiiHandlingConfig {
    /// Automatically detect and redact PII
    pub auto_detect_pii: bool,

    /// PII types to detect
    pub pii_types: Vec<PiiType>,

    /// Action when PII detected
    pub detection_action: PiiDetectionAction,

    /// Log PII detection events
    pub log_detection: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum PiiType {
    EmailAddress,
    PhoneNumber,
    SocialSecurityNumber,
    CreditCardNumber,
    IpAddress,
    FullName,
    StreetAddress,
}

#[derive(Debug, Clone, Copy)]
pub enum PiiDetectionAction {
    /// Log warning but allow
    Warn,

    /// Redact/mask the PII
    Redact,

    /// Reject the request
    Reject,
}

/// Data retention policy
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Default retention period for logs
    pub log_retention_days: u32,

    /// Audit log retention (typically longer)
    pub audit_retention_days: u32,

    /// Metrics retention
    pub metrics_retention_days: u32,

    /// Automatic purging enabled
    pub auto_purge: bool,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            log_retention_days: 90,
            audit_retention_days: 2555,  // 7 years for compliance
            metrics_retention_days: 365,
            auto_purge: true,
        }
    }
}
```

### 4.2 Data Protection Implementation

**Key Principles**:
1. **No PII Storage**: System designed to NOT store sensitive user data
2. **Ephemeral Sessions**: Session data expires automatically
3. **Log Sanitization**: Automatic PII redaction in logs
4. **Secure Defaults**: Encryption enabled by default

```rust
/// Data sanitization for logging
pub struct DataSanitizer {
    pii_patterns: Vec<PiiPattern>,
}

impl DataSanitizer {
    /// Sanitize string for safe logging
    pub fn sanitize_for_logging(&self, data: &str) -> String {
        let mut sanitized = data.to_string();

        for pattern in &self.pii_patterns {
            sanitized = pattern.redact(&sanitized);
        }

        sanitized
    }

    /// Redact API keys (show only prefix)
    pub fn redact_api_key(&self, key: &str) -> String {
        if key.len() > 8 {
            format!("{}...", &key[..8])
        } else {
            "***".to_string()
        }
    }
}

#[derive(Debug, Clone)]
pub struct PiiPattern {
    /// Regex pattern for detection
    pattern: regex::Regex,

    /// Replacement string
    replacement: String,

    /// PII type
    pii_type: PiiType,
}

impl PiiPattern {
    pub fn redact(&self, text: &str) -> String {
        self.pattern.replace_all(text, self.replacement.as_str()).to_string()
    }
}

/// Default PII patterns
pub fn default_pii_patterns() -> Vec<PiiPattern> {
    vec![
        // Email addresses
        PiiPattern {
            pattern: regex::Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b").unwrap(),
            replacement: "[EMAIL_REDACTED]".to_string(),
            pii_type: PiiType::EmailAddress,
        },
        // Credit card numbers
        PiiPattern {
            pattern: regex::Regex::new(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b").unwrap(),
            replacement: "[CC_REDACTED]".to_string(),
            pii_type: PiiType::CreditCardNumber,
        },
        // SSN
        PiiPattern {
            pattern: regex::Regex::new(r"\b\d{3}-\d{2}-\d{4}\b").unwrap(),
            replacement: "[SSN_REDACTED]".to_string(),
            pii_type: PiiType::SocialSecurityNumber,
        },
        // IPv4 addresses
        PiiPattern {
            pattern: regex::Regex::new(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b").unwrap(),
            replacement: "[IP_REDACTED]".to_string(),
            pii_type: PiiType::IpAddress,
        },
    ]
}
```

---

## 5. Audit and Compliance

### 5.1 Audit Logging Architecture

```rust
// File: src/security/audit.rs

/// Comprehensive audit logging system
pub struct AuditLogger {
    /// Audit log writer
    writer: Arc<RwLock<AuditWriter>>,

    /// Audit configuration
    config: AuditConfig,

    /// Tamper-evident log chain
    log_chain: Arc<RwLock<AuditChain>>,
}

/// Audit logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    /// Enable audit logging
    pub enabled: bool,

    /// Audit log level
    pub level: AuditLevel,

    /// Log output destination
    pub destination: AuditDestination,

    /// Enable tamper-evident logging
    pub tamper_evident: bool,

    /// Enable log signing
    pub sign_logs: bool,

    /// Signing key (for tamper detection)
    pub signing_key: Option<SecretString>,

    /// Include request/response bodies
    pub include_bodies: bool,

    /// Maximum body size to log (bytes)
    pub max_body_size: usize,

    /// Retention period (days)
    pub retention_days: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AuditLevel {
    Minimal,      // Only critical events
    Standard,     // Authentication, authorization, errors
    Comprehensive,// All API calls and state changes
    Forensic,     // Everything including request/response bodies
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditDestination {
    File { path: PathBuf },
    Syslog { host: String, port: u16 },
    CloudWatch { log_group: String, log_stream: String },
    Splunk { hec_url: String, token: SecretString },
    Elasticsearch { url: String, index: String },
    Stdout,
}

/// Audit event structure (OWASP compliant)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Event timestamp (ISO 8601 UTC)
    pub timestamp: DateTime<Utc>,

    /// Event ID (UUID)
    pub event_id: Uuid,

    /// Event type
    pub event_type: AuditEventType,

    /// Event category
    pub category: AuditCategory,

    /// Actor/principal
    pub actor: ActorInfo,

    /// Resource accessed
    pub resource: ResourceInfo,

    /// Action performed
    pub action: String,

    /// Result (success/failure)
    pub result: AuditResult,

    /// Request ID (correlation)
    pub request_id: Uuid,

    /// Session ID
    pub session_id: Option<SessionId>,

    /// Source IP address
    pub source_ip: String,

    /// User agent
    pub user_agent: Option<String>,

    /// Request details
    pub request: Option<RequestDetails>,

    /// Response details
    pub response: Option<ResponseDetails>,

    /// Error details (if failure)
    pub error: Option<ErrorDetails>,

    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,

    /// Tamper-detection signature
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditEventType {
    // Authentication events
    AuthenticationSuccess,
    AuthenticationFailure,
    AuthenticationLockout,
    SessionCreated,
    SessionExpired,
    SessionTerminated,

    // Authorization events
    AuthorizationGranted,
    AuthorizationDenied,
    PermissionChanged,
    RoleAssigned,
    RoleRevoked,

    // Resource access
    ResourceAccessed,
    ResourceCreated,
    ResourceModified,
    ResourceDeleted,

    // Configuration changes
    ConfigurationChanged,
    ScenarioActivated,
    ScenarioDeactivated,

    // Security events
    RateLimitExceeded,
    SuspiciousActivity,
    SecurityPolicyViolation,
    IpBlocked,

    // Admin actions
    AdminActionPerformed,
    AdminLoginSuccess,
    AdminLoginFailure,

    // System events
    SystemStarted,
    SystemStopped,
    SystemError,
    HealthCheckFailed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditCategory {
    Authentication,
    Authorization,
    DataAccess,
    Configuration,
    Security,
    Administrative,
    System,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActorInfo {
    /// Principal identifier
    pub id: String,

    /// Principal type (user, service, system)
    pub principal_type: PrincipalType,

    /// Role
    pub role: Option<String>,

    /// Additional attributes
    pub attributes: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrincipalType {
    User,
    ServiceAccount,
    System,
    Anonymous,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceInfo {
    /// Resource type
    pub resource_type: String,

    /// Resource identifier
    pub resource_id: Option<String>,

    /// Resource path/endpoint
    pub path: String,

    /// HTTP method
    pub method: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditResult {
    /// Success or failure
    pub status: ResultStatus,

    /// Result code
    pub code: Option<String>,

    /// Result message
    pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResultStatus {
    Success,
    Failure,
    Partial,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestDetails {
    /// Request size (bytes)
    pub size_bytes: usize,

    /// Request body (if configured to log)
    pub body: Option<String>,

    /// Request headers
    pub headers: HashMap<String, String>,

    /// Query parameters
    pub query_params: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseDetails {
    /// HTTP status code
    pub status_code: u16,

    /// Response size (bytes)
    pub size_bytes: usize,

    /// Response body (if configured to log)
    pub body: Option<String>,

    /// Response headers
    pub headers: HashMap<String, String>,

    /// Processing duration (ms)
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetails {
    /// Error code
    pub code: String,

    /// Error message
    pub message: String,

    /// Stack trace (sanitized)
    pub stack_trace: Option<String>,
}

impl AuditLogger {
    /// Log audit event
    pub async fn log_event(&self, event: AuditEvent) -> Result<(), AuditError> {
        if !self.config.enabled {
            return Ok(());
        }

        // Sign event if tamper-evident logging enabled
        let mut event = event;
        if self.config.tamper_evident {
            event.signature = Some(self.sign_event(&event));

            // Add to audit chain
            self.log_chain.write().await.add_event(&event)?;
        }

        // Write to configured destination
        self.writer.write().await.write_event(&event).await?;

        Ok(())
    }

    /// Sign audit event for tamper detection
    fn sign_event(&self, event: &AuditEvent) -> String {
        if let Some(ref key) = self.config.signing_key {
            // HMAC-SHA256 signature
            let mut mac = Hmac::<Sha256>::new_from_slice(key.expose_secret().as_bytes())
                .expect("HMAC creation failed");

            let event_json = serde_json::to_string(event).unwrap();
            mac.update(event_json.as_bytes());

            let result = mac.finalize();
            hex::encode(result.into_bytes())
        } else {
            String::new()
        }
    }
}

/// Tamper-evident audit chain
pub struct AuditChain {
    /// Chain of event hashes
    chain: Vec<ChainLink>,

    /// Last hash
    last_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainLink {
    /// Event ID
    pub event_id: Uuid,

    /// Event timestamp
    pub timestamp: DateTime<Utc>,

    /// Event hash
    pub event_hash: String,

    /// Previous hash (forms chain)
    pub previous_hash: String,

    /// Chain index
    pub index: u64,
}

impl AuditChain {
    pub fn new() -> Self {
        Self {
            chain: Vec::new(),
            last_hash: "0".repeat(64), // Genesis hash
        }
    }

    pub fn add_event(&mut self, event: &AuditEvent) -> Result<(), AuditError> {
        let event_json = serde_json::to_string(event)?;
        let event_hash = Self::hash_event(&event_json);

        let link = ChainLink {
            event_id: event.event_id,
            timestamp: event.timestamp,
            event_hash: event_hash.clone(),
            previous_hash: self.last_hash.clone(),
            index: self.chain.len() as u64,
        };

        self.chain.push(link);
        self.last_hash = event_hash;

        Ok(())
    }

    fn hash_event(event_json: &str) -> String {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(event_json.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Verify chain integrity
    pub fn verify_integrity(&self) -> bool {
        let mut expected_hash = "0".repeat(64);

        for link in &self.chain {
            if link.previous_hash != expected_hash {
                return false;
            }
            expected_hash = link.event_hash.clone();
        }

        expected_hash == self.last_hash
    }
}
```

### 5.2 Audit Event Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "LLM-Simulator Audit Event",
  "type": "object",
  "required": [
    "timestamp",
    "event_id",
    "event_type",
    "category",
    "actor",
    "resource",
    "action",
    "result",
    "source_ip"
  ],
  "properties": {
    "timestamp": {
      "type": "string",
      "format": "date-time",
      "description": "ISO 8601 UTC timestamp"
    },
    "event_id": {
      "type": "string",
      "format": "uuid",
      "description": "Unique event identifier"
    },
    "event_type": {
      "type": "string",
      "enum": [
        "AuthenticationSuccess",
        "AuthenticationFailure",
        "AuthorizationGranted",
        "AuthorizationDenied",
        "ResourceAccessed",
        "ConfigurationChanged"
      ]
    },
    "category": {
      "type": "string",
      "enum": [
        "Authentication",
        "Authorization",
        "DataAccess",
        "Configuration",
        "Security",
        "Administrative",
        "System"
      ]
    },
    "actor": {
      "type": "object",
      "properties": {
        "id": { "type": "string" },
        "principal_type": { "type": "string" },
        "role": { "type": "string" },
        "attributes": { "type": "object" }
      }
    },
    "resource": {
      "type": "object",
      "properties": {
        "resource_type": { "type": "string" },
        "resource_id": { "type": "string" },
        "path": { "type": "string" },
        "method": { "type": "string" }
      }
    },
    "action": {
      "type": "string",
      "description": "Action performed"
    },
    "result": {
      "type": "object",
      "properties": {
        "status": {
          "type": "string",
          "enum": ["Success", "Failure", "Partial"]
        },
        "code": { "type": "string" },
        "message": { "type": "string" }
      }
    },
    "source_ip": {
      "type": "string",
      "format": "ipv4"
    },
    "signature": {
      "type": "string",
      "description": "HMAC-SHA256 signature for tamper detection"
    }
  }
}
```

### 5.3 Example Audit Events

```json
// Authentication Success
{
  "timestamp": "2025-11-26T12:34:56.789Z",
  "event_id": "123e4567-e89b-12d3-a456-426614174000",
  "event_type": "AuthenticationSuccess",
  "category": "Authentication",
  "actor": {
    "id": "sk-abc123...",
    "principal_type": "User",
    "role": "User",
    "attributes": {}
  },
  "resource": {
    "resource_type": "api",
    "path": "/v1/chat/completions",
    "method": "POST"
  },
  "action": "authenticate",
  "result": {
    "status": "Success",
    "message": "API key authentication successful"
  },
  "request_id": "req-789xyz",
  "source_ip": "192.168.1.100",
  "user_agent": "python-requests/2.31.0",
  "signature": "a1b2c3d4..."
}

// Authorization Denied
{
  "timestamp": "2025-11-26T12:35:01.123Z",
  "event_id": "223e4567-e89b-12d3-a456-426614174001",
  "event_type": "AuthorizationDenied",
  "category": "Authorization",
  "actor": {
    "id": "sk-xyz789...",
    "principal_type": "User",
    "role": "User",
    "attributes": {}
  },
  "resource": {
    "resource_type": "admin",
    "path": "/admin/config",
    "method": "POST"
  },
  "action": "modify_configuration",
  "result": {
    "status": "Failure",
    "code": "PERMISSION_DENIED",
    "message": "Insufficient permissions for admin API"
  },
  "request_id": "req-890abc",
  "source_ip": "192.168.1.100",
  "error": {
    "code": "403",
    "message": "Forbidden: Admin role required"
  },
  "signature": "e5f6g7h8..."
}

// Configuration Changed
{
  "timestamp": "2025-11-26T12:40:15.456Z",
  "event_id": "323e4567-e89b-12d3-a456-426614174002",
  "event_type": "ConfigurationChanged",
  "category": "Administrative",
  "actor": {
    "id": "sk-admin123...",
    "principal_type": "User",
    "role": "Admin",
    "attributes": {
      "source": "ci-cd-system"
    }
  },
  "resource": {
    "resource_type": "configuration",
    "resource_id": "latency_profiles",
    "path": "/admin/config",
    "method": "POST"
  },
  "action": "update_latency_profile",
  "result": {
    "status": "Success",
    "message": "Latency profile updated"
  },
  "request_id": "req-901def",
  "source_ip": "10.0.1.50",
  "metadata": {
    "config_version": "2.1.0",
    "changed_fields": ["gpt-4.p50_ms", "gpt-4.p99_ms"],
    "previous_values": {"gpt-4.p50_ms": 500, "gpt-4.p99_ms": 2000},
    "new_values": {"gpt-4.p50_ms": 450, "gpt-4.p99_ms": 1800}
  },
  "signature": "i9j0k1l2..."
}
```

---

## 6. Secret Management

### 6.1 Secret Management Architecture

```rust
// File: src/security/secrets.rs

/// Secret management system integration
pub struct SecretManager {
    /// Secret provider
    provider: Arc<dyn SecretProvider>,

    /// Secret cache (encrypted in memory)
    cache: Arc<RwLock<SecretCache>>,

    /// Configuration
    config: SecretConfig,
}

/// Secret management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecretConfig {
    /// Secret provider type
    pub provider: SecretProviderType,

    /// Enable secret caching
    pub enable_caching: bool,

    /// Cache TTL (seconds)
    pub cache_ttl: u64,

    /// Auto-rotation enabled
    pub auto_rotation: bool,

    /// Rotation period (days)
    pub rotation_period_days: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecretProviderType {
    /// AWS Secrets Manager
    AwsSecretsManager {
        region: String,
        secret_prefix: String,
    },

    /// Google Secret Manager
    GcpSecretManager {
        project_id: String,
        secret_prefix: String,
    },

    /// Azure Key Vault
    AzureKeyVault {
        vault_url: String,
        secret_prefix: String,
    },

    /// HashiCorp Vault
    HashicorpVault {
        addr: String,
        mount_path: String,
        namespace: Option<String>,
    },

    /// Kubernetes Secrets
    KubernetesSecrets {
        namespace: String,
    },

    /// Environment variables (not recommended for production)
    Environment,

    /// File-based secrets
    File {
        secrets_dir: PathBuf,
    },
}

#[async_trait::async_trait]
pub trait SecretProvider: Send + Sync {
    /// Retrieve secret value
    async fn get_secret(&self, key: &str) -> Result<SecretString, SecretError>;

    /// Store secret value
    async fn set_secret(&self, key: &str, value: &SecretString) -> Result<(), SecretError>;

    /// Delete secret
    async fn delete_secret(&self, key: &str) -> Result<(), SecretError>;

    /// List secret keys
    async fn list_secrets(&self) -> Result<Vec<String>, SecretError>;

    /// Rotate secret
    async fn rotate_secret(&self, key: &str) -> Result<SecretString, SecretError>;
}

/// Secure string wrapper (zeroizes on drop)
pub struct SecretString(String);

impl SecretString {
    pub fn new(value: String) -> Self {
        Self(value)
    }

    pub fn expose_secret(&self) -> &str {
        &self.0
    }
}

impl Drop for SecretString {
    fn drop(&mut self) {
        // Zeroize memory to prevent secrets in memory dumps
        unsafe {
            let bytes = self.0.as_bytes_mut();
            for b in bytes {
                std::ptr::write_volatile(b, 0);
            }
        }
    }
}

/// Example: AWS Secrets Manager integration
pub struct AwsSecretsManagerProvider {
    client: aws_sdk_secretsmanager::Client,
    secret_prefix: String,
}

#[async_trait::async_trait]
impl SecretProvider for AwsSecretsManagerProvider {
    async fn get_secret(&self, key: &str) -> Result<SecretString, SecretError> {
        let secret_name = format!("{}/{}", self.secret_prefix, key);

        let response = self.client
            .get_secret_value()
            .secret_id(secret_name)
            .send()
            .await
            .map_err(|e| SecretError::ProviderError(e.to_string()))?;

        let secret_string = response
            .secret_string()
            .ok_or(SecretError::SecretNotFound)?
            .to_string();

        Ok(SecretString::new(secret_string))
    }

    async fn set_secret(&self, key: &str, value: &SecretString) -> Result<(), SecretError> {
        let secret_name = format!("{}/{}", self.secret_prefix, key);

        self.client
            .put_secret_value()
            .secret_id(secret_name)
            .secret_string(value.expose_secret())
            .send()
            .await
            .map_err(|e| SecretError::ProviderError(e.to_string()))?;

        Ok(())
    }

    async fn delete_secret(&self, key: &str) -> Result<(), SecretError> {
        let secret_name = format!("{}/{}", self.secret_prefix, key);

        self.client
            .delete_secret()
            .secret_id(secret_name)
            .send()
            .await
            .map_err(|e| SecretError::ProviderError(e.to_string()))?;

        Ok(())
    }

    async fn list_secrets(&self) -> Result<Vec<String>, SecretError> {
        let response = self.client
            .list_secrets()
            .send()
            .await
            .map_err(|e| SecretError::ProviderError(e.to_string()))?;

        let secrets = response
            .secret_list()
            .unwrap_or_default()
            .iter()
            .filter_map(|s| s.name().map(|n| n.to_string()))
            .filter(|n| n.starts_with(&self.secret_prefix))
            .map(|n| n.strip_prefix(&format!("{}/", self.secret_prefix))
                .unwrap_or(&n)
                .to_string())
            .collect();

        Ok(secrets)
    }

    async fn rotate_secret(&self, key: &str) -> Result<SecretString, SecretError> {
        // Trigger rotation in AWS Secrets Manager
        let secret_name = format!("{}/{}", self.secret_prefix, key);

        self.client
            .rotate_secret()
            .secret_id(secret_name)
            .send()
            .await
            .map_err(|e| SecretError::ProviderError(e.to_string()))?;

        // Retrieve new secret value
        self.get_secret(key).await
    }
}
```

### 6.2 Secrets to Manage

| Secret Type | Purpose | Rotation Period | Storage Location |
|-------------|---------|-----------------|------------------|
| `admin_api_key` | Admin API authentication | 90 days | Secret Manager |
| `jwt_signing_key` | JWT token signing | 365 days | Secret Manager |
| `tls_private_key` | TLS certificate private key | 365 days | Secret Manager |
| `metrics_auth_token` | Prometheus metrics auth | 180 days | Secret Manager |
| `audit_signing_key` | Audit log signing | 365 days | Secret Manager |
| `encryption_key` | Data encryption at rest | 365 days | KMS |

### 6.3 Secret Rotation Process

```
┌──────────────────────────────────────────────────────────────┐
│              Secret Rotation Workflow                        │
└──────────────────────────────────────────────────────────────┘

Scheduler            Secret Manager         Application        Audit
   │                      │                      │               │
   │  rotation_due()      │                      │               │
   ├─────────────────────>│                      │               │
   │                      │                      │               │
   │                      │  generate_new_key()  │               │
   │                      ├──────────────────────│               │
   │                      │                      │               │
   │                      │  store_new_secret()  │               │
   │                      │─────────────────────>│               │
   │                      │                      │               │
   │                      │  notify_rotation()   │               │
   │                      │─────────────────────>│               │
   │                      │                      │               │
   │                      │  reload_config()     │               │
   │                      │<─────────────────────│               │
   │                      │                      │               │
   │                      │  verify_new_key()    │               │
   │                      ├──────────────────────│               │
   │                      │                      │               │
   │                      │  deprecate_old_key() │               │
   │                      │─────────────────────>│               │
   │                      │                      │               │
   │                      │  log_rotation_event()│               │
   │                      │──────────────────────┼──────────────>│
   │                      │                      │               │
```

---

## 7. Supply Chain Security

### 7.1 Dependency Management

```toml
# Cargo.toml - Security-focused dependency management

[package]
name = "llm-simulator"
version = "1.0.0"
edition = "2021"
rust-version = "1.75"

[dependencies]
# Core dependencies with version pinning
axum = { version = "=0.7.5", features = ["macros"] }
tokio = { version = "=1.38.0", features = ["full"] }
tower = "=0.4.13"
serde = { version = "=1.0.203", features = ["derive"] }

# Security dependencies
ring = "=0.17.8"  # Cryptography
rustls = "=0.23.7"  # TLS implementation
webpki-roots = "=0.26.3"  # Root certificates
hmac = "=0.12.1"  # HMAC for audit log signing
sha2 = "=0.10.8"  # SHA-256 hashing

# Audit all dependencies
[dev-dependencies]
cargo-audit = "=0.20.0"

[profile.release]
# Hardening options
opt-level = 3
lto = true
codegen-units = 1
strip = true
panic = "abort"

# Security flags
overflow-checks = true
```

### 7.2 Supply Chain Security Checks

```bash
#!/bin/bash
# security-checks.sh - Automated security scanning

set -e

echo "=== Supply Chain Security Checks ==="

# 1. Dependency Vulnerability Scanning
echo "[1/6] Running cargo-audit..."
cargo audit

# 2. Dependency License Check
echo "[2/6] Checking dependency licenses..."
cargo-license --json | jq '.[] | select(.license | contains("GPL") or contains("AGPL"))' && exit 1

# 3. SBOM Generation
echo "[3/6] Generating SBOM..."
cargo-sbom --output-format spdx > sbom.spdx.json

# 4. Static Analysis
echo "[4/6] Running clippy security lints..."
cargo clippy -- \
  -D clippy::unwrap_used \
  -D clippy::expect_used \
  -D clippy::panic \
  -D clippy::todo \
  -D clippy::unimplemented

# 5. Secret Scanning
echo "[5/6] Scanning for secrets..."
gitleaks detect --source . --verbose

# 6. Container Scanning (if building container)
echo "[6/6] Scanning container image..."
trivy image llm-simulator:latest --severity HIGH,CRITICAL

echo "✅ All security checks passed"
```

### 7.3 SBOM (Software Bill of Materials)

```json
{
  "spdxVersion": "SPDX-2.3",
  "dataLicense": "CC0-1.0",
  "SPDXID": "SPDXRef-DOCUMENT",
  "name": "llm-simulator",
  "documentNamespace": "https://example.com/sbom/llm-simulator-1.0.0",
  "packages": [
    {
      "SPDXID": "SPDXRef-Package-llm-simulator",
      "name": "llm-simulator",
      "versionInfo": "1.0.0",
      "packageFileName": "llm-simulator",
      "supplier": "Organization: Example Corp",
      "licenseConcluded": "Apache-2.0",
      "copyrightText": "Copyright 2025 Example Corp"
    },
    {
      "SPDXID": "SPDXRef-Package-axum-0.7.5",
      "name": "axum",
      "versionInfo": "0.7.5",
      "supplier": "crates.io",
      "licenseConcluded": "MIT",
      "checksums": [
        {
          "algorithm": "SHA256",
          "checksumValue": "abc123..."
        }
      ]
    }
  ],
  "relationships": [
    {
      "spdxElementId": "SPDXRef-Package-llm-simulator",
      "relationshipType": "DEPENDS_ON",
      "relatedSpdxElement": "SPDXRef-Package-axum-0.7.5"
    }
  ]
}
```

---

## 8. Threat Model

### 8.1 Threat Modeling Summary

**Methodology**: STRIDE (Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, Elevation of Privilege)

| Threat Category | Threat | Mitigation | Risk Level |
|----------------|--------|------------|------------|
| **Spoofing** | API key forgery | Format validation, prefix verification | Low |
| **Spoofing** | Session hijacking | Session timeouts, IP binding | Medium |
| **Tampering** | Audit log tampering | Tamper-evident logging, log signing | Low |
| **Tampering** | Configuration tampering | Admin auth, audit logging | Low |
| **Repudiation** | Deny API usage | Comprehensive audit logs | Low |
| **Information Disclosure** | API key leakage | Key redaction in logs, secure storage | Medium |
| **Information Disclosure** | Request/response leakage | TLS encryption, PII redaction | Low |
| **Denial of Service** | Request flooding | Rate limiting, backpressure | Medium |
| **Denial of Service** | Resource exhaustion | Concurrency limits, timeouts | Medium |
| **Elevation of Privilege** | Admin API access | RBAC, admin key separation | Low |

### 8.2 Threat Scenarios and Mitigations

```markdown
## Threat Scenario 1: Unauthorized Admin Access

**Attack Vector**: Attacker attempts to access admin API without credentials

**Attacker Goal**: Modify system configuration, activate chaos scenarios

**Mitigations**:
1. Separate admin API key from user keys
2. Admin endpoints require Admin role
3. All admin actions logged with audit trail
4. IP allowlisting for admin access (optional)
5. Rate limiting on admin endpoints

**Detection**:
- Failed authorization attempts logged
- Anomaly detection on admin endpoint access patterns
- SIEM alerts on repeated authorization failures

---

## Threat Scenario 2: API Key Compromise

**Attack Vector**: User API key exposed in logs or source control

**Attacker Goal**: Abuse API for DoS or data exfiltration

**Mitigations**:
1. API keys never logged in full (prefix only)
2. Automatic key redaction in logs
3. Rate limiting per key
4. Key status monitoring and manual revocation
5. Session concurrency limits per key

**Detection**:
- Unusual usage patterns (spike in requests)
- Requests from unexpected IP addresses
- Concurrent sessions exceeding limit

---

## Threat Scenario 3: Denial of Service

**Attack Vector**: Distributed request flooding

**Attacker Goal**: Overwhelm system to cause unavailability

**Mitigations**:
1. Rate limiting at multiple layers (IP, API key, endpoint)
2. Concurrency limits via semaphore
3. Request queue with backpressure
4. Graceful degradation under load
5. DDoS protection at ingress (CloudFlare, AWS Shield)

**Detection**:
- Request rate metrics
- Queue depth monitoring
- Backpressure events
- Error rate increase

---

## Threat Scenario 4: Audit Log Tampering

**Attack Vector**: Attacker with system access modifies audit logs

**Attacker Goal**: Hide malicious activity

**Mitigations**:
1. Tamper-evident audit chain (hash chaining)
2. Audit log signing with HMAC
3. Write-once log storage
4. Regular integrity verification
5. Ship logs to external SIEM

**Detection**:
- Chain integrity verification
- Signature validation
- Log gap detection
- External log correlation
```

### 8.3 Attack Surface Analysis

```
┌───────────────────────────────────────────────────────────────┐
│                    Attack Surface Map                         │
└───────────────────────────────────────────────────────────────┘

External Attack Surface (Internet-Facing):
├─ HTTP/HTTPS Endpoints
│  ├─ /v1/chat/completions (User API) ──┬─ Auth: API Key
│  ├─ /v1/messages (User API)           │  Rate Limit: 1000/min
│  ├─ /v1/embeddings (User API)         │  Input Validation: Schema
│  ├─ /health (Public)                  └─ Risk: Medium
│  └─ /metrics (Public/Protected)       ── Risk: Low

Administrative Attack Surface (Restricted):
├─ Admin API Endpoints
│  ├─ /admin/config ──────────────────┬─ Auth: Admin API Key
│  ├─ /admin/scenarios                │  IP Allowlist: Optional
│  └─ /admin/stats                    └─ Risk: High (if compromised)

Internal Attack Surface (Container/Host):
├─ Configuration Files ────────────────┬─ Permissions: 0600
│  ├─ /etc/llm-simulator/config.yaml  │  Owner: llm-simulator
│  └─ /etc/certs/*.pem                └─ Risk: Medium

Network Attack Surface:
├─ Ingress Traffic ────────────────────┬─ Protection: WAF, DDoS Shield
│  ├─ Port 443 (HTTPS)                │  Encryption: TLS 1.3
│  └─ Port 8080 (Internal)            └─ Risk: Low

Egress Traffic:
├─ Secret Manager ─────────────────────┬─ Auth: IAM Role/Service Account
├─ Metrics Export (Prometheus)        │  Encryption: TLS
└─ Audit Log Export (SIEM)            └─ Risk: Low
```

---

## 9. Security Configuration Reference

### 9.1 Security Configuration File

```yaml
# config/security.yaml - Complete security configuration

# Authentication Configuration
authentication:
  enabled: true
  methods:
    api_key:
      enabled: true
      format_regex: "^sk-[a-zA-Z0-9]{32,}$"
      max_sessions_per_key: 100
    jwt:
      enabled: false
      secret_key: "${JWT_SECRET}"  # From secret manager
      expiration_seconds: 3600
      algorithm: "HS256"
    mtls:
      enabled: false
      client_ca_path: "/etc/certs/client-ca.pem"
      require_cert: false

  # Brute force protection
  brute_force_protection:
    enabled: true
    max_failed_attempts: 5
    lockout_duration_seconds: 900
    reset_duration_seconds: 3600

# Authorization Configuration
authorization:
  enabled: true
  default_deny: true

  roles:
    user:
      permissions:
        - resource: "completions"
          actions: ["execute"]
        - resource: "chat"
          actions: ["execute"]
        - resource: "embeddings"
          actions: ["execute"]
        - resource: "models"
          actions: ["read"]

    admin:
      permissions:
        - resource: "*"
          actions: ["*"]

    readonly:
      permissions:
        - resource: "metrics"
          actions: ["read"]
        - resource: "health"
          actions: ["read"]
        - resource: "stats"
          actions: ["read"]

# Network Security
network_security:
  tls:
    enabled: true
    min_version: "1.3"
    cert_path: "/etc/certs/tls.crt"
    key_path: "/etc/certs/tls.key"
    cipher_suites:
      - "TLS_AES_256_GCM_SHA384"
      - "TLS_CHACHA20_POLY1305_SHA256"
      - "TLS_AES_128_GCM_SHA256"
    mtls:
      enabled: false
      client_ca_path: "/etc/certs/client-ca.pem"

  rate_limiting:
    enabled: true
    requests_per_minute: 1000
    burst_size: 100
    per_api_key: true

  ip_allowlist:
    enabled: false
    allowed_cidrs:
      - "10.0.0.0/8"
      - "192.168.0.0/16"

# Data Protection
data_protection:
  encryption_at_rest:
    enabled: false  # No sensitive data stored

  pii_handling:
    auto_detect: true
    detection_action: "redact"  # warn | redact | reject
    log_detection: true

  retention:
    log_retention_days: 90
    audit_retention_days: 2555  # 7 years
    metrics_retention_days: 365
    auto_purge: true

# Audit Logging
audit_logging:
  enabled: true
  level: "comprehensive"  # minimal | standard | comprehensive | forensic

  destination:
    type: "file"
    path: "/var/log/llm-simulator/audit.log"

  tamper_evident: true
  sign_logs: true
  signing_key: "${AUDIT_SIGNING_KEY}"

  include_bodies: false
  max_body_size: 10240

  retention_days: 2555

# Secret Management
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

# Security Headers
security_headers:
  strict_transport_security: "max-age=31536000; includeSubDomains; preload"
  x_frame_options: "DENY"
  x_content_type_options: "nosniff"
  x_xss_protection: "1; mode=block"
  referrer_policy: "strict-origin-when-cross-origin"
  content_security_policy: "default-src 'self'"
  permissions_policy: "geolocation=(), microphone=(), camera=()"

# Monitoring and Alerting
security_monitoring:
  enabled: true

  alerts:
    - name: "high_failed_auth_rate"
      condition: "failed_auth_rate > 10 per minute"
      severity: "high"
      action: "notify_security_team"

    - name: "admin_api_unauthorized_access"
      condition: "admin_authz_denied > 0"
      severity: "critical"
      action: "notify_security_team"

    - name: "unusual_request_pattern"
      condition: "request_rate > 2x baseline"
      severity: "medium"
      action: "notify_ops_team"
```

### 9.2 Environment Variables (Secrets)

```bash
# Environment variables for sensitive configuration
# These should be loaded from secret manager, not stored in files

# Authentication
export ADMIN_API_KEY="sk-admin-..."
export JWT_SECRET="..."

# TLS Certificates (paths to secret-mounted files)
export TLS_CERT_PATH="/run/secrets/tls.crt"
export TLS_KEY_PATH="/run/secrets/tls.key"

# Audit Log Signing
export AUDIT_SIGNING_KEY="..."

# Secret Manager Credentials
export AWS_REGION="us-east-1"
export AWS_SECRET_ACCESS_KEY="..."
# OR use IAM role (preferred)

# Database Encryption (if used)
export DB_ENCRYPTION_KEY="..."

# Metrics Authentication
export METRICS_AUTH_TOKEN="..."
```

---

## 10. Compliance Control Mappings

### 10.1 SOC 2 Control Mappings

| SOC 2 Control | LLM-Simulator Implementation | Evidence |
|---------------|------------------------------|----------|
| **CC6.1** - Logical and Physical Access Controls | RBAC, API key authentication, admin separation | Auth logs, config files |
| **CC6.2** - Prior to issuing system credentials | API key validation, format checking | Auth module code |
| **CC6.3** - Removes access when no longer required | Key revocation, status management | Audit logs |
| **CC6.6** - Logical access security measures | Rate limiting, brute force protection | Middleware code |
| **CC6.7** - Restricts transmission and storage of sensitive information | TLS encryption, no PII storage | Network config |
| **CC7.2** - System monitoring | Audit logging, metrics, alerts | Audit logs, metrics |
| **CC7.3** - Security event logging | Comprehensive audit events | Audit log samples |
| **CC7.4** - Protection against logical and physical threats | DDoS protection, rate limiting, network policies | Infrastructure config |

### 10.2 HIPAA Control Mappings (if handling PHI)

| HIPAA Requirement | Implementation | Status |
|-------------------|----------------|--------|
| **164.308(a)(1)(i)** - Security Management | Risk assessment, threat model | ✅ Implemented |
| **164.308(a)(3)(i)** - Workforce Security | RBAC, role separation | ✅ Implemented |
| **164.308(a)(4)(i)** - Access Management | Authentication, authorization | ✅ Implemented |
| **164.308(a)(5)(i)** - Security Awareness | Security documentation | ✅ This document |
| **164.312(a)(1)** - Access Control | Unique user IDs (API keys) | ✅ Implemented |
| **164.312(a)(2)(i)** - Emergency Access | Admin override (logged) | ⚠️ Configurable |
| **164.312(b)** - Audit Controls | Comprehensive audit logging | ✅ Implemented |
| **164.312(c)(1)** - Integrity | Tamper-evident logging | ✅ Implemented |
| **164.312(d)** - Person or Entity Authentication | API key/JWT authentication | ✅ Implemented |
| **164.312(e)(1)** - Transmission Security | TLS 1.3 encryption | ✅ Implemented |

### 10.3 PCI-DSS Control Mappings (if processing payments)

| PCI-DSS Requirement | Implementation | Status |
|---------------------|----------------|--------|
| **1.1** - Firewall Configuration | Network policies, ingress rules | ✅ Implemented |
| **2.2** - System Hardening | Minimal attack surface, secure defaults | ✅ Implemented |
| **2.3** - Encryption for non-console access | TLS 1.3, mTLS optional | ✅ Implemented |
| **8.1** - Unique User IDs | API key per user | ✅ Implemented |
| **8.2** - User Authentication | API key validation, JWT | ✅ Implemented |
| **8.3** - Multi-factor Authentication | mTLS (optional) | ⚠️ Configurable |
| **10.1** - Audit Trails | Comprehensive audit logging | ✅ Implemented |
| **10.2** - Automated Audit Trail | All events logged | ✅ Implemented |
| **10.3** - Audit Trail Protection | Tamper-evident, signed logs | ✅ Implemented |

### 10.4 Compliance Reporting

```rust
// File: src/security/compliance.rs

/// Compliance reporting module
pub struct ComplianceReporter {
    audit_logger: Arc<AuditLogger>,
    config: ComplianceConfig,
}

#[derive(Debug, Clone)]
pub struct ComplianceConfig {
    pub frameworks: Vec<ComplianceFramework>,
    pub report_format: ReportFormat,
    pub report_period: ReportPeriod,
}

#[derive(Debug, Clone, Copy)]
pub enum ComplianceFramework {
    Soc2Type2,
    Hipaa,
    PciDss,
    Gdpr,
    Iso27001,
}

#[derive(Debug, Clone, Copy)]
pub enum ReportFormat {
    Json,
    Csv,
    Pdf,
}

#[derive(Debug, Clone, Copy)]
pub enum ReportPeriod {
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Annually,
}

impl ComplianceReporter {
    /// Generate compliance report
    pub async fn generate_report(
        &self,
        framework: ComplianceFramework,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<ComplianceReport, ComplianceError> {
        let events = self.audit_logger
            .query_events(start_date, end_date)
            .await?;

        let report = match framework {
            ComplianceFramework::Soc2Type2 => {
                self.generate_soc2_report(events, start_date, end_date).await?
            }
            ComplianceFramework::Hipaa => {
                self.generate_hipaa_report(events, start_date, end_date).await?
            }
            ComplianceFramework::PciDss => {
                self.generate_pci_report(events, start_date, end_date).await?
            }
            _ => return Err(ComplianceError::UnsupportedFramework),
        };

        Ok(report)
    }

    async fn generate_soc2_report(
        &self,
        events: Vec<AuditEvent>,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<ComplianceReport, ComplianceError> {
        let mut report = ComplianceReport::new(
            ComplianceFramework::Soc2Type2,
            start_date,
            end_date,
        );

        // CC6.1: Access Control Events
        let auth_events: Vec<&AuditEvent> = events.iter()
            .filter(|e| matches!(e.category, AuditCategory::Authentication))
            .collect();

        report.add_control("CC6.1", ControlEvidence {
            control_name: "Logical Access Controls".to_string(),
            total_events: auth_events.len(),
            successful_events: auth_events.iter()
                .filter(|e| e.result.status == ResultStatus::Success)
                .count(),
            failed_events: auth_events.iter()
                .filter(|e| e.result.status == ResultStatus::Failure)
                .count(),
            sample_events: auth_events.iter().take(10).cloned().collect(),
        });

        // CC7.2: System Monitoring
        report.add_control("CC7.2", ControlEvidence {
            control_name: "System Monitoring".to_string(),
            total_events: events.len(),
            successful_events: events.iter()
                .filter(|e| e.result.status == ResultStatus::Success)
                .count(),
            failed_events: events.iter()
                .filter(|e| e.result.status == ResultStatus::Failure)
                .count(),
            sample_events: events.iter().take(10).cloned().collect(),
        });

        Ok(report)
    }
}
```

---

## 11. Incident Response

### 11.1 Security Incident Response Plan

```yaml
# Incident Response Runbook

incident_types:
  - name: "Unauthorized Access Attempt"
    severity: "high"
    detection:
      - "Multiple failed authentication attempts"
      - "Authorization denied events for admin API"
    response_steps:
      - "Verify incident from audit logs"
      - "Identify source IP and API key (if any)"
      - "Block IP at firewall/WAF level"
      - "Revoke compromised API key if identified"
      - "Notify security team"
      - "Document incident in security tracker"
    escalation: "Security Team Lead"

  - name: "API Key Compromise"
    severity: "critical"
    detection:
      - "API key used from unexpected IP"
      - "Unusual request patterns"
      - "Rate limit exceeded"
    response_steps:
      - "Immediately revoke compromised key"
      - "Notify key owner"
      - "Review audit logs for unauthorized activity"
      - "Assess impact (data accessed, modifications made)"
      - "Generate new key for affected user"
      - "Update security documentation"
    escalation: "Security Team Lead, CTO"

  - name: "Denial of Service"
    severity: "high"
    detection:
      - "Request rate >10x baseline"
      - "Queue depth >90% capacity"
      - "High error rate"
    response_steps:
      - "Enable aggressive rate limiting"
      - "Activate DDoS protection"
      - "Identify attack source IPs"
      - "Block malicious IPs"
      - "Scale infrastructure if needed"
      - "Notify operations team"
    escalation: "Operations Lead"

  - name: "Configuration Tampering"
    severity: "critical"
    detection:
      - "Unauthorized configuration change"
      - "Admin API access without proper auth"
    response_steps:
      - "Rollback configuration to last known good"
      - "Review audit logs for unauthorized changes"
      - "Identify attacker (IP, credentials used)"
      - "Revoke compromised admin credentials"
      - "Enable IP allowlist for admin API"
      - "Conduct security review"
    escalation: "Security Team Lead, CTO"
```

### 11.2 Incident Response Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                 Security Incident Response Workflow             │
└─────────────────────────────────────────────────────────────────┘

1. DETECTION
   ├─ Automated Alerts (SIEM, Metrics)
   ├─ Manual Report (User, Team Member)
   └─ Audit Log Analysis
          │
          ▼
2. TRIAGE
   ├─ Assess Severity (Low, Medium, High, Critical)
   ├─ Determine Impact (Confidentiality, Integrity, Availability)
   └─ Assign Incident Owner
          │
          ▼
3. CONTAINMENT
   ├─ Immediate Actions (Block IP, Revoke Keys)
   ├─ Isolate Affected Systems
   └─ Preserve Evidence (Logs, Network Captures)
          │
          ▼
4. INVESTIGATION
   ├─ Analyze Audit Logs
   ├─ Review System Logs
   ├─ Identify Root Cause
   └─ Determine Attack Vector
          │
          ▼
5. ERADICATION
   ├─ Remove Malicious Access
   ├─ Patch Vulnerabilities
   └─ Update Security Controls
          │
          ▼
6. RECOVERY
   ├─ Restore Service (if down)
   ├─ Verify System Integrity
   └─ Resume Normal Operations
          │
          ▼
7. POST-INCIDENT
   ├─ Document Incident (Timeline, Actions)
   ├─ Root Cause Analysis
   ├─ Update Runbooks
   ├─ Improve Controls
   └─ Team Debriefing
```

---

## 12. Security Testing

### 12.1 Security Test Plan

```yaml
# Security Testing Strategy

test_categories:
  authentication_testing:
    tests:
      - name: "API key format validation"
        type: "unit"
        description: "Verify invalid key formats are rejected"

      - name: "Brute force protection"
        type: "integration"
        description: "Verify account lockout after failed attempts"

      - name: "Session timeout"
        type: "integration"
        description: "Verify sessions expire after TTL"

      - name: "JWT token validation"
        type: "integration"
        description: "Verify expired/invalid tokens rejected"

  authorization_testing:
    tests:
      - name: "RBAC enforcement"
        type: "integration"
        description: "Verify users cannot access unauthorized resources"

      - name: "Admin API protection"
        type: "integration"
        description: "Verify admin endpoints require admin role"

      - name: "Privilege escalation prevention"
        type: "penetration"
        description: "Attempt to escalate from user to admin"

  network_security_testing:
    tests:
      - name: "TLS version enforcement"
        type: "configuration"
        description: "Verify TLS 1.3 minimum version"

      - name: "Cipher suite validation"
        type: "configuration"
        description: "Verify only secure ciphers enabled"

      - name: "mTLS certificate validation"
        type: "integration"
        description: "Verify client certificate validation works"

  injection_testing:
    tests:
      - name: "SQL injection prevention"
        type: "penetration"
        description: "Attempt SQL injection in all inputs (N/A - no SQL)"

      - name: "Command injection prevention"
        type: "penetration"
        description: "Attempt command injection in parameters"

      - name: "JSON injection"
        type: "penetration"
        description: "Attempt malformed JSON payloads"

  audit_logging_testing:
    tests:
      - name: "Authentication event logging"
        type: "integration"
        description: "Verify auth events are logged"

      - name: "Authorization event logging"
        type: "integration"
        description: "Verify authz decisions are logged"

      - name: "Audit log integrity"
        type: "integration"
        description: "Verify tamper-evident chain works"

      - name: "Audit log signing"
        type: "integration"
        description: "Verify log signatures are valid"

  rate_limiting_testing:
    tests:
      - name: "Per-key rate limiting"
        type: "load"
        description: "Verify rate limits enforced per API key"

      - name: "Burst handling"
        type: "load"
        description: "Verify burst allowance works"

      - name: "Rate limit headers"
        type: "integration"
        description: "Verify X-RateLimit headers present"

  vulnerability_scanning:
    tests:
      - name: "Dependency vulnerabilities"
        type: "static"
        tool: "cargo-audit"
        frequency: "daily"

      - name: "Container vulnerabilities"
        type: "static"
        tool: "trivy"
        frequency: "on-build"

      - name: "Code vulnerabilities"
        type: "static"
        tool: "semgrep"
        frequency: "on-commit"
```

### 12.2 Penetration Testing Checklist

```markdown
# LLM-Simulator Penetration Testing Checklist

## Authentication Security
- [ ] Test API key brute forcing
- [ ] Test weak API key generation
- [ ] Test session fixation
- [ ] Test session hijacking
- [ ] Test JWT token manipulation
- [ ] Test credential stuffing
- [ ] Test authentication bypass

## Authorization Security
- [ ] Test horizontal privilege escalation
- [ ] Test vertical privilege escalation
- [ ] Test IDOR (Insecure Direct Object Reference)
- [ ] Test missing authorization checks
- [ ] Test role-based access control bypass

## Input Validation
- [ ] Test JSON injection
- [ ] Test XXE (XML External Entity) - N/A for JSON-only API
- [ ] Test SSRF (Server-Side Request Forgery)
- [ ] Test path traversal
- [ ] Test command injection
- [ ] Test header injection

## Network Security
- [ ] Test TLS/SSL configuration (testssl.sh)
- [ ] Test for TLS downgrade attacks
- [ ] Test certificate validation
- [ ] Test for man-in-the-middle vulnerabilities

## Business Logic
- [ ] Test rate limit bypass
- [ ] Test resource exhaustion
- [ ] Test workflow bypasses
- [ ] Test race conditions

## Information Disclosure
- [ ] Test for sensitive data in responses
- [ ] Test for verbose error messages
- [ ] Test for information leakage in headers
- [ ] Test for API key exposure in logs

## Denial of Service
- [ ] Test request flooding
- [ ] Test slowloris attacks
- [ ] Test resource exhaustion
- [ ] Test algorithmic complexity attacks

## API Security
- [ ] Test mass assignment
- [ ] Test excessive data exposure
- [ ] Test lack of resources & rate limiting
- [ ] Test broken function level authorization
- [ ] Test security misconfiguration
```

### 12.3 Automated Security Testing

```bash
#!/bin/bash
# automated-security-tests.sh

set -e

echo "=== Running Automated Security Tests ==="

# 1. Unit Tests (with security focus)
echo "[1/7] Running unit tests..."
cargo test --all

# 2. Static Analysis
echo "[2/7] Running static analysis..."
cargo clippy -- -D warnings

# 3. Dependency Audit
echo "[3/7] Auditing dependencies..."
cargo audit

# 4. Secret Detection
echo "[4/7] Scanning for secrets..."
gitleaks detect --source . --verbose

# 5. Integration Tests (auth/authz)
echo "[5/7] Running integration tests..."
cargo test --test security_integration_tests

# 6. API Security Tests
echo "[6/7] Running API security tests..."
newman run tests/security-tests.postman_collection.json

# 7. Load Tests (rate limiting)
echo "[7/7] Running load tests..."
k6 run tests/load-test-rate-limits.js

echo "✅ All automated security tests passed"
```

---

## Summary and Recommendations

### Implementation Priority

**Phase 1: Core Security (MVP)**
1. ✅ API key authentication (format validation only)
2. ✅ RBAC for admin endpoints
3. ✅ TLS 1.3 encryption
4. ✅ Rate limiting
5. ✅ Audit logging (basic)

**Phase 2: Enhanced Security**
1. ⚠️ JWT authentication
2. ⚠️ mTLS support
3. ⚠️ Tamper-evident audit logging
4. ⚠️ Secret manager integration
5. ⚠️ Enhanced PII detection

**Phase 3: Enterprise Security**
1. ⬜ SIEM integration
2. ⬜ Advanced threat detection
3. ⬜ Automated compliance reporting
4. ⬜ Security orchestration

### Security Metrics to Track

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Failed Authentication Rate | <1% | >5% |
| Authorization Denial Rate | <2% | >10% |
| Rate Limit Exceeded Events | <5% | >15% |
| Audit Log Integrity Check | 100% | <100% |
| TLS Handshake Success Rate | >99.9% | <99% |
| Secret Rotation Compliance | 100% | <100% |
| Vulnerability SLA (Critical) | <24h | >24h |

### Conclusion

This security architecture provides enterprise-grade security for LLM-Simulator with:

1. **Defense in Depth**: Multiple layers of security controls
2. **Zero Trust**: All requests authenticated and authorized
3. **Compliance Ready**: SOC2, HIPAA, PCI-DSS control mappings
4. **Audit Trail**: Comprehensive, tamper-evident logging
5. **Secure by Default**: Security features enabled in default configuration
6. **Minimal Attack Surface**: Only essential endpoints exposed
7. **Secret Management**: Integration with enterprise secret managers
8. **Supply Chain Security**: SBOM generation, dependency scanning

The architecture is designed to be production-ready for regulated environments while maintaining flexibility for different deployment scenarios.

---

**Document Control**:
- **Version**: 1.0.0
- **Last Updated**: 2025-11-26
- **Next Review**: 2026-02-26
- **Classification**: Internal - Security Architecture
- **Distribution**: Security Team, Engineering Leads, Compliance Officer
