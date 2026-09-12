# Security & Privacy Policy Specification

## 1. Security Architecture & Threat Model

SiteSafe Vision is engineered to adhere to modern cybersecurity and container security standards [^docker_security].

### 1.1 Non-Root Container Execution
The Docker runtime strictly avoids root execution:
- Container runs under an unprivileged user `appuser:appgroup` (UID `10001`).
- Filesystem permissions are locked down with read-only application layers where feasible [^docker_security].

### 1.2 Image Ingestion & Input Sanitization
- **Payload Limit**: Maximum upload payload size capped at 10MB via Starlette middleware.
- **MIME & Magic Byte Verification**: Uploads are verified using PIL decoding and binary magic byte headers.
- **Decompression Bomb Defense**: PIL decompression bomb protections enforced to prevent Denial of Service (DoS) memory exhaustion attacks [^pillow_security].
- **Dimension Guards**: Images smaller than $16 \times 16\text{px}$ or containing invalid aspect ratios are rejected with structured HTTP 400 Bad Request responses.

---

## 2. Privacy & Data Governance

- **Zero Image Retention Policy**: Uploaded worker images are processed strictly in volatile memory (RAM) and immediately garbage-collected upon response dispatch.
- **No Biometric Identification**: The model performs apparel compliance screening only and does not perform facial recognition or individual identity tracking.
- **Structured Telemetry**: Access logs record request IDs, processing latency, and compliance prediction without logging raw image bytes or personal identifiable information (PII).

---

## 3. Secret Management & Git Hygiene

- `.env` files and cloud credentials are excluded from version control via `.gitignore`.
- Configuration values in `configs/config.yaml` contain zero API keys or credentials.
- Local and CI runs use environment variables for external authentication.

---

## 4. References & Citations

[^docker_security]: Docker OCI Security Best Practices Guide. URL: [https://docs.docker.com/develop/security-best-practices/](https://docs.docker.com/develop/security-best-practices/)
[^pillow_security]: Pillow Security and Decompression Bomb Protection Guide. URL: [https://pillow.readthedocs.io/en/stable/releasenotes/index.html](https://pillow.readthedocs.io/en/stable/releasenotes/index.html)
