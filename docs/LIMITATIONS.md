# System Limitations, Assumptions & Safety Boundaries

## 1. Regulatory & Operational Scope

> [!CAUTION]
> **Primary Safety Boundary**:
> SiteSafe Vision is an **image-based AI screening aid**. It is **NOT**:
> - An autonomous safety compliance inspector.
> - A certified regulatory auditing mechanism (e.g. OSHA 1926 or ISO 45001 compliance tool) [^osha_standard].
> - A guaranteed safety barrier or interlocking device.
> - A worker-level multi-object detector.

---

## 2. Environmental & Technical Constraints

### 2.1 Resolution & Distance Thresholds
- Classification reliability degrades significantly on worker bounding boxes smaller than $32 \times 32\text{px}$.
- The system must be supplied with worker-centered crops or high-resolution images where workers occupy $\ge 15\%$ of frame height.

### 2.2 Adverse Lighting & Atmospheric Conditions
- The training distribution reflects daylight and overcast outdoor construction environments.
- Severe glare, deep shadows, night operations without floodlights, heavy rain, dust storms, and fog will impair confidence.

### 2.3 Partial Occlusion
- If a worker is carrying large equipment (e.g. drywall, pipes) or is situated behind scaffolding, mandatory PPE items (such as high-vis vests) may be physically obscured, leading to `PARTIAL_PPE` or `NO_PPE` classifications.

### 2.4 Scope of Protected Equipment
- The 3-class target system (`FULL_PPE`, `PARTIAL_PPE`, `NO_PPE`) specifically models **Hardhats** and **High-Visibility Safety Vests**.
- Specialized PPE such as fall arrest harnesses, respirators, welding helmets, hearing protection, and steel-toe shank verification are outside the scope of this baseline classifier.

---

## 3. Human-in-the-Loop Requirement

SiteSafe Vision is explicitly designed to support human safety supervisors by highlighting potential compliance anomalies. Final authorization for site access must always remain under human oversight.

---

## 4. References & Citations

[^osha_standard]: Occupational Safety and Health Administration (OSHA) Construction PPE Standard 1926.95. URL: [https://www.osha.gov/laws-regs/regulations/standardnumber/1926/1926.95](https://www.osha.gov/laws-regs/regulations/standardnumber/1926/1926.95)
