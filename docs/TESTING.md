# Automated Testing & Verification Strategy

## 1. Testing Philosophy

The SiteSafe Vision test suite ensures behavioral determinism, API robustness, and model reliability across local and CI environments [^pytest_doc].

```
tests/
├── conftest.py               # Global pytest environment setup & path injector
├── fixtures/
│   ├── __init__.py
│   └── create_fixtures.py    # Deterministic in-memory JPEG fixtures & corrupt buffers
├── unit/
│   ├── test_architectures.py # ResNet50/MobileNetV3 shapes and layer freeze phases
│   ├── test_metrics.py       # Macro F1, Recall, and Critical False Negative math
│   ├── test_risk_engine.py   # Decision matrix and risk level thresholds
│   ├── test_schemas.py       # Pydantic input/output validation contracts
│   ├── test_seed.py          # Deterministic PRNG stability verification
│   └── test_transforms.py    # Preprocessing tensor shapes and normalization
└── integration/
    ├── test_api.py           # FastAPI TestClient endpoints (/health, /predict, etc.)
    └── test_pipeline.py      # Manifest integrity, zero leakage, artifact existence
```

---

## 2. Test Execution Commands

### Run Full Test Suite
```bash
pytest tests/unit tests/integration -v --durations=10
```

### Run Unit Tests Only
```bash
pytest tests/unit -v
```

### Run Integration & API Endpoint Tests
```bash
pytest tests/integration -v
```

---

## 3. Negative & Edge-Case Testing Matrix

| Scenario | Input Tested | Expected Behavior |
| :--- | :--- | :--- |
| **Corrupted Image** | Truncated / malformed byte buffer | HTTP 400 Bad Request with descriptive diagnostic |
| **Empty File** | 0-byte uploaded file | HTTP 400 Bad Request ("Empty file uploaded") |
| **Oversized Upload** | Payload $> 10\text{MB}$ | HTTP 413 Payload Too Large |
| **Tiny Image** | Image $< 16 \times 16\text{px}$ | HTTP 400 Bad Request ("Image dimensions too small") |
| **Missing Model** | File `production_model.pt` absent | HTTP 503 Service Unavailable ("Model unavailable") |

---

## 4. References & Citations

[^pytest_doc]: Pytest Automated Testing Framework Documentation. URL: [https://docs.pytest.org/en/stable/](https://docs.pytest.org/en/stable/)
