# Reproducibility Specification & Audit Boundary

## 1. Reproducibility Architecture

SiteSafe Vision is engineered to guarantee deterministic, auditable reproduction from an empty environment without relying on hidden local state.

### 1.1 Controlled Random Seeds
All pseudo-random number generators are seeded with a globally defined seed (`SEED = 42`) specified in `configs/config.yaml`:
- `PYTHONHASHSEED = "42"`
- `random.seed(42)`
- `numpy.random.seed(42)`
- `torch.manual_seed(42)`
- `torch.cuda.manual_seed_all(42)`
- PyTorch DataLoader worker initialization seeded via `seed_worker(worker_id)` [^pytorch_randomness].

### 1.2 Deterministic Algorithms Flag
PyTorch is configured with:
```python
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
According to PyTorch documentation, this forces deterministic algorithm selections in convolution and pooling operations where available [^pytorch_randomness].

---

## 2. Clean Machine Reproduction Guide

### Linux / macOS (POSIX)
```bash
git clone <repository_url> sitesafe-vision
cd sitesafe-vision
chmod +x scripts/bootstrap.sh
./scripts/bootstrap.sh
```

### Windows (PowerShell)
```powershell
git clone <repository_url> sitesafe-vision
cd sitesafe-vision
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\bootstrap.ps1
```

---

## 3. DVC Pipeline Execution

The complete 10-stage pipeline is captured in `dvc.yaml` and locked in `dvc.lock` [^dvc_doc]:
```bash
# Pull versioned dataset and models
dvc pull

# Reproduce pipeline from source code and configs
dvc repro
```

---

## 4. Hardware & Software Reproducibility Boundary

> [!NOTE]
> **Reproducibility Boundary Disclosure**:
> As documented by PyTorch [^pytorch_randomness], bit-for-bit floating-point parity across distinct hardware architectures (e.g. Intel x86_64 vs. ARM64 vs. CUDA GPUs with differing Streaming Multiprocessor compute capabilities) is constrained by IEEE 754 non-associative atomic addition and SIMD hardware optimizations.
> 
> **Guaranteed Parity**:
> - 100% identical dataset ingestion, bounding box extraction, and split manifests across all operating systems.
> - 100% identical evaluation metrics within the same Python 3.11 / PyTorch 2.13.x CPU runtime.

---

## 5. References & Citations

[^pytorch_randomness]: PyTorch Official Documentation on Reproducibility and Randomness. URL: [https://pytorch.org/docs/stable/notes/randomness.html](https://pytorch.org/docs/stable/notes/randomness.html)
[^dvc_doc]: DVC (Data Version Control) Data and Pipeline Reproducibility Documentation. URL: [https://dvc.org/doc](https://dvc.org/doc)
