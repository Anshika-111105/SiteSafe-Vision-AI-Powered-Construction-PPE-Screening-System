# SiteSafe Vision: Data Leakage & Split Audit Report

- **Audit Status**: `PASSED`
- **Audit Timestamp (UTC)**: `2026-09-12T07:09:03.207605+00:00`
- **Grouping Strategy**: `group_by_source_image_id` (Crops from same scene strictly coupled)
- **Random Seed**: `42`

## Split Summary

| Split | Scene Groups | Worker Samples | FULL_PPE | PARTIAL_PPE | NO_PPE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `train` | 84 | 134 | 50 | 49 | 35 |
| `val` | 18 | 24 | 7 | 9 | 8 |
| `test` | 18 | 27 | 8 | 12 | 7 |

## Leakage Verification Results

- Train-Val Disjoint Source Images: `PASS`
- Train-Test Disjoint Source Images: `PASS`
- Val-Test Disjoint Source Images: `PASS`
- Cross-Split SHA-256 Hash Collisions: `0`

✅ **Zero cross-split data leakage detected. Strict scene grouping verified.**
