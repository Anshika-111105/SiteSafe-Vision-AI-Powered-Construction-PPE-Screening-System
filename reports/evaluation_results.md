# SiteSafe Vision: Model Evaluation & Benchmark Report

- **Champion Model**: `mobilenet_v3_large`
- **Evaluation Split**: `test` (Untouched during training and selection)
- **Test Set Size**: `27` worker crops
- **Single Image Latency**: `26.98 ms`

## Overall Performance

- **Accuracy**: `88.89%`
- **Macro F1**: `0.8931`
- **Weighted F1**: `0.8863`
- **Macro Recall**: `0.9167`

## Safety-Critical Audit

- **Unsafe Class Minimum Recall**: `0.7500`
- **Critical NO_PPE False Negatives**: `0` (Rate: `0.00%`)
- **PARTIAL_PPE False Negatives**: `2` (Rate: `16.67%`)

## Per-Class Breakdown

| Class | Precision | Recall | F1 Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| `FULL_PPE` | 0.8000 | 1.0000 | 0.8889 | 8 |
| `PARTIAL_PPE` | 1.0000 | 0.7500 | 0.8571 | 12 |
| `NO_PPE` | 0.8750 | 1.0000 | 0.9333 | 7 |

## Confusion Matrix

Rows = Ground Truth, Columns = Prediction (`[FULL_PPE, PARTIAL_PPE, NO_PPE]`)

```
[8, 0, 0]
[2, 9, 1]
[0, 0, 7]
```

## Error Analysis Summary

- **Total Misclassified Samples**: `3`
- Detailed error trace saved to [`reports/misclassified_samples.csv`](file:///C:/Users/hp/OneDrive/Desktop/Code/Site Safe Vision/reports/misclassified_samples.csv)
