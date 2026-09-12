from src.utils.metrics import compute_classification_metrics


def test_compute_metrics_perfect():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]
    m = compute_classification_metrics(y_true, y_pred)

    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["safety_audit"]["no_ppe_critical_fn_count"] == 0
    assert m["safety_audit"]["no_ppe_critical_fn_rate"] == 0.0


def test_safety_critical_false_negative_detection():
    # True: NO_PPE (2), Pred: FULL_PPE (0) -> Critical safety violation
    y_true = [2, 2, 1, 0]
    y_pred = [0, 2, 1, 0]
    m = compute_classification_metrics(y_true, y_pred)

    assert m["safety_audit"]["no_ppe_critical_fn_count"] == 1
    assert m["safety_audit"]["no_ppe_critical_fn_rate"] == 0.5
