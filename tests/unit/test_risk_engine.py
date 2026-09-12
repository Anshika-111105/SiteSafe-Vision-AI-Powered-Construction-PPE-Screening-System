from app.risk import evaluate_risk_and_recommendation


def test_risk_mapping_no_ppe():
    risk_level, rec = evaluate_risk_and_recommendation("NO_PPE", 0.95)
    assert risk_level == "HIGH"
    assert "CRITICAL" in rec


def test_risk_mapping_partial_ppe():
    risk_level, rec = evaluate_risk_and_recommendation("PARTIAL_PPE", 0.88)
    assert risk_level == "MEDIUM"
    assert "CAUTION" in rec


def test_risk_mapping_full_ppe():
    risk_level, rec = evaluate_risk_and_recommendation("FULL_PPE", 0.92)
    assert risk_level == "LOW"
    assert "COMPLIANT" in rec


def test_risk_mapping_low_confidence():
    risk_level, rec = evaluate_risk_and_recommendation("FULL_PPE", 0.55)
    assert risk_level == "MEDIUM"
    assert "low" in rec.lower()
