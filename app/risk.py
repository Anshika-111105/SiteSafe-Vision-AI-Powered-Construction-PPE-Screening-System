from typing import Tuple, Dict


def evaluate_risk_and_recommendation(
    prediction: str,
    confidence: float,
) -> Tuple[str, str]:
    """
    Computes risk level and standardized safety recommendation.

    Args:
        prediction: Predicted class ('FULL_PPE', 'PARTIAL_PPE', 'NO_PPE')
        confidence: Prediction confidence score between 0.0 and 1.0

    Returns:
        Tuple of (risk_level, recommendation)
    """
    if prediction == "NO_PPE":
        risk_level = "HIGH"
        rec = "CRITICAL: Mandatory PPE missing. Immediate site entry restriction and safety supervisor intervention required."
    elif prediction == "PARTIAL_PPE":
        risk_level = "MEDIUM"
        rec = "CAUTION: Incomplete PPE detected (missing helmet or high-vis vest). Manual safety inspection required before site entry."
    elif prediction == "FULL_PPE":
        if confidence < 0.70:
            risk_level = "MEDIUM"
            rec = "CAUTION: Full PPE indicated but classification confidence is low. Manual physical verification recommended."
        else:
            risk_level = "LOW"
            rec = "COMPLIANT: Standard PPE detected (Hardhat and High-Visibility Vest). Authorized for site entry under standard safety protocols."
    else:
        risk_level = "HIGH"
        rec = "UNKNOWN STATUS: Unable to establish compliance state. Physical safety inspection mandatory."

    if confidence < 0.65 and "low" not in rec:
        rec += " (Screening confidence below standard threshold; on-site confirmation advised.)"

    return risk_level, rec
