
import sys
from pathlib import Path

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from risk_engine.feature_aggregator import aggregate_features
from risk_engine.anomaly_detector import detect_anomalies
from risk_engine.risk_engine import calculate_risk


def main():
    print("=" * 70)
    print("SENTINAL COMPLETE RISK PIPELINE TEST")
    print("=" * 70)

    attendance = {
        "session_id": "test-session-001",
        "total_tracked": 10,
        "staff": 2,
        "beneficiary": 7,
        "unknown": 1,
        "duration_seconds": 600,
        "observation_count": 120,
    }

    project = {
        "project_id": "project-001",
        "status": "active",
        "risk_level": "medium",
        "total_inspections": 10,
        "completed_inspections": 7,
        "pending_inspections": 3,
        "high_risk_findings": 1,
    }

    inspections = [
        {
            "inspection_id": "inspection-001",
            "risk_level": "medium",
            "findings": [
                {
                    "title": "Attendance mismatch",
                    "status": "open",
                }
            ],
        },
        {
            "inspection_id": "inspection-002",
            "risk_level": "high",
            "findings": [
                {
                    "title": "Attendance mismatch",
                    "status": "open",
                }
            ],
        },
    ]

    print()
    print("[1/3] Aggregating features...")

    features = aggregate_features(
        attendance=attendance,
        project=project,
        inspections=inspections,
    )

    print("Feature aggregation: SUCCESS")

    print()
    print("[2/3] Detecting anomalies...")

    anomaly_result = detect_anomalies(features)

    print("Anomaly detection: SUCCESS")

    print()
    print("[3/3] Calculating risk...")

    risk_result = calculate_risk(
        features=features,
        anomaly_result=anomaly_result,
    )

    print("Risk calculation: SUCCESS")

    print()
    print("-" * 70)
    print("FINAL RISK RESULT")
    print("-" * 70)

    print(f"Risk score : {risk_result['risk_score']}")
    print(f"Risk level : {risk_result['risk_level']}")
    print(f"Anomalies  : {risk_result['anomaly_count']}")

    print()
    print("Component scores:")

    for name, score in risk_result["component_scores"].items():
        print(f"  {name:<12} {score}")

    print()
    print("Reasons:")

    for reason in risk_result["reasons"]:
        print(f"  - {reason}")

    assert isinstance(features, dict)
    assert isinstance(anomaly_result, dict)
    assert isinstance(risk_result, dict)

    assert 0 <= risk_result["risk_score"] <= 100

    assert risk_result["risk_level"] in {
        "low",
        "medium",
        "high",
        "critical",
    }

    assert isinstance(risk_result["reasons"], list)

    print()
    print("=" * 70)
    print("COMPLETE RISK PIPELINE TEST SUCCESSFUL")
    print("=" * 70)


if __name__ == "__main__":
    main()

