import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import app

def test_full_flow():
    client = app.test_client()

    print("--- Test 1: Sample Images (Healthy) ---")
    resp = client.get("/api/sample-images?category=Healthy")
    assert resp.status_code == 200, f"Status {resp.status_code}"
    data = json.loads(resp.data)
    samples = data.get("samples", [])
    print(f"Retrieved {len(samples)} healthy samples directly from data folder.")
    assert len(samples) > 0, "No samples returned"
    first_s = samples[0]
    print(f"Sample 0: {first_s['name']} | Path: {first_s['path']} | Rct: {first_s.get('rct')} | Ipa: {first_s.get('ipa')}")

    print("\n--- Test 2: Sample Images (Abnormal) ---")
    resp = client.get("/api/sample-images?category=Abnormal")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    ab_samples = data.get("samples", [])
    print(f"Retrieved {len(ab_samples)} abnormal samples directly from data folder.")
    assert len(ab_samples) > 0

    print("\n--- Test 3: Image Preview ---")
    resp = client.get(f"/api/image-preview?path={first_s['path']}")
    assert resp.status_code == 200, f"Preview failed: {resp.status_code}"
    assert resp.content_type == "image/png"
    print(f"Preview image successfully loaded: {len(resp.data)} bytes")

    print("\n--- Test 4: Analyze with Preset Path ---")
    resp = client.post("/api/analyze", data={
        "preset_path": first_s["path"],
        "cell_condition": "Healthy",
        "noise_level": "0.04",
        "notes": "Test run from dataset"
    })
    assert resp.status_code == 200, f"Analyze failed: {resp.status_code}, data: {resp.data}"
    res_data = json.loads(resp.data)
    print(f"Analysis Prediction: {res_data['prediction']} | Confidence: {res_data['confidence']}% | Risk Score: {res_data['risk_score']}")
    print(f"Data Source Label: {res_data.get('sensor_data_source')}")
    assert res_data.get("is_dataset_record") is True
    assert "Dataset File" in res_data.get("sensor_data_source")
    assert res_data["prediction"] in ["Healthy", "Abnormal"]

    print("\n--- Test 5: History Audit Log ---")
    resp = client.get("/api/history")
    assert resp.status_code == 200
    hist_data = json.loads(resp.data)
    records = hist_data.get("records", [])
    print(f"Total history records logged: {len(records)}")
    assert len(records) >= 1

    print("\nALL VERIFICATIONS PASSED WITH ZERO ERRORS!")


if __name__ == "__main__":
    test_full_flow()
