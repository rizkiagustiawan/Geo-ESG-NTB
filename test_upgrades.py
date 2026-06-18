"""Tests for research-backed upgrades: XGBoost, DL detection, Carbon Pricing."""

import pytest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python-gee-ai"))


# ─── XGBoost Model Tests ─────────────────────────────────────────

class TestXGBoostModel:
    def test_model_file_exists(self):
        model_path = os.path.join("python-gee-ai", "ml_models", "xgboost_biomass_model.joblib")
        assert os.path.exists(model_path), "XGBoost model not found. Run train_xgboost_model.py"

    def test_report_file_exists(self):
        report_path = os.path.join("python-gee-ai", "ml_models", "xgboost_validation_report.json")
        assert os.path.exists(report_path)

    def test_model_loads_and_predicts(self):
        import joblib
        model_path = os.path.join("python-gee-ai", "ml_models", "xgboost_biomass_model.joblib")
        if not os.path.exists(model_path):
            pytest.skip("XGBoost model not trained yet")
        model = joblib.load(model_path)
        features = np.array([[0.7, -14.0, -7.4, -8.0, -14.0]])
        pred = model.predict(features)
        assert pred[0] > 0, "Prediction should be positive"
        assert pred[0] < 500, "Prediction should be realistic (<500 Mg/ha)"

    def test_report_metrics_valid(self):
        import json
        report_path = os.path.join("python-gee-ai", "ml_models", "xgboost_validation_report.json")
        if not os.path.exists(report_path):
            pytest.skip("Report not generated yet")
        with open(report_path) as f:
            report = json.load(f)
        assert report["validation_metrics"]["R2"] > 0.85
        assert report["validation_metrics"]["RMSE_Mg_ha"] < 20

    def test_extractor_uses_xgboost(self):
        from extractor import estimate_biomass_carbon
        b, c = estimate_biomass_carbon(0.7, -14.0, -7.4, -8.0, -14.0)
        assert b > 0
        assert c == round(b * 0.46, 2)


# ─── DL Tree Crown Detection Tests ───────────────────────────────

class TestDLTreeCrownDetection:
    def test_dl_module_importable(self):
        from tree_crown_dl import DeepTreeCrownDetector
        assert DeepTreeCrownDetector is not None

    def test_detector_creates_output(self):
        from tree_crown_detector import TreeCrownDetector
        d = TreeCrownDetector()
        img_path = d.generate_synthetic_imagery("TestDL", density=0.8)
        assert os.path.exists(img_path)
        count, result = d.detect_tree_crowns(img_path, "TestDL")
        assert count > 0
        assert result is not None
        assert os.path.exists(result)

    def test_detector_with_low_density(self):
        from tree_crown_detector import TreeCrownDetector
        d = TreeCrownDetector()
        img_path = d.generate_synthetic_imagery("TestLowDensity", density=0.2)
        count, _ = d.detect_tree_crowns(img_path, "TestLowDensity")
        assert count >= 0  # May be 0 for very sparse forests


# ─── Carbon Pricing Tests ────────────────────────────────────────

class TestCarbonPricing:
    def test_module_importable(self):
        from carbon_pricing import calculate_carbon_value, format_carbon_report
        assert calculate_carbon_value is not None

    def test_basic_calculation(self):
        from carbon_pricing import calculate_carbon_value
        result = calculate_carbon_value(
            estimated_carbon_mg_ha=69.23,
            area_ha=1500,
            audit_status="AUDIT_PASS",
        )
        assert result["carbon_stock_mg_c_ha"] == 69.23
        assert result["total_co2e"] > 0
        assert result["total_value_rp"] > 0
        assert result["integrity_premium"] == 1.0

    def test_failed_audit_zero_value(self):
        from carbon_pricing import calculate_carbon_value
        result = calculate_carbon_value(
            estimated_carbon_mg_ha=69.23,
            area_ha=1500,
            audit_status="AUDIT_FAIL",
        )
        assert result["integrity_premium"] == 0.0
        assert result["total_value_rp"] == 0

    def test_warning_audit_discounted(self):
        from carbon_pricing import calculate_carbon_value
        result = calculate_carbon_value(
            estimated_carbon_mg_ha=69.23,
            area_ha=1500,
            audit_status="AUDIT_WARN",
        )
        assert result["integrity_premium"] == 0.7
        assert result["effective_price_rp_per_ton"] < result["idxcarbon_price_rp_per_ton"]

    def test_co2e_conversion(self):
        from carbon_pricing import calculate_carbon_value, C_TO_CO2E
        result = calculate_carbon_value(100.0, 1.0, "AUDIT_PASS")
        expected_co2e = 100.0 * C_TO_CO2E
        assert abs(result["carbon_stock_ton_co2e_ha"] - expected_co2e) < 0.1

    def test_format_report(self):
        from carbon_pricing import calculate_carbon_value, format_carbon_report
        val = calculate_carbon_value(69.23, 1500, "AUDIT_PASS")
        md = format_carbon_report(val, "Sumbawa Barat")
        assert "IDXCarbon" in md
        assert "Sumbawa Barat" in md or "Rp" in md


# ─── Integration Test ────────────────────────────────────────────

class TestIntegration:
    def test_api_server_with_upgrades(self):
        from fastapi.testclient import TestClient
        from api_server import app
        client = TestClient(app)
        response = client.get("/api/health")
        assert response.status_code == 200