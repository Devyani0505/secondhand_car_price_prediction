import unittest
from unittest.mock import patch

import pandas as pd

import app as app_module


class DummyPipeline:
    def predict(self, df):
        assert isinstance(df, pd.DataFrame)
        return [12.0]


class AutoValueAppTests(unittest.TestCase):
    def setUp(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def valid_payload(self):
        return {
            "make": "Maruti",
            "model": "Alto",
            "variant": "VXI",
            "city": "Hyderabad",
            "year": 2018,
            "kms": 40000,
            "mileage": 18.0,
            "owners": 1,
            "fuel": "Petrol",
            "transmission": "Manual",
            "color": "White",
            "rc_status": "Active",
            "loan_clearance": "Cleared",
            "accident_history": "Not Reported",
            "insurance_claims": "Single",
            "service_records": "Partial",
            "body_panels": "Average",
            "rust": "Average",
            "glass_condition": "Average",
            "lights_condition": "Average",
            "tyres": "Average",
            "seats_condition": "Average",
            "electronics_condition": "Average",
            "smell": "Average",
            "engine_condition": "Average",
            "transmission_condition": "Average",
            "brakes_condition": "Average",
            "steering_condition": "Average",
            "suspension_condition": "Average",
            "battery_condition": "Average",
            "ac_cooling": "Average",
            "rattling_noises": "Occasional",
            "engine_start": "Yes",
            "idle_vibration": "Average",
            "acceleration_condition": "Average",
            "brakes_drive": "Average",
            "airbags_present": "Yes",
            "safety_systems": "Seat belts, ABS, EBD",
            "airbags_count": 2,
            "market_trend_index": 1.05,
        }

    def test_pages_render(self):
        for route in ("/", "/valuation", "/emi", "/dealers"):
            response = self.client.get(route)
            self.assertEqual(response.status_code, 200)

    def test_predict_requires_model_pipeline(self):
        with patch.object(app_module, "pipeline_model", None):
            response = self.client.post("/predict", json=self.valid_payload())

        body = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertFalse(body["success"])
        self.assertIn("Model pipeline not found", body["error"])

    def test_predict_rejects_missing_required_fields(self):
        payload = self.valid_payload()
        payload["make"] = ""

        with patch.object(app_module, "pipeline_model", DummyPipeline()):
            response = self.client.post("/predict", json=payload)

        body = response.get_json()
        self.assertFalse(body["success"])
        self.assertIn("Missing required fields", body["error"])
        self.assertIn("make", body["error"])

    def test_predict_rejects_invalid_year(self):
        payload = self.valid_payload()
        payload["year"] = 1985

        with patch.object(app_module, "pipeline_model", DummyPipeline()):
            response = self.client.post("/predict", json=payload)

        body = response.get_json()
        self.assertFalse(body["success"])
        self.assertEqual(body["error"], "Enter a valid manufacturing year.")

    def test_predict_rejects_non_positive_kms(self):
        payload = self.valid_payload()
        payload["kms"] = 0

        with patch.object(app_module, "pipeline_model", DummyPipeline()):
            response = self.client.post("/predict", json=payload)

        body = response.get_json()
        self.assertFalse(body["success"])
        self.assertEqual(body["error"], "Kilometers driven must be greater than 0.")

    def test_predict_returns_price_and_range(self):
        with patch.object(app_module, "pipeline_model", DummyPipeline()):
            response = self.client.post("/predict", json=self.valid_payload())

        body = response.get_json()
        self.assertTrue(body["success"])
        self.assertGreaterEqual(body["predicted_price"], 10000)
        self.assertGreaterEqual(body["price_low"], 10000)
        self.assertGreater(body["price_high"], body["price_low"])
        self.assertEqual(body["market_trend_source"], "manual")
        self.assertAlmostEqual(body["market_trend_multiplier"], 1.05, places=2)

    def test_predict_uses_auto_market_trends_when_manual_override_missing(self):
        payload = self.valid_payload()
        payload["market_trend_index"] = ""

        with patch.object(app_module, "pipeline_model", DummyPipeline()):
            with patch.object(
                app_module,
                "_load_market_trends",
                return_value={
                    "last_updated_utc": "2026-04-23T00:00:00Z",
                    "overall_index": 1.1,
                    "city_adjustments": {"hyderabad": 1.0},
                    "make_adjustments": {"maruti": 1.0},
                    "fuel_adjustments": {"petrol": 1.0},
                },
            ):
                response = self.client.post("/predict", json=payload)

        body = response.get_json()
        self.assertTrue(body["success"])
        self.assertEqual(body["market_trend_source"], "auto")
        self.assertAlmostEqual(body["market_trend_multiplier"], 1.1, places=2)

    def test_dealer_prices_requires_make_or_model(self):
        response = self.client.post("/dealer-prices", json={})
        body = response.get_json()

        self.assertFalse(body["success"])
        self.assertIn("Enter at least Make or Model", body["error"])

    def test_dealer_prices_returns_results(self):
        fake_results = {
            "CARS24": {
                "url": "https://example.com",
                "price": 450000,
                "sample_count": 3,
                "error": None,
            },
            "Spinny": {
                "url": "https://example.com",
                "price": None,
                "sample_count": 0,
                "error": "timed out",
            },
        }

        with patch.object(app_module, "_fetch_dealer_prices", return_value=("Maruti Alto Hyderabad", fake_results)):
            response = self.client.post("/dealer-prices", json={"make": "Maruti", "model": "Alto"})

        body = response.get_json()
        self.assertTrue(body["success"])
        self.assertEqual(body["query"], "Maruti Alto Hyderabad")
        self.assertEqual(body["found_count"], 1)
        self.assertEqual(body["results"]["CARS24"]["price"], 450000)


if __name__ == "__main__":
    unittest.main()
