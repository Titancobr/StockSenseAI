import unittest

from app import app, feature_columns


class StockSenseAppTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_pages_and_health(self):
        for path in ["/", "/guide", "/analysis", "/health"]:
            self.assertEqual(self.client.get(path).status_code, 200)

    def test_analysis_page_includes_missing_value_guidance(self):
        response = self.client.get("/analysis")
        self.assertIn(b"Do not leave any field empty", response.data)
        self.assertIn(b"enter <strong>100</strong> for Int Coverage", response.data)

    def test_prediction_uses_all_model_features(self):
        payload = {column: "10" for column in feature_columns}
        response = self.client.post("/analysis", data=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Stock Strength", response.data)
        self.assertIn(b"Exit or re-evaluate immediately", response.data)

    def test_invalid_input_is_rejected(self):
        payload = {column: "10" for column in feature_columns}
        payload["company_pe"] = ""
        response = self.client.post("/analysis", data=payload)
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
