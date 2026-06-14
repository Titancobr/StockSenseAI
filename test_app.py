import unittest
from uuid import uuid4

from app import app, feature_columns


class StockSenseAppTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def signup(self):
        email = f"test-{uuid4().hex}@example.com"
        response = self.client.post(
            "/signup",
            data={"name": "Test User", "email": email, "password": "password123"},
        )
        self.assertEqual(response.status_code, 302)
        return email

    def test_pages_and_health(self):
        for path in ["/", "/guide", "/login", "/signup", "/health"]:
            self.assertEqual(self.client.get(path).status_code, 200)

    def test_analysis_page_includes_missing_value_guidance(self):
        self.signup()
        response = self.client.get("/analysis")
        self.assertIn(b"Do not leave any field empty", response.data)
        self.assertIn(b"enter <strong>100</strong> for Int Coverage", response.data)

    def test_prediction_uses_all_model_features(self):
        self.signup()
        payload = {column: "10" for column in feature_columns}
        payload["stock_name"] = "Demo Stock"
        response = self.client.post("/analysis", data=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Stock Strength", response.data)
        self.assertIn(b"Exit or re-evaluate immediately", response.data)
        self.assertIn(b"Demo Stock", self.client.get("/history").data)

    def test_invalid_input_is_rejected(self):
        self.signup()
        payload = {column: "10" for column in feature_columns}
        payload["company_pe"] = ""
        response = self.client.post("/analysis", data=payload)
        self.assertEqual(response.status_code, 400)

    def test_analysis_requires_login(self):
        response = self.client.get("/analysis")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers["Location"])


if __name__ == "__main__":
    unittest.main()
