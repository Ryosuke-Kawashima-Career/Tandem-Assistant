"""
This file tests the server.py file to connect the frontend and the backend.
"""

from src.server import app
import unittest

class TestEchoSphereServer(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_landing_page(self):
        response = self.app.get("/")
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")
        self.assertTrue("EchoSphere" in content or "Tandem" in content)

    def test_health_check(self):
        response = self.app.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "online")
        self.assertEqual(data["service"], "EchoSphere Tandem Co-Teacher")

    def test_session_lifecycle_and_turn_processing(self):
        # 1. Start session
        start_res = self.app.post("/api/session/start")
        self.assertEqual(start_res.status_code, 200)
        self.assertTrue(start_res.get_json()["success"])

        # 2. Process turn
        turn_res = self.app.post(
            "/api/session/turn",
            json={"speaker_id": "Kenji", "text": "一期一会ですね！", "language": "ja"}
        )
        self.assertEqual(turn_res.status_code, 200)
        turn_data = turn_res.get_json()
        self.assertTrue(turn_data["success"])
        self.assertIn("subtitles", turn_data["result"])
        self.assertIn("idiom_card", turn_data["result"])

        # 3. Stop session
        stop_res = self.app.post("/api/session/stop")
        self.assertEqual(stop_res.status_code, 200)
        self.assertTrue(stop_res.get_json()["success"])

if __name__ == '__main__':
    unittest.main()
