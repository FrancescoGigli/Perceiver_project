"""Verifica confini reali: autenticazione, fonti, guasti e azioni del modello."""
import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import app as tutor


class TutorContractTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(tutor.app, client=("127.0.0.1", 50123))
        tutor.sessions.clear()
        self.remote = patch.object(tutor, "REMOTE", False)
        self.lan = patch.object(tutor, "LAN", False)
        self.remote.start()
        self.lan.start()
        self.addCleanup(self.remote.stop)
        self.addCleanup(self.lan.stop)

    def question(self, **fields):
        return self.client.post("/api/chat", json={"message": "Da dove viene Q?", "scene": "qkv", "step": 0, **fields})

    def test_remote_health_and_generation_require_the_token(self):
        with patch.object(tutor, "REMOTE", True), patch.object(tutor, "TOKEN", "a-local-test-token"):
            self.assertEqual(self.client.get("/api/health").status_code, 401)
            self.assertEqual(self.question().status_code, 401)
            response = self.client.get("/api/health", headers={"X-Tutor-Token": "a-local-test-token"})
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["ok"])

    def test_cross_origin_generation_is_rejected_even_locally(self):
        response = self.client.post("/api/chat", json={"message": "Q?"}, headers={"Origin": "https://another-site.example"})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["error"]["code"], "ORIGIN_DENIED")

    def test_prompt_contains_actual_code_and_lesson_and_session_continues(self):
        captured = []

        def reply(path, payload=None, timeout=10):
            captured.append(payload)
            return {"message": {"content": json.dumps({"answer": "Q viene dai latenti.", "action": {"scene": "qkv", "step": 0}})}}

        with patch.object(tutor, "ollama_request", reply):
            first = self.question()
            self.assertEqual(first.status_code, 200)
            session = first.json()["session_id"]
            second = self.question(message="E K?", session_id=session)
            self.assertEqual(second.json()["session_id"], session)
        system = captured[0]["messages"][0]["content"]
        self.assertIn("q = self.q_proj(self.norm_attn(x_latent))", system)
        self.assertIn("domanda", system)
        self.assertIn("etichetta", system)
        self.assertFalse(captured[0]["think"])
        self.assertEqual(captured[1]["messages"][1]["content"], "Da dove viene Q?")
        self.assertEqual(first.json()["action"], {"scene": "qkv", "step": 0})

    def test_fabricated_model_actions_cannot_select_unknown_scenes(self):
        output = {"answer": "Q viene dai latenti.", "action": {"scene": "shell-command", "step": 999}}
        with patch.object(tutor, "ollama_request", return_value={"message": {"content": json.dumps(output)}}):
            response = self.question()
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["action"])

    def test_unknown_scene_is_rejected_without_calling_model(self):
        with patch.object(tutor, "ollama_request") as model:
            response = self.question(scene="invented")
            self.assertEqual(response.status_code, 422)
            model.assert_not_called()

    def test_down_model_is_an_error_not_a_fake_answer(self):
        with patch.object(tutor, "ollama_request", side_effect=tutor.TutorError(503, "MODEL_UNAVAILABLE", "Ollama non risponde.")):
            response = self.question()
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("answer", response.json())
        self.assertEqual(response.json()["error"]["code"], "MODEL_UNAVAILABLE")

    def test_plain_model_answer_is_preserved(self):
        with patch.object(tutor, "ollama_request", return_value={"message": {"content": "Q viene dai latenti, K e V dall'input."}}):
            response = self.question()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Q viene dai latenti, K e V dall'input.")

    def test_malformed_model_response_is_an_error(self):
        with patch.object(tutor, "ollama_request", return_value={"message": {"content": "{malformed JSON"}}):
            response = self.question()
        self.assertEqual(response.status_code, 502)

    def test_audio_format_and_size_are_checked_before_whisper(self):
        with patch.object(tutor, "transcribe_audio") as transcribe:
            response = self.client.post("/api/transcribe", content=b"text", headers={"Content-Type": "text/plain"})
            self.assertEqual(response.status_code, 415)
            response = self.client.post("/api/transcribe", content=b"", headers={"Content-Type": "audio/webm", "Content-Length": str(tutor.MAX_AUDIO + 1)})
            self.assertEqual(response.status_code, 413)
            transcribe.assert_not_called()

    def test_private_repository_files_are_not_served(self):
        self.assertEqual(self.client.get("/.git/config").status_code, 404)
        self.assertEqual(self.client.get("/progetto/data/private.txt").status_code, 404)
        self.assertEqual(self.client.get("/sito/slide/_copione/copione_orale.pdf").status_code, 404)
        self.assertEqual(self.client.get("/progetto/src/perceiver/attention.py").status_code, 200)

    def test_direct_public_clients_are_rejected(self):
        client = TestClient(tutor.app, client=("8.8.8.8", 50123))
        self.assertEqual(client.get("/api/status").status_code, 403)


if __name__ == "__main__":
    unittest.main()
