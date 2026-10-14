"""Voice provider contracts: opt-in cloud voice and explicit local voices."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import app as tutor


class TutorVoiceTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(tutor.app, client=("127.0.0.1", 50123))
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.runtime = Path(directory.name)
        for name in ("python.exe", "helper.py", "kokoro-v1.0.onnx", "voices-v1.0.bin"):
            (self.runtime / name).touch()
        for name, value in (("REMOTE", False), ("LAN", False), ("TTS_RUNTIME", self.runtime),
                            ("TTS_PYTHON", self.runtime / "python.exe"), ("TTS_HELPER", self.runtime / "helper.py")):
            replacement = patch.object(tutor, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)
        self.cloud = patch.object(tutor.elevenlabs_tts, "status", return_value={
            "enabled": True, "available": True, "engine": "ElevenLabs · italiano", "reason": None})
        self.cloud.start()
        self.addCleanup(self.cloud.stop)

    def test_available_cloud_voice_is_additive_and_default_without_account_fields(self):
        result = tutor.voice_status()
        self.assertEqual(result["voices"], ["alice", "if_sara", "im_nicola"])
        self.assertEqual(result["defaultVoice"], "alice")
        self.assertEqual(result["engine"], "ElevenLabs · italiano")
        self.assertNotIn("cli", result)
        self.assertNotIn("account", result)

    def test_omitted_voice_uses_alice_and_mp3_header(self):
        with patch.object(tutor, "synthesize_voice", return_value=b"mp3") as generate:
            response = self.client.post("/api/tts", json={"text": "Spiega il mio codice."})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/mpeg")
        self.assertEqual(response.headers["X-Voice-Engine"], "ElevenLabs")
        generate.assert_called_once_with("Spiega il mio codice.", "alice")

    def test_explicit_local_voices_keep_wav_contract_even_with_cloud_enabled(self):
        for voice in ("if_sara", "im_nicola"):
            with self.subTest(voice=voice), patch.object(tutor, "synthesize_voice", return_value=b"RIFF") as generate:
                response = self.client.post("/api/tts", json={"text": "Test", "voice": voice})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["content-type"], "audio/wav")
                self.assertEqual(response.headers["X-Voice-Engine"], "Kokoro")
                generate.assert_called_once_with("Test", voice)

    def test_unconfigured_cloud_keeps_original_sara_default(self):
        with patch.object(tutor.elevenlabs_tts, "status", return_value={"enabled": False, "available": False}), \
                patch.object(tutor, "synthesize_voice", return_value=b"RIFF") as generate:
            response = self.client.post("/api/tts", json={"text": "Test"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        generate.assert_called_once_with("Test", "if_sara")

    def test_quota_error_is_clear_and_does_not_fall_back_or_retry(self):
        failure = tutor.elevenlabs_tts.ElevenLabsError("ELEVENLABS_QUOTA", "Non bastano i caratteri gratuiti.")
        with patch.object(tutor.elevenlabs_tts, "synthesize", side_effect=failure) as generate, \
                patch.object(tutor.subprocess, "run") as local:
            response = self.client.post("/api/tts", json={"text": "Test", "voice": "alice"})
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json()["error"]["code"], "ELEVENLABS_QUOTA")
        self.assertEqual(generate.call_count, 1)
        local.assert_not_called()

    def test_cloud_returns_created_audio_and_temporary_files_are_removed(self):
        paths = []
        def generate(text, output):
            paths.append(output)
            output.write_bytes(b"mp3-result")
        with patch.object(tutor.elevenlabs_tts, "synthesize", side_effect=generate):
            self.assertEqual(tutor.synthesize_voice("Test", "alice"), b"mp3-result")
        self.assertFalse(paths[0].exists())

    def test_remote_auth_and_origin_are_checked_before_provider(self):
        with patch.object(tutor, "REMOTE", True), patch.object(tutor, "TOKEN", "test-token"), \
                patch.object(tutor.elevenlabs_tts, "synthesize") as generate:
            denied = self.client.post("/api/tts", json={"text": "Test", "voice": "alice"})
            self.assertEqual(denied.status_code, 401)
            cross_origin = self.client.post("/api/tts", json={"text": "Test", "voice": "alice"},
                                           headers={"X-Tutor-Token": "test-token", "Origin": "https://other.example"})
            self.assertEqual(cross_origin.status_code, 403)
            generate.assert_not_called()

    def test_invalid_voice_does_not_reach_provider_and_local_missing_is_reported(self):
        with patch.object(tutor, "synthesize_voice") as generate:
            response = self.client.post("/api/tts", json={"text": "Test", "voice": "arbitrary-voice"})
            self.assertEqual(response.status_code, 422)
            (self.runtime / "kokoro-v1.0.onnx").unlink()
            response = self.client.post("/api/tts", json={"text": "Test", "voice": "im_nicola"})
            self.assertEqual(response.status_code, 503)
            generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
