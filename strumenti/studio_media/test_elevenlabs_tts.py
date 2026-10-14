"""Mock the CLI: verify Free limits and cached audio without generating speech."""
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

import elevenlabs_tts as tts

MP3 = b"\xff\xfb\x90\x00" + b"\x00" * 1024


class ElevenLabsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config_path = self.root / "runtime.json"
        self.cli = self.root / "elevenlabs.exe"
        self.cli.touch()
        self.config = {"enabled": True, "cli": str(self.cli), "voice": "Alice-id", "model": "eleven_v4"}
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        self.state = self.root / "state"
        self.state.mkdir()
        self.account = {"tier": "free", "used": 0, "limit": 10000, "reset": int(time.time()) + 86400, "can": False, "allowed": False}
        self.budget = {"version": 1, "period": self.account["reset"], "reservedCharacters": 526}
        self.save_budget()
        for name, value in (("CONFIG", self.config_path), ("STATE", self.state)):
            replacement = patch.object(tts, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)

    def save_budget(self):
        (self.state / "budget.json").write_text(json.dumps(self.budget), encoding="utf-8")

    def read_budget(self):
        return json.loads((self.state / "budget.json").read_text())

    def fake_cli(self, config, args, payload=None, timeout=45):
        if args[:3] == ["user", "subscription", "get"]:
            self.assertIn("--no-retry", args)
            return json.dumps(self.account)
        self.assertEqual(args[:2], ["text-to-speech", "convert"])
        self.assertIn("--no-retry", args)
        self.assertEqual(args[args.index("--json") + 1], "-")
        self.assertEqual(payload["language_code"], "it")
        self.assertEqual(payload["voice_settings"], {"stability": .5, "similarity_boost": .75})
        self.assertEqual(self.read_budget()["reservedCharacters"], 526 + len(payload["text"]))
        Path(args[args.index("--output") + 1]).write_bytes(MP3)
        return "Created audio"

    def test_status_is_local_only_and_config_is_optional(self):
        with patch.object(tts, "_run") as run:
            self.assertTrue(tts.status()["available"])
            self.assertNotIn("cli", tts.status())
            self.config_path.unlink()
            self.assertFalse(tts.status()["enabled"])
            run.assert_not_called()

    def test_enabled_but_broken_configuration_does_not_silently_select_local_voice(self):
        self.cli.unlink()
        result = tts.status()
        self.assertTrue(result["enabled"])
        self.assertFalse(result["available"])
        self.assertIsNotNone(result["reason"])

    def test_initialization_never_lowers_existing_reserved_usage(self):
        self.budget["reservedCharacters"] = 9586
        self.save_budget()
        with patch.object(tts, "_run", side_effect=self.fake_cli):
            self.assertEqual(tts.initialize_budget(526)["remainingCharacters"], 414)
        self.assertEqual(self.read_budget()["reservedCharacters"], 9586)

    def test_conversion_reserves_before_cli_then_reuses_cache_without_network(self):
        text = "Le stesse parole originali."
        with patch.object(tts, "_run", side_effect=self.fake_cli) as run:
            first = tts.synthesize(text, self.root / "one.mp3")
            self.assertFalse(first["cached"])
            self.assertEqual(run.call_count, 2)
        with patch.object(tts, "_run") as run:
            second = tts.synthesize(text, self.root / "two.mp3")
            self.assertTrue(second["cached"])
            run.assert_not_called()
        self.assertEqual((self.root / "two.mp3").read_bytes(), MP3)
        self.assertEqual(self.read_budget()["reservedCharacters"], 526 + len(text))

    def test_preflight_counts_unique_uncached_texts_and_does_not_reserve(self):
        cached = self.state / "audio" / (tts._key("già pronto", self.config) + ".mp3")
        cached.parent.mkdir()
        cached.write_bytes(MP3)
        with patch.object(tts, "_run", side_effect=self.fake_cli):
            result = tts.preflight(["già pronto", "nuovo", "nuovo"])
        self.assertEqual(result["uncachedCharacters"], 5)
        self.assertEqual(result["cachedRequests"], 1)
        self.assertEqual(result["remainingCharacters"], 9474)
        self.assertEqual(self.read_budget(), self.budget)

    def test_whole_batch_rejected_before_any_conversion_if_it_exceeds_free(self):
        self.budget["reservedCharacters"] = 9586
        self.save_budget()
        with patch.object(tts, "_run", side_effect=self.fake_cli) as run:
            with self.assertRaises(tts.ElevenLabsError) as error:
                tts.preflight(["a" * 300, "b" * 300])
            self.assertEqual(error.exception.code, "ELEVENLABS_QUOTA")
            self.assertEqual(run.call_count, 1)
        self.assertEqual(self.read_budget()["reservedCharacters"], 9586)

    def test_api_usage_can_raise_conservative_budget_even_if_other_clients_spent(self):
        self.account["used"] = 9999
        with patch.object(tts, "_run", side_effect=self.fake_cli) as run:
            with self.assertRaises(tts.ElevenLabsError) as error:
                tts.synthesize("troppo lungo", self.root / "no.mp3")
            self.assertEqual(error.exception.code, "ELEVENLABS_QUOTA")
            self.assertEqual(run.call_count, 1)
        self.assertFalse((self.root / "no.mp3").exists())

    def test_failed_request_keeps_reservation_and_cleans_temporary_output(self):
        def fail(config, args, payload=None, timeout=45):
            if args[:3] == ["user", "subscription", "get"]:
                return json.dumps(self.account)
            Path(args[args.index("--output") + 1]).write_bytes(b"partial")
            raise tts.ElevenLabsError("ELEVENLABS_TIMEOUT", "timeout")
        with patch.object(tts, "_run", side_effect=fail) as run:
            with self.assertRaises(tts.ElevenLabsError):
                tts.synthesize("prova", self.root / "no.mp3")
            self.assertEqual(run.call_count, 2)
        self.assertEqual(self.read_budget()["reservedCharacters"], 531)
        self.assertEqual(list(self.state.glob("request-*")), [])
        self.assertFalse((self.root / "no.mp3").exists())

    def test_invalid_audio_keeps_reserved_chars_and_is_not_cached(self):
        def invalid(config, args, payload=None, timeout=45):
            if payload is None:
                return json.dumps(self.account)
            Path(args[args.index("--output") + 1]).write_bytes(b"<html>an error</html>" * 100)
            return "success"
        with patch.object(tts, "_run", side_effect=invalid):
            with self.assertRaises(tts.ElevenLabsError) as error:
                tts.synthesize("cinque", self.root / "no.mp3")
            self.assertEqual(error.exception.code, "ELEVENLABS_AUDIO")
        self.assertEqual(self.read_budget()["reservedCharacters"], 532)
        self.assertEqual(list((self.state / "audio").glob("*.mp3")), [])

    def test_paid_extension_malformed_usage_and_premature_reset_refuse_generation(self):
        cases = ({"tier": "creator"}, {"can": True}, {"allowed": True}, {"used": "0"}, {"used": True},
                 {"limit": 10001}, {"limit": 0}, {"reset": int(time.time()) - 1})
        for change in cases:
            with self.subTest(change=change), patch.object(tts, "_run", return_value=json.dumps({**self.account, **change})) as run:
                with self.assertRaises(tts.ElevenLabsError):
                    tts.synthesize("test", self.root / "no.mp3")
                self.assertEqual(run.call_count, 1)
        self.account["reset"] += 86400
        with patch.object(tts, "_run", side_effect=self.fake_cli):
            with self.assertRaises(tts.ElevenLabsError) as error:
                tts.synthesize("test", self.root / "no.mp3")
            self.assertEqual(error.exception.code, "ELEVENLABS_BUDGET")
        self.assertEqual(self.read_budget(), self.budget)

    def test_reset_occurs_only_after_previous_period_has_ended(self):
        self.budget.update(period=int(time.time()) - 10, reservedCharacters=10000)
        self.save_budget()
        self.account["used"] = 12
        with patch.object(tts, "_run", side_effect=self.fake_cli):
            result = tts.initialize_budget()
        self.assertEqual(result["remainingCharacters"], 9988)
        self.assertEqual(self.read_budget()["period"], self.account["reset"])

    def test_missing_ledger_and_wrong_output_do_not_generate(self):
        (self.state / "budget.json").unlink()
        with patch.object(tts, "_run", side_effect=self.fake_cli) as run:
            with self.assertRaises(tts.ElevenLabsError) as error:
                tts.synthesize("test", self.root / "no.mp3")
            self.assertEqual(error.exception.code, "ELEVENLABS_BUDGET")
            self.assertEqual(run.call_count, 1)
            run.reset_mock()
            with self.assertRaises(tts.ElevenLabsError):
                tts.synthesize("test", self.root / "no.wav")
            run.assert_not_called()

    def test_cli_uses_stdin_hidden_window_and_never_exposes_stderr(self):
        with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "ok", "secret")) as run:
            self.assertEqual(tts._run(self.config, ["text-to-speech", "convert"], {"text": "a\"$()"}), "ok")
            args, kwargs = run.call_args
            self.assertNotIn("a\"$()", args[0])
            self.assertEqual(json.loads(kwargs["input"])["text"], "a\"$()")
            self.assertIn("creationflags", kwargs)
            self.assertNotIn("shell", kwargs)
        with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "token", "private email")):
            with self.assertRaises(tts.ElevenLabsError) as error:
                tts._run(self.config, ["convert"])
            self.assertNotIn("token", str(error.exception))
            self.assertNotIn("email", str(error.exception))

    def test_cli_error_diagnostics_keep_only_status_and_allowlisted_code(self):
        cases = [
            ('Error: HTTP 429 Too Many Requests\n{"detail":{"status":"rate_limit_exceeded","message":"private@example.com token-secret"}}', "ELEVENLABS_RATE_LIMIT", 429, "rate_limit_exceeded"),
            ('HTTP error: 403\n{"detail":{"status":"insufficient_permissions","message":"private@example.com"}}', "ELEVENLABS_FORBIDDEN", 403, "insufficient_permissions"),
            ('{"status_code":429,"detail":{"status":"quota_exceeded","message":"private account"}}', "ELEVENLABS_QUOTA", 429, "quota_exceeded"),
            ('Not authenticated. Run auth login. private@example.com', "ELEVENLABS_AUTH", None, None),
            ('HTTP 500\n{"detail":{"status":"private@example.com","message":"token-secret"}}', "ELEVENLABS_UNAVAILABLE", 500, None),
        ]
        for raw, code, http_status, api_code in cases:
            with self.subTest(code=code):
                error = tts._cli_failure("", raw)
                self.assertEqual((error.code, error.http_status, error.api_code), (code, http_status, api_code))
                self.assertNotIn("private", str(error))
                self.assertNotIn("token", str(error))
                self.assertNotIn("raw", vars(error))

    def test_formatted_status_error_and_lengths_remain_private_safe(self):
        cases = ["Request failed (status: 500), private@example.com", "status_code: '429' account-secret", "status: \"403\" token-secret"]
        for raw, expected in zip(cases, (500, 429, 403)):
            with self.subTest(expected=expected):
                error = tts._cli_failure("abc", raw, 2)
                self.assertEqual(error.http_status, expected)
                self.assertEqual(error.return_code, 2)
                self.assertEqual(error.stdout_length, 3)
                self.assertEqual(error.stderr_length, len(raw))
                self.assertNotIn("secret", str(error))
                self.assertNotIn("private", str(error))

    def test_success_exit_with_oversized_cli_output_exposes_only_lengths(self):
        result = subprocess.CompletedProcess([], 0, "x" * 65537, "")
        with patch.object(subprocess, "run", return_value=result):
            with self.assertRaises(tts.ElevenLabsError) as error:
                tts._run(self.config, ["convert"])
        self.assertEqual(error.exception.return_code, 0)
        self.assertEqual(error.exception.stdout_length, 65537)
        self.assertEqual(error.exception.stderr_length, 0)
        self.assertIsNone(error.exception.http_status)
        self.assertNotIn("stdout", vars(error.exception))

    def test_read_only_transient_failure_retries_once_without_reserving(self):
        transient = tts.ElevenLabsError("ELEVENLABS_UNAVAILABLE", "Temporary", http_status=500)
        with patch.object(tts, "_run", side_effect=[transient, json.dumps(self.account)]) as run, patch.object(tts.time, "sleep") as sleep:
            result = tts.preflight(["test"])
        self.assertEqual(result["uncachedCharacters"], 4)
        self.assertEqual(run.call_count, 2)
        self.assertTrue(all(call.args[1][:3] == ["user", "subscription", "get"] for call in run.call_args_list))
        sleep.assert_called_once_with(1)
        self.assertEqual(self.read_budget(), self.budget)

    def test_read_only_retry_is_bounded_and_auth_errors_are_never_retried(self):
        for error, expected in ((tts.ElevenLabsError("ELEVENLABS_RATE_LIMIT", "Temporary", http_status=429), 2),
                                (tts.ElevenLabsError("ELEVENLABS_FORBIDDEN", "Denied", http_status=403), 1),
                                (tts.ElevenLabsError("ELEVENLABS_AUTH", "Login", http_status=401), 1)):
            with self.subTest(code=error.code), patch.object(tts, "_run", side_effect=error) as run, patch.object(tts.time, "sleep"):
                with self.assertRaises(tts.ElevenLabsError) as raised:
                    tts.preflight(["test"])
                self.assertEqual(raised.exception.operation, "subscription")
                self.assertEqual(run.call_count, expected)
                self.assertEqual(self.read_budget(), self.budget)

    def test_real_run_tags_subscription_and_speech_without_retrying_speech(self):
        result = subprocess.CompletedProcess([], 2, "", "status: 500")
        for arguments, operation in ((["user", "subscription", "get"], "subscription"), (["text-to-speech", "convert"], "speech")):
            with self.subTest(operation=operation), patch.object(subprocess, "run", return_value=result) as run:
                with self.assertRaises(tts.ElevenLabsError) as error:
                    tts._run(self.config, arguments)
                self.assertEqual(error.exception.operation, operation)
                self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
