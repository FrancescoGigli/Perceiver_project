"""Streaming dei soli video esportati: autenticazione, seek e percorsi reali."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import app as tutor


class VideoLibraryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="perceiver-video-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.videos = self.root / "videos"
        self.videos.mkdir()
        self.payload = b"\x00\x00\x00\x18ftypisom" + bytes(range(256)) * 5
        (self.videos / "cap-01.mp4").write_bytes(self.payload)
        self.voice_payload = b"ID3" + bytes(range(256)) * 2
        (self.videos / "cap-01-voce-elevenlabs.mp3").write_bytes(self.voice_payload)
        (self.videos / "cap-01.vtt").write_text("WEBVTT\n\n00:00.000 --> 00:01.000\nIl mio codice.\n", encoding="utf-8")
        (self.videos / "cap-01.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nIl mio codice.\n", encoding="utf-8")
        self.plan = {"chapter": 1, "title": "La complessità quadratica", "duration": 12.25, "mediaGenerated": True}
        self.write_plan("cap-01", self.plan)
        for name, value in (("VIDEO_DIR", self.videos), ("REMOTE", False), ("LAN", False), ("TOKEN", "test-token-only")):
            patched = patch.object(tutor, name, value)
            patched.start()
            self.addCleanup(patched.stop)
        self.client = TestClient(tutor.app, base_url="https://testserver", client=("127.0.0.1", 50123))
        self.addCleanup(self.client.close)

    def write_plan(self, identifier, plan):
        (self.videos / (identifier + "-piano.json")).write_text(json.dumps(plan), encoding="utf-8")

    def test_local_listing_uses_export_metadata_and_actual_file_size(self):
        (self.root / "private.txt").write_text("Private downloads stay private", encoding="utf-8")
        (self.videos / "notes.txt").write_text("Not an export", encoding="utf-8")
        response = self.client.get("/api/videos")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["videos"], [{
            "id": "cap-01", "chapter": 1, "title": self.plan["title"], "duration": 12.25,
            "bytes": len(self.payload), "videoUrl": "/api/videos/cap-01.mp4",
            "planUrl": "/api/videos/cap-01-piano.json", "subtitlesUrl": "/api/videos/cap-01.vtt",
        }])
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_missing_directory_returns_empty_library(self):
        with patch.object(tutor, "VIDEO_DIR", self.root / "absent"):
            self.assertEqual(self.client.get("/api/videos").json(), {"videos": []})

    def test_only_generated_matched_exports_are_listed(self):
        for number, plan in ((2, {**self.plan, "chapter": 2, "mediaGenerated": False}),
                             (3, {**self.plan, "chapter": 4}),
                             (4, {**self.plan, "chapter": 4, "duration": float("nan")}),
                             (5, {**self.plan, "chapter": 5, "duration": True})):
            identifier = f"cap-{number:02d}"
            (self.videos / (identifier + ".mp4")).write_bytes(b"export")
            self.write_plan(identifier, plan)
        (self.videos / "cap-06.mp4").write_bytes(b"no metadata yet")
        (self.videos / "cap-07.mp4").write_bytes(b"bad metadata")
        (self.videos / "cap-07-piano.json").write_text("broken json", encoding="utf-8")
        self.write_plan("cap-08", {**self.plan, "chapter": 8})
        self.assertEqual([video["chapter"] for video in self.client.get("/api/videos").json()["videos"]], [1])

    def test_missing_subtitles_are_explicit(self):
        (self.videos / "cap-01.vtt").unlink()
        self.assertIsNone(self.client.get("/api/videos").json()["videos"][0]["subtitlesUrl"])

    def test_full_video_and_head_have_correct_media_headers(self):
        response = self.client.get("/api/videos/cap-01.mp4")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, self.payload)
        self.assertEqual(response.headers["content-type"], "video/mp4")
        self.assertEqual(response.headers["content-length"], str(len(self.payload)))
        self.assertEqual(response.headers["accept-ranges"], "bytes")
        self.assertEqual(response.headers["cache-control"], "private, no-store")
        head = self.client.head("/api/videos/cap-01.mp4")
        self.assertEqual(head.status_code, 200)
        self.assertEqual(head.content, b"")
        self.assertEqual(head.headers["content-length"], str(len(self.payload)))

    def test_ranges_return_exact_bytes_for_mobile_seeking(self):
        for value, first, last in (("bytes=0-9", 0, 9), ("bytes=1200-", 1200, len(self.payload) - 1),
                                   ("bytes=-12", len(self.payload) - 12, len(self.payload) - 1),
                                   ("bytes=1200-999999", 1200, len(self.payload) - 1),
                                   ("bytes=-999999", 0, len(self.payload) - 1)):
            with self.subTest(value=value):
                response = self.client.get("/api/videos/cap-01.mp4", headers={"Range": value})
                self.assertEqual(response.status_code, 206)
                self.assertEqual(response.content, self.payload[first:last + 1])
                self.assertEqual(response.headers["content-range"], f"bytes {first}-{last}/{len(self.payload)}")
                self.assertEqual(response.headers["content-length"], str(last - first + 1))
        head = self.client.head("/api/videos/cap-01.mp4", headers={"Range": "bytes=10-19"})
        self.assertEqual(head.status_code, 206)
        self.assertEqual(head.content, b"")
        self.assertEqual(head.headers["content-length"], "10")

    def test_invalid_ranges_return_416_without_a_body(self):
        for value in ("bytes=999999-", "bytes=20-10", "bytes=-0", "bytes=-", "bytes=0-2,8-9",
                      "items=0-10", "bytes=" + "9" * 5000 + "-"):
            with self.subTest(value=value):
                response = self.client.get("/api/videos/cap-01.mp4", headers={"Range": value})
                self.assertEqual(response.status_code, 416)
                self.assertEqual(response.headers["content-range"], f"bytes */{len(self.payload)}")
                self.assertEqual(response.content, b"")

    def test_voice_preview_supports_mp3_head_and_exact_audio_ranges(self):
        path = "/api/videos/cap-01-voce-elevenlabs.mp3"
        response = self.client.get(path)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, self.voice_payload)
        self.assertEqual(response.headers["content-type"], "audio/mpeg")
        self.assertEqual(response.headers["cache-control"], "private, no-store")
        self.assertEqual(response.headers["accept-ranges"], "bytes")
        head = self.client.head(path)
        self.assertEqual(head.status_code, 200)
        self.assertEqual(head.content, b"")
        self.assertEqual(head.headers["content-type"], "audio/mpeg")
        self.assertEqual(head.headers["content-length"], str(len(self.voice_payload)))
        part = self.client.get(path, headers={"Range": "bytes=10-19"})
        self.assertEqual(part.status_code, 206)
        self.assertEqual(part.content, self.voice_payload[10:20])
        self.assertEqual(part.headers["content-range"], f"bytes 10-19/{len(self.voice_payload)}")
        self.assertEqual(self.client.head(path, headers={"Range": "bytes=10-19"}).headers["content-length"], "10")
        invalid = self.client.get(path, headers={"Range": "bytes=999999-"})
        self.assertEqual(invalid.status_code, 416)
        self.assertEqual(invalid.content, b"")
        self.assertEqual(invalid.headers["content-range"], f"bytes */{len(self.voice_payload)}")

    def test_only_the_exact_elevenlabs_preview_filename_is_exposed(self):
        for filename in ("cap-02-voce-elevenlabs.mp3", "cap-01-voce-sara.mp3", "cap-01.mp3",
                         "cap-01-voce-elevenlabs.wav", "cap-01-voce-elevenlabs.mp3.bak"):
            (self.videos / filename).write_bytes(b"Never exposed")
            with self.subTest(filename=filename):
                self.assertEqual(self.client.get("/api/videos/" + filename).status_code, 404)
        for filename in ("sub/cap-01-voce-elevenlabs.mp3", "..%5ccap-01-voce-elevenlabs.mp3"):
            with self.subTest(filename=filename):
                self.assertEqual(self.client.get("/api/videos/" + filename).status_code, 404)
        (self.videos / "cap-01-voce-elevenlabs.mp3").unlink()
        self.assertEqual(self.client.head("/api/videos/cap-01-voce-elevenlabs.mp3").status_code, 404)

    def test_voice_preview_requires_the_existing_remote_session_and_same_origin(self):
        path = "/api/videos/cap-01-voce-elevenlabs.mp3"
        with patch.object(tutor, "REMOTE", True):
            self.assertEqual(self.client.get(path).status_code, 401)
            self.assertEqual(self.client.head(path).status_code, 401)
            header = {"X-Tutor-Token": "test-token-only"}
            self.assertEqual(self.client.get(path, headers=header).content, self.voice_payload)
            self.assertEqual(self.client.post("/api/media-session", headers=header).status_code, 200)
            self.assertEqual(self.client.head(path).status_code, 200)
            self.assertEqual(self.client.get(path, headers={"Range": "bytes=-12"}).content, self.voice_payload[-12:])
            self.assertEqual(self.client.get(path, headers={"Origin": "https://other.example"}).status_code, 403)
            self.client.delete("/api/media-session")
            self.assertEqual(self.client.get(path).status_code, 401)

    def test_subtitles_and_plan_preserve_the_exported_contents(self):
        vtt = self.client.get("/api/videos/cap-01.vtt")
        self.assertEqual(vtt.status_code, 200)
        self.assertTrue(vtt.headers["content-type"].startswith("text/vtt"))
        self.assertEqual(vtt.content, (self.videos / "cap-01.vtt").read_bytes())
        self.assertEqual(self.client.get("/api/videos/cap-01-piano.json").json(), self.plan)
        self.assertEqual(self.client.get("/api/videos/cap-01.srt").status_code, 200)

    def test_paths_outside_the_dedicated_export_names_are_not_served(self):
        (self.root / "private.txt").write_text("Never exposed", encoding="utf-8")
        (self.videos / "cap-01.exe").write_bytes(b"not video")
        for path in ("private.txt", "notes.txt", "cap-01.exe", "cap-00.mp4", "cap-53.mp4", "cap-1.mp4",
                     "sub/cap-01.mp4", "%2e%2e%2fprivate.txt", "..%5cprivate.txt", "cap-01.mp4%00", "cap-02.mp4"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get("/api/videos/" + path).status_code, 404)

    def test_export_symlinks_cannot_expose_other_downloads(self):
        external = self.root / "private.mp4"
        external.write_bytes(b"Never exposed")
        try:
            (self.videos / "cap-02.mp4").symlink_to(external)
        except OSError:
            self.skipTest("Il sistema non consente link simbolici per questo utente.")
        self.write_plan("cap-02", {**self.plan, "chapter": 2})
        self.assertEqual(self.client.get("/api/videos/cap-02.mp4").status_code, 404)
        self.assertEqual(len(self.client.get("/api/videos").json()["videos"]), 1)

    def test_remote_media_requires_header_or_a_dedicated_cookie(self):
        with patch.object(tutor, "REMOTE", True):
            self.assertEqual(self.client.get("/api/videos").status_code, 401)
            self.assertEqual(self.client.get("/api/videos/cap-01.mp4").status_code, 401)
            self.assertEqual(self.client.post("/api/media-session").status_code, 401)
            header = {"X-Tutor-Token": "test-token-only"}
            self.assertEqual(self.client.get("/api/videos/cap-01.mp4", headers=header).content, self.payload)
            self.assertEqual(self.client.get("/api/videos", headers=header).status_code, 200)
            response = self.client.post("/api/media-session", headers=header)
            self.assertEqual(response.status_code, 200)
            cookie = response.headers["set-cookie"]
            self.assertIn("HttpOnly", cookie)
            self.assertIn("Secure", cookie)
            self.assertIn("SameSite=strict", cookie)
            self.assertIn("Path=/api/videos", cookie)
            self.assertNotIn("test-token-only", cookie)
            self.assertEqual(self.client.get("/api/videos/cap-01.mp4").content, self.payload)
            self.assertEqual(self.client.get("/api/videos/cap-01.vtt").status_code, 200)
            self.assertEqual(self.client.get("/api/videos").status_code, 401)
            self.assertEqual(self.client.get("/api/health").status_code, 401)
            self.assertEqual(self.client.post("/api/media-session").status_code, 401)

    def test_cookie_is_expiring_signed_and_invalidated_when_token_changes(self):
        with patch.object(tutor, "REMOTE", True):
            self.client.post("/api/media-session", headers={"X-Tutor-Token": "test-token-only"})
            with patch.object(tutor, "TOKEN", "rotated-test-token"):
                self.assertEqual(self.client.get("/api/videos/cap-01.mp4").status_code, 401)
            with patch.object(tutor.time, "time", return_value=time.time() + tutor.MEDIA_SESSION_SECONDS + 1):
                self.assertEqual(self.client.get("/api/videos/cap-01.mp4").status_code, 401)
            valid = self.client.cookies.get(tutor.MEDIA_COOKIE)
            self.client.cookies.clear()
            for cookie in (valid[:-1] + ("a" if valid[-1] != "a" else "b"), "999999999999." + "a" * 24 + "." + "f" * 64,
                           "9" * 5000 + ".nonce.signature", "9999999999.nonce.ä"):
                response = self.client.get("/api/videos/cap-01.mp4", headers={"Cookie": (tutor.MEDIA_COOKIE + "=" + cookie).encode("latin-1")})
                self.assertEqual(response.status_code, 401)

    def test_disconnect_clears_only_browser_media_cookie_without_revoking_header_access(self):
        with patch.object(tutor, "REMOTE", True):
            header = {"X-Tutor-Token": "test-token-only"}
            self.client.post("/api/media-session", headers=header)
            self.assertEqual(self.client.get("/api/videos/cap-01.mp4").status_code, 200)
            response = self.client.delete("/api/media-session")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"ok": True})
            cookie = response.headers["set-cookie"]
            for attribute in ("Max-Age=0", "Path=/api/videos", "HttpOnly", "Secure", "SameSite=strict"):
                self.assertIn(attribute, cookie)
            self.assertIsNone(self.client.cookies.get(tutor.MEDIA_COOKIE))
            self.assertEqual(self.client.get("/api/videos/cap-01.mp4").status_code, 401)
            self.assertEqual(self.client.get("/api/health").status_code, 401)
            self.assertEqual(self.client.get("/api/health", headers=header).status_code, 200)
            self.assertEqual(self.client.get("/api/videos/cap-01.mp4", headers=header).status_code, 200)
            self.assertEqual(self.client.delete("/api/media-session", headers={"X-Tutor-Token": "old-expired-token"}).status_code, 200)

    def test_cross_origin_session_and_media_requests_are_rejected(self):
        for method, path in (("post", "/api/media-session"), ("delete", "/api/media-session"),
                             ("get", "/api/videos/cap-01.mp4"), ("get", "/api/videos")):
            with self.subTest(path=path):
                response = getattr(self.client, method)(path, headers={"Origin": "https://other.example"})
                self.assertEqual(response.status_code, 403)

    def test_lan_also_requires_authorization_and_allows_plain_http_cookie(self):
        with patch.object(tutor, "LAN", True):
            with TestClient(tutor.app, base_url="http://testserver", client=("192.168.1.10", 50123)) as client:
                self.assertEqual(client.get("/api/videos/cap-01.mp4").status_code, 401)
                response = client.post("/api/media-session", headers={"X-Tutor-Token": "test-token-only"})
                self.assertEqual(response.status_code, 200)
                self.assertNotIn("Secure", response.headers["set-cookie"])
                self.assertEqual(client.get("/api/videos/cap-01.mp4").content, self.payload)

    def test_loopback_http_server_streams_cookie_authenticated_ranges(self):
        """Verifica socket HTTP reali senza avviare player o toccare il server dell'utente."""
        import http.client
        import socket
        import threading
        import uvicorn

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(tutor.app, host="127.0.0.1", port=port, log_level="error",
                                              access_log=False, proxy_headers=False, lifespan="off", ws="none"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)

        def request(method, path, headers=None):
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            try:
                connection.request(method, path, headers=headers or {})
                response = connection.getresponse()
                return response.status, {key.lower(): value for key, value in response.getheaders()}, response.read()
            finally:
                connection.close()

        with patch.object(tutor, "REMOTE", True):
            try:
                thread.start()
                deadline = time.monotonic() + 5
                while not server.started and thread.is_alive() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(server.started, "Il server HTTP di prova non è partito.")
                self.assertEqual(request("GET", "/api/videos/cap-01.mp4")[0], 401)
                status, headers, body = request("POST", "/api/media-session", {"X-Tutor-Token": "test-token-only"})
                self.assertEqual(status, 200)
                self.assertTrue(json.loads(body)["ok"])
                cookie = headers["set-cookie"].split(";", 1)[0]
                status, headers, body = request("GET", "/api/videos/cap-01.mp4", {"Cookie": cookie, "Range": "bytes=10-19"})
                self.assertEqual(status, 206)
                self.assertEqual(body, self.payload[10:20])
                self.assertEqual(headers["content-range"], f"bytes 10-19/{len(self.payload)}")
                self.assertEqual(headers["content-length"], "10")
                status, headers, body = request("HEAD", "/api/videos/cap-01.mp4", {"Cookie": cookie})
                self.assertEqual(status, 200)
                self.assertEqual(headers["content-length"], str(len(self.payload)))
                self.assertEqual(body, b"")
                self.assertEqual(request("GET", "/api/videos/cap-01.mp4", {"Cookie": cookie, "Range": "bytes=999999-"})[0], 416)
            finally:
                server.should_exit = True
                thread.join(timeout=5)
                listener.close()
                self.assertFalse(thread.is_alive(), "Il server HTTP di prova deve terminare.")


if __name__ == "__main__":
    unittest.main()
