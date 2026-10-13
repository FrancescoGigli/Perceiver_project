"""Plan/source checks only. These tests never invoke TTS, encoding or playback."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import export_chapter as exporter


class ChapterPlanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="perceiver-export-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.lesson = self.root / "sito/lezione/index.html"
        self.lesson.parent.mkdir(parents=True)
        self.lesson.write_text(
            '<section data-chapter="1" id="cap-1">\n'
            '<p>Q = una domanda. V è il contenuto.</p>\n'
            '<table><tr><th>Simbolo</th><th>Significato</th></tr></table>\n'
            '<p>Da dove viene Q?</p><p>Dai latenti.</p>\n'
            '<script>Testo nascosto non è una citazione.</script></section>', encoding="utf-8")
        self.code_file = self.root / "progetto/src/perceiver/attention.py"
        self.code_file.parent.mkdir(parents=True)
        self.code_file.write_text('def forward(q, k, v):\n    dots = q @ k.T\n    return dots @ v\n', encoding="utf-8")
        self.manifest = {
            "version": 1,
            "source": {"path": "sito/lezione/index.html", "sha256": hashlib.sha256(self.lesson.read_text(encoding="utf-8").encode()).hexdigest()},
            "chapters": [{"id": 1, "title": "Q, K, V", "steps": [{
                "id": "c1-s1", "title": "Domanda e contenuto", "text": "Q = una domanda. V è il contenuto.",
                "source": {"path": "sito/lezione/index.html", "line": 2, "anchor": "cap-1"},
                "code": [{"path": "progetto/src/perceiver/attention.py", "start": 1, "end": 3,
                          "text": 'def forward(q, k, v):\n    dots = q @ k.T\n    return dots @ v'}],
                "animation": {"kind": "original", "targetId": "lab-attention", "href": "../lezione/#cap-1"},
            }], "quiz": [{"id": "c1-q1", "type": "open", "question": "Da dove viene Q?", "expected": "Dai latenti.",
                           "explanation": "Dai latenti.", "source": {"path": "sito/lezione/index.html", "line": 4, "anchor": "cap-1"}}]}],
        }

    def plan(self, manifest=None):
        return exporter.build_plan(manifest or self.manifest, 1, root=self.root)

    def test_plan_is_verified_but_has_no_media_or_measured_timings(self):
        plan = self.plan()
        self.assertTrue(plan["sourceVerified"])
        self.assertFalse(plan["mediaGenerated"])
        self.assertFalse(plan["timingsVerified"])
        self.assertEqual([scene["kind"] for scene in plan["scenes"]], ["explanation", "question", "answer"])
        self.assertEqual(plan["scenes"][0]["originalAnimation"]["targetId"], "lab-attention")
        self.assertEqual(plan["scenes"][0]["animationExport"], "code_highlight")

    def test_changed_lesson_stops_before_media(self):
        self.lesson.write_text(self.lesson.read_text(encoding="utf-8") + "\n<p>Modificato.</p>", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "lezione è cambiata"):
            self.plan()

    def test_fabricated_quote_is_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["chapters"][0]["steps"][0]["text"] = "Q viene dall'input."
        with self.assertRaisesRegex(ValueError, "Citazione non presente"):
            self.plan(manifest)

    def test_hidden_script_text_cannot_be_a_quote(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["chapters"][0]["steps"][0]["text"] = "Testo nascosto non è una citazione."
        with self.assertRaisesRegex(ValueError, "Citazione non presente"):
            self.plan(manifest)

    def test_changed_code_is_rejected(self):
        self.code_file.write_text('def forward(q, k, v):\n    dots = q @ k.T\n    return q\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Codice cambiato"):
            self.plan()

    def test_original_table_cells_remain_separated(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["chapters"][0]["steps"][0]["text"] = "Simbolo Significato"
        self.assertEqual(self.plan(manifest)["scenes"][0]["text"], "Simbolo Significato")

    def test_unknown_chapter_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non presente"):
            exporter.build_plan(self.manifest, 53, root=self.root)

    def test_invalid_code_interval_is_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["chapters"][0]["steps"][0]["code"][0]["end"] = 99
        with self.assertRaisesRegex(ValueError, "Intervallo di codice"):
            self.plan(manifest)

    def test_private_and_traversal_paths_are_rejected(self):
        for path in ("sito/slide/_copione/copione.py", "../attention.py", "C:/secret.py", "progetto/logs/private.py"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                exporter.source_path(self.root, path, code=True)

    def test_output_cannot_enter_repository(self):
        with self.assertRaisesRegex(ValueError, "fuori dal repository"):
            exporter.output_directory(self.root / "sito/media", root=self.root)

    def test_whitespace_splits_preserve_all_source_words(self):
        quote = "Un esempio di Q, K e V. " * 65
        chunks = exporter.split_quote(quote)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 260 for chunk in chunks))
        self.assertEqual(" ".join(chunks), exporter.normalise(quote))

    def test_long_code_is_paginated_without_losing_source_lines(self):
        lines = [f"    value_{n} = {n}" for n in range(33)]
        pages = exporter.code_pages([{"path": "progetto/train.py", "start": 10, "text": "\n".join(lines)}])
        self.assertEqual(len(pages), 5)
        rows = [row for page in pages for row in page["rows"]]
        self.assertEqual([row["text"] for row in rows], lines)
        self.assertEqual([row["line"] for row in rows], list(range(10, 43)))

    def test_svg_is_valid_and_escapes_source_characters(self):
        plan = self.plan()
        scene = copy.deepcopy(plan["scenes"][0])
        scene["text"] = "Q < K & V > 0"
        svg = exporter.scene_svg(plan, scene, 0)
        tree = ET.fromstring(svg)
        self.assertIn("Q < K & V > 0", "".join(tree.itertext()))
        self.assertIn('font-family="Consolas"', svg)
        self.assertIn('fill="#1a237e"', svg)

    def test_subtitle_timing_keeps_question_pause(self):
        plan = self.plan()
        for scene, bounds in zip(plan["scenes"], [(0., 2.), (2., 12.), (12., 15.)]):
            scene["start"], scene["end"] = bounds
        vtt, srt = exporter.subtitle_text(plan["scenes"])
        self.assertIn("00:00:02.000 --> 00:00:12.000", vtt)
        self.assertIn("00:00:02,000 --> 00:00:12,000", srt)
        self.assertIn("Da dove viene Q?", vtt)

    def test_future_encoding_command_is_mobile_compatible_without_running_it(self):
        command = exporter.ffmpeg_command("ffmpeg", Path("audio.wav"), Path("cap.srt"), Path("cap.mp4"), 20)
        self.assertIn("libx264", command)
        self.assertIn("aac", command)
        self.assertIn("yuv420p", command)
        self.assertIn("+faststart", command)
        self.assertIn("mov_text", command)
        self.assertIn("1280x720", command)

    def test_cli_default_and_plan_do_not_render_or_create_files(self):
        manifest_file = self.root / "manifest.json"
        manifest_file.write_text(json.dumps(self.manifest), encoding="utf-8")
        build = exporter.build_plan
        def fixture_build(manifest, chapter, **kwargs):
            return build(manifest, chapter, root=self.root, **kwargs)
        before = sorted(str(path) for path in self.root.rglob("*"))
        for flag in ([], ["--plan"]):
            with self.subTest(flag=flag), patch.object(exporter, "build_plan", side_effect=fixture_build), \
                    patch.object(exporter, "render_plan", side_effect=AssertionError("Media must not run")) as renderer, \
                    patch.object(exporter, "runtime_directory", side_effect=AssertionError("Voice must not load")), \
                    contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(exporter.main(["--chapter", "1", "--manifest", str(manifest_file), *flag]), 0)
                self.assertFalse(json.loads(stdout.getvalue())["mediaGenerated"])
                renderer.assert_not_called()
        self.assertEqual(before, sorted(str(path) for path in self.root.rglob("*")))

    def test_render_requires_explicit_external_output_before_loading_any_model(self):
        with patch.object(exporter, "render_plan", side_effect=AssertionError("Must not start")) as renderer, \
                contextlib.redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(exporter.main(["--chapter", "1", "--render"]), 2)
        self.assertIn("richiede --output", stderr.getvalue())
        renderer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
