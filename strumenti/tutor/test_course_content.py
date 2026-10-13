"""Check source fidelity and complete coverage of the generated guided course."""
import hashlib
from html import unescape
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from course_content import (build_course, chapter_grounding, get_chapter,
                            LESSON_PATH, load_course, MANIFEST_PATH, parse_lesson,
                            plain_text)

ROOT = Path(__file__).resolve().parents[2]


def strip_inline_markup(raw):
    return re.sub(r"\s+", " ", unescape(re.sub(r"</?[A-Za-z][^>]*>", "", raw))).strip()


class CourseContentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.course = build_course(ROOT)
        cls.lesson = (ROOT / LESSON_PATH).read_text(encoding="utf-8")
        cls.source_chapters = parse_lesson(cls.lesson)

    def test_exactly_all_original_52_chapters_and_titles(self):
        self.assertEqual([chapter["id"] for chapter in self.course["chapters"]], list(range(1, 53)))
        # Independent raw HTML scan catches a parser dropping or relabelling a
        # chapter while the manifest and parser happen to agree with each other.
        titles = {int(number): strip_inline_markup(title) for number, title in re.findall(r'<section\b[^>]*data-chapter="(\d+)"[^>]*>.*?<h1>(.*?)</h1>', self.lesson, flags=re.S)}
        for chapter in self.course["chapters"]:
            self.assertEqual(chapter["title"], titles[chapter["id"]])
            self.assertEqual(chapter["originalHref"], f"../lezione/#cap-{chapter['id']}")
            self.assertGreater(len(chapter["steps"]), 0)

    def test_every_quote_is_one_contiguous_original_block_at_its_source_line(self):
        for source_chapter, chapter in zip(self.source_chapters, self.course["chapters"]):
            blocks = {(node.line, plain_text(node)) for node in source_chapter.descendants()}
            for step in chapter["steps"]:
                self.assertIn((step["source"]["line"], step["text"]), blocks, step["id"])
                self.assertEqual(step["source"]["path"], LESSON_PATH)
                self.assertEqual(step["source"]["anchor"], f"cap-{chapter['id']}")
            self.assertEqual([step["source"]["line"] for step in chapter["steps"]], sorted(step["source"]["line"] for step in chapter["steps"]))

    def test_numerical_claims_and_their_qualifiers_are_preserved(self):
        original_claim = "Il decoder di Perceiver IO aiuta su CIFAR-10?"
        quiz = next(item for item in get_chapter(self.course, 46)["quiz"] if item["question"] == original_claim)
        self.assertEqual(quiz["expected"], "No: 71,79% e 71,69% contro 71,63%, dentro la banda. Il decoder a query serve quando l'output è strutturato, non per una sola etichetta.")
        text = " ".join(step["text"] for step in get_chapter(self.course, 46)["steps"])
        self.assertIn("È stato rimosso, non aggiornato", text)
        self.assertIn("due degli otto valori non sono apprendimento", text)

    def test_all_original_open_questions_and_reference_answers_without_fake_choices(self):
        originals = re.findall(r'<details class="qa-item"><summary>(.*?)</summary><p class="qa-a">(.*?)</p></details>', self.lesson, flags=re.S)
        quiz = [item for chapter in self.course["chapters"] for item in chapter["quiz"]]
        self.assertEqual(len(quiz), len(originals))
        self.assertEqual(len(quiz), 285)
        original_pairs = {(strip_inline_markup(question), strip_inline_markup(answer)) for question, answer in originals}
        for question in quiz:
            self.assertTrue((question["question"], question["expected"]) in original_pairs, f"Fonte Q&A alterata: {question['id']}")
            self.assertEqual(question["type"], "open")
            self.assertEqual(question["options"], [])
            self.assertEqual(question["correct"], [])
            self.assertEqual(question["expected"], question["explanation"])
            self.assertEqual(question["evaluation"], "model-reference")
        self.assertEqual([chapter["id"] for chapter in self.course["chapters"] if not chapter["quiz"]], [52])

    def test_real_code_ranges_and_hashes(self):
        for chapter in self.course["chapters"]:
            codes = chapter["code"] + [code for step in chapter["steps"] for code in step["code"]]
            for code in codes:
                path = ROOT / code["path"]
                self.assertTrue(path.is_file())
                self.assertTrue(path.resolve().is_relative_to((ROOT / "progetto").resolve()))
                lines = path.read_text(encoding="utf-8").splitlines()
                self.assertEqual(code["text"], "\n".join(lines[code["start"] - 1:code["end"]]))
                self.assertGreaterEqual(code["start"], 1)
                self.assertLessEqual(code["end"], len(lines))
        for path, digest in self.course["source"]["code_sha256"].items():
            self.assertEqual(digest, hashlib.sha256((ROOT / path).read_text(encoding="utf-8").encode("utf-8")).hexdigest())
        self.assertIn("q_proj", get_chapter(self.course, 7)["code"][0]["text"])
        self.assertIn("self.decoder", get_chapter(self.course, 15)["code"][0]["text"])
        self.assertIn("Lamb", get_chapter(self.course, 25)["code"][0]["text"])
        self.assertIn("randaugment_num_ops", get_chapter(self.course, 39)["code"][0]["text"])

    def test_cifar_baseline_and_permutation_override_have_priority_in_small_context(self):
        for chapter_id in (44, 45):
            chapter = get_chapter(self.course, chapter_id)
            self.assertEqual(chapter["code"][0]["symbol"], "BASE")
            context, sources = chapter_grounding(self.course, chapter_id, max_chars=4500)
            self.assertIn('"--num_latents", "96"', context)
            self.assertIn('"--num_cross_attend_stages", "4"', context)
            self.assertIn('"--num_transformer_blocks", "4"', context)
            self.assertIn('_exp("e01_baseline", "tab1", [])', context)
            self.assertIn('_exp("e02_permuted", "tab2", ["--permute_pixels", "--permute_pixels_seed", "42"])', context)
            self.assertTrue(any(source.get("line_start") == 12 for source in sources))
        chapter = get_chapter(self.course, 45)
        focused = [step for step in chapter["steps"] if step.get("experimentId") in {"e01_baseline", "e02_permuted"}]
        self.assertEqual(len(focused), 8)
        for step in focused:
            context, _ = chapter_grounding(self.course, 45, step["id"], max_chars=4500)
            self.assertIn('"--num_latents", "96"', context)
            self.assertIn('"--num_cross_attend_stages", "4"', context)
            self.assertNotIn('"--num_cross_attend_stages", "1"', context)
            self.assertNotIn("e03_learned_pe", context)
            self.assertNotIn("e04_learned_pe_permuted", context)
            if step["experimentId"] == "e02_permuted":
                self.assertIn('["--permute_pixels", "--permute_pixels_seed", "42"]', context)

    def test_single_run_grounding_uses_its_actual_base_without_adjacent_run_overrides(self):
        chapter = get_chapter(self.course, 45)
        learned = next(step for step in chapter["steps"] if step.get("experimentId") == "e03_learned_pe")
        context, _ = chapter_grounding(self.course, 45, learned["id"], max_chars=4500)
        self.assertIn('"--num_cross_attend_stages", "1"', context)
        self.assertNotIn("e02_permuted", context)
        modelnet = next(step for step in chapter["steps"] if step.get("experimentId") == "mn01_baseline")
        self.assertEqual(modelnet["code"][0]["symbol"], "BASE_MODELNET")
        context, _ = chapter_grounding(self.course, 45, modelnet["id"], max_chars=4500)
        self.assertIn('"--num_latents", "128"', context)
        self.assertIn('"--num_cross_attend_stages", "2"', context)

    def test_theoretical_architectures_are_not_presented_as_existing_code(self):
        for number in (26, 28, 29, 30, 35, 47, 52):
            chapter = get_chapter(self.course, number)
            self.assertEqual(chapter["code"], [])
            self.assertTrue(chapter["codeNote"])
        self.assertIn("MLM su byte", get_chapter(self.course, 48)["codeNote"])

    def test_all_81_original_labs_are_mapped_within_their_own_chapter(self):
        lab_count = 0
        for source, chapter in zip(self.source_chapters, self.course["chapters"]):
            originals = {(node.attrs["data-lab"], node.attrs.get("data-preset")) for node in source.descendants() if "data-lab" in node.attrs}
            mapped = {(animation["labId"], animation.get("preset")) for animation in chapter["animations"] if "labId" in animation}
            self.assertEqual(mapped, originals)
            lab_count += len(mapped)
            ids = {node.attrs["id"] for node in source.descendants() if "id" in node.attrs}
            for step in chapter["steps"]:
                animation = step["animation"]
                self.assertEqual(animation["kind"], "original")
                if "labId" in animation:
                    self.assertIn((animation["labId"], animation.get("preset")), originals)
                if "targetId" in animation:
                    self.assertIn(animation["targetId"], ids)
        self.assertEqual(lab_count, 81)

    def test_bibliography_links_and_formula_explanations_are_original_content(self):
        bibliography = get_chapter(self.course, 52)
        self.assertTrue(any("Perceiver: General Perception" in step["text"] for step in bibliography["steps"]))
        links = [link for step in bibliography["steps"] for link in step.get("links", [])]
        self.assertTrue(any(link["href"] == "https://arxiv.org/abs/2103.03206" for link in links))
        formulas = get_chapter(self.course, 41)
        self.assertTrue(any("riduzione della complessità tramite latent bottleneck" in step["text"] for step in formulas["steps"]))
        forward = get_chapter(self.course, 42)
        self.assertTrue(any(animation.get("targetId") == "flowPanel" for animation in forward["animations"]))

    def test_course_manifest_current_and_has_no_private_or_generated_media(self):
        saved = json.loads((ROOT / MANIFEST_PATH).read_text(encoding="utf-8"))
        self.assertEqual(saved, self.course)
        self.assertFalse(saved["media"]["generated"])
        self.assertFalse(saved["media"]["autoplay"])
        self.assertNotIn("_copione/", json.dumps(saved))
        self.assertNotIn("backups/", json.dumps(saved))
        self.assertFalse(any(key in json.dumps(saved) for key in (".mp4", ".wav", ".mp3")))

    def test_grounding_keeps_active_step_and_real_code_and_rejects_foreign_step(self):
        chapter = get_chapter(self.course, 7)
        context, sources = chapter_grounding(self.course, 7, chapter["steps"][0]["id"])
        self.assertIn(chapter["steps"][0]["text"], context)
        self.assertIn("Codice reale", context)
        self.assertIn("q_proj", context)
        self.assertEqual(sources[0]["anchor"], "cap-7")
        with self.assertRaises(ValueError):
            chapter_grounding(self.course, 7, "c15-s1")
        for invalid in (0, 53, True, "7"):
            with self.assertRaises(ValueError):
                get_chapter(self.course, invalid)

    def test_changed_lesson_does_not_load_stale_manifest_or_write_anything(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / LESSON_PATH).parent.mkdir(parents=True)
            (root / LESSON_PATH).write_text(self.lesson + "\n<!-- changed -->", encoding="utf-8")
            (root / MANIFEST_PATH).parent.mkdir(parents=True)
            (root / MANIFEST_PATH).write_text(json.dumps(self.course), encoding="utf-8")
            original_file = (root / MANIFEST_PATH).read_bytes()
            marker = {"rebuilt": True}
            with patch("course_content.build_course", return_value=marker) as rebuild:
                self.assertEqual(load_course(root), marker)
                rebuild.assert_called_once_with(root.resolve())
            self.assertEqual((root / MANIFEST_PATH).read_bytes(), original_file)


if __name__ == "__main__":
    unittest.main()
