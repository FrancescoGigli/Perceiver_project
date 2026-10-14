"""Guided teaching uses the real model contract and original quiz snapshots."""
import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import app as tutor
from course_content import get_chapter, load_course


class TeachingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.course = load_course(tutor.ROOT)

    def setUp(self):
        tutor.sessions.clear()
        self.client = TestClient(tutor.app, client=("127.0.0.1", 50123))
        for name in ("REMOTE", "LAN"):
            setting = patch.object(tutor, name, False)
            setting.start()
            self.addCleanup(setting.stop)

    def request(self, **fields):
        return self.client.post("/api/chat", json={
            "message": "Fammi capire il perché di questo passaggio.",
            "scene": "course", "mode": "coach", "chapter": 1, "course_step": "c1-s2", **fields,
        })

    @staticmethod
    def result(answer="Esempio didattico: raddoppiare gli elementi quadruplica i confronti fra tutte le coppie.", action=None):
        return {"message": {"content": json.dumps({"answer": answer, "action": action})}}

    def test_coach_calls_model_with_verified_step_code_and_causal_prompt_without_tts(self):
        with patch.object(tutor, "ollama_request", return_value=self.result()) as model, \
                patch.object(tutor, "synthesize_voice") as voice:
            response = self.request()
        self.assertEqual(response.status_code, 200)
        model.assert_called_once()
        voice.assert_not_called()
        path, payload = model.call_args.args[:2]
        self.assertEqual(path, "/api/chat")
        self.assertEqual(payload["model"], tutor.MODEL)
        context = payload["messages"][0]["content"]
        step = get_chapter(self.course, 1)["steps"][1]
        self.assertIn(step["text"], context)
        self.assertIn("latents = self._apply_cross(stage, latents, data", context)
        self.assertIn('"symbol": "PerceiverEncoder.forward"', context)
        self.assertIn("latents = self._apply_latent_stage(stage, latents)", context)
        self.assertIn("latents = cross(latents, data", context)
        self.assertIn("latents = block(latents)", context)
        self.assertIn('"symbol": "PerceiverEncoder._apply_cross"', context)
        self.assertIn('"symbol": "PerceiverEncoder._apply_latent_stage"', context)
        self.assertIn("N×M, non M²", context)
        self.assertIn("Non attribuire M² a _apply_cross", context)
        self.assertIn("encoder.py", context)
        self.assertIn("causa-effetto", context)
        self.assertIn("Esempio didattico", context)
        self.assertIn("non risultati del progetto", context)
        self.assertIn("entro 90 parole", context)
        self.assertIn("coppie orientate", context)
        self.assertIn("con sé stesso", context)
        self.assertIn("4×4=16", context)
        self.assertIn("8×8=64", context)
        self.assertIn("4 righe e 4 colonne", context)
        self.assertIn("non attribuire questa matrice a un encoder Perceiver", context)
        self.assertIn("quale stato viene aggiornato", context)
        result = response.json()
        self.assertEqual(result["engine"], "local_model")
        self.assertEqual(result["answer"], json.loads(self.result()["message"]["content"])["answer"])
        self.assertNotEqual(result["answer"], step["text"])
        self.assertTrue(any(source["path"].endswith("encoder.py") for source in result["sources"]))
        self.assertTrue(any(source["label"] == "Autoverifica originale · c1-q1" for source in result["sources"]))
        from course_content import code_excerpt
        for symbol in ("PerceiverEncoder._apply_cross", "PerceiverEncoder._apply_latent_stage"):
            code = code_excerpt(tutor.ROOT, "progetto/src/perceiver/encoder.py", symbol)
            source = next(source for source in result["sources"] if source["label"] == "Codice originale · " + symbol)
            self.assertEqual((source["line_start"], source["line_end"]), (code["start"], code["end"]))
        self.assertEqual(payload["options"]["num_predict"], 360)
        self.assertEqual(payload["options"]["num_ctx"], 6144)

    def test_missing_encoder_helper_is_a_source_error_without_model_generation(self):
        with patch("course_content.code_excerpt", side_effect=ValueError("Simbolo originale non trovato.")), \
                patch.object(tutor, "ollama_request") as model:
            response = self.request()
            model.assert_not_called()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "COURSE_SOURCE_UNAVAILABLE")

    def test_quiz_is_relevant_original_open_question_not_the_rotation_counter(self):
        with patch.object(tutor, "ollama_request", return_value=self.result()):
            first = self.request().json()
            tutor.sessions[first["session_id"]]["course_quiz_count"] = {1: 4}
            second = self.request(session_id=first["session_id"], course_step="c1-s3").json()
        chapter = get_chapter(self.course, 1)
        for result, question in ((first, chapter["quiz"][0]), (second, chapter["quiz"][1])):
            quiz = result["quiz"]
            self.assertEqual(quiz["question"], question["question"])
            self.assertEqual(quiz["lesson_id"], question["id"])
            self.assertEqual(quiz["type"], "open")
            self.assertEqual(quiz["options"], [])
        stored = tutor.sessions[first["session_id"]]["course_quiz"]
        self.assertEqual(stored["step"], "c1-s3")
        self.assertEqual(stored["expected"], chapter["quiz"][1]["expected"])

    def test_named_original_question_and_theory_limit_are_preserved(self):
        chapter = get_chapter(self.course, 29)
        original = chapter["quiz"][-1]
        with patch.object(tutor, "ollama_request", return_value=self.result()) as model:
            result = self.request(chapter=29, course_step=chapter["steps"][0]["id"], course_question=original["id"]).json()
        self.assertEqual(result["quiz"]["question"], original["question"])
        prompt = model.call_args.args[1]["messages"][0]["content"]
        self.assertIn("non c'è un modulo LSTM", prompt)
        self.assertNotIn("SIMBOLI VERIFICATI NEGLI SNIPPET FORNITI", prompt)
        self.assertFalse(any(source["path"].endswith(".py") for source in result["sources"]))

    def test_try_this_step_quiz_selects_cross_attention_question_without_model(self):
        with patch.object(tutor, "ollama_request") as model:
            response = self.request(mode="quiz", message="Fammi provare", course_step="c1-s3")
            model.assert_not_called()
        self.assertEqual(response.status_code, 200)
        result = response.json()
        original = get_chapter(self.course, 1)["quiz"][1]
        self.assertEqual(result["engine"], "activity")
        self.assertEqual(result["quiz"]["lesson_id"], "c1-q2")
        self.assertEqual(result["quiz"]["question"], original["question"])
        snapshot = tutor.sessions[result["session_id"]]["course_quiz"]
        self.assertEqual(snapshot["step"], "c1-s3")
        self.assertEqual(snapshot["expected"], original["expected"])
        with patch.object(tutor, "ollama_request") as model:
            explicit = self.request(mode="quiz", course_step="c1-s3", course_question="c1-q1",
                                    session_id=result["session_id"])
            model.assert_not_called()
        self.assertEqual(explicit.json()["quiz"]["lesson_id"], "c1-q1")

    def test_omitted_code_cannot_be_advertised_as_a_provided_function(self):
        # Uno snippet troppo lungo può essere escluso dal budget di grounding:
        # i metadati del manifesto non autorizzano a descriverlo come mostrato.
        sources = [{"path": "/sito/lezione/#cap-1", "label": "Lezione", "line_start": 60}]
        with patch.object(tutor, "course_context", return_value=("Solo il testo originale del passo.", sources)), \
                patch.object(tutor, "ollama_request", return_value=self.result()) as model:
            response = self.request()
        self.assertEqual(response.status_code, 200)
        prompt = model.call_args.args[1]["messages"][0]["content"]
        self.assertNotIn("SIMBOLI VERIFICATI NEGLI SNIPPET FORNITI", prompt)
        self.assertNotIn("PerceiverEncoder.forward", prompt)

    def test_feedback_comments_saved_question_before_any_advance(self):
        with patch.object(tutor, "ollama_request", return_value=self.result()):
            asked = self.request().json()
        move = {"scene": "course", "chapter": 1, "step": "c1-s3"}
        with patch.object(tutor, "ollama_request", return_value=self.result("Hai riconosciuto le coppie. Manca che ogni elemento viene confrontato con tutti gli altri.", move)) as model:
            response = self.request(mode="feedback", session_id=asked["session_id"], quiz_id=asked["quiz"]["id"],
                                    quiz_answer="Basta contare i pixel", course_step="c1-s59")
        result = response.json()
        self.assertEqual(response.status_code, 200)
        prompt = model.call_args.args[1]["messages"][0]["content"]
        self.assertIn("PASSO ATTUALE: c1-s2", prompt)
        self.assertIn("Basta contare i pixel", prompt)
        self.assertIn(get_chapter(self.course, 1)["quiz"][0]["expected"], prompt)
        self.assertEqual(result["feedback"]["assessment"], "review")
        self.assertIsNone(result["feedback"]["correct"])
        self.assertIsNone(result["action"])

    def test_coach_rejects_invented_or_advance_actions_but_allows_current_step(self):
        current = {"scene": "course", "chapter": 1, "step": "c1-s2"}
        actions = [current, {**current, "step": "c1-s3"}, {**current, "chapter": 2},
                   {**current, "step": "invented"}, {**current, "play_video": True}]
        for action in actions:
            with self.subTest(action=action), patch.object(tutor, "ollama_request", return_value=self.result(action=action)) as model:
                response = self.request().json()
                self.assertEqual(response["action"], current if action == current else None)
                action_schema = model.call_args.args[1]["format"]["properties"]["action"]["anyOf"][1]["properties"]
                self.assertEqual(action_schema["step"], {"const": "c1-s2"})

    def test_model_failure_does_not_substitute_reading_or_create_a_quiz(self):
        with patch.object(tutor, "ollama_request", side_effect=tutor.TutorError(503, "MODEL_UNAVAILABLE", "Ollama non risponde.")):
            response = self.request()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "MODEL_UNAVAILABLE")
        self.assertNotIn("answer", response.json())
        self.assertTrue(all("course_quiz" not in session for session in tutor.sessions.values()))

    def test_malformed_model_answer_is_an_explicit_error(self):
        with patch.object(tutor, "ollama_request", return_value={"message": {"content": "{bad json"}}):
            response = self.request()
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["error"]["code"], "INVALID_MODEL_RESPONSE")

    def test_invalid_course_context_and_noncourse_coach_are_rejected_before_model(self):
        with patch.object(tutor, "ollama_request") as model:
            for fields in ({"course_step": "c5-s1"}, {"scene": "qkv"}, {"course_question": "c5-q1"}, {"lab": {}}):
                with self.subTest(fields=fields):
                    self.assertEqual(self.request(**fields).status_code, 422)
            model.assert_not_called()

    def test_bibliography_does_not_invent_a_quiz(self):
        with patch.object(tutor, "ollama_request", return_value=self.result()) as model:
            response = self.request(chapter=52, course_step="c52-s1")
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["quiz"])
        self.assertIn("Non inventare un'autoverifica", model.call_args.args[1]["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()
