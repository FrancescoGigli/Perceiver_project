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

    def test_perceiver_io_question_retrieves_decoder_and_original_lesson(self):
        captured = []

        def reply(path, payload=None, timeout=10):
            captured.append(payload)
            return {"message": {"content": "Le output queries leggono i latenti attraverso il decoder."}}

        with patch.object(tutor, "ollama_request", reply):
            response = self.question(message="Nel mio codice Perceiver IO da dove vengono le output queries del decoder?")
        self.assertEqual(response.status_code, 200)
        context = captured[0]["messages"][0]["content"]
        self.assertIn("self.output_queries", context)
        self.assertIn("perceiver_io.py", context)
        self.assertTrue(any("cap-15" in s["path"] or "cap-16" in s["path"] for s in response.json()["sources"]))

    def test_other_project_topics_do_not_inherit_lab_numbers_or_move_encoder(self):
        captured = []

        def reply(path, payload=None, timeout=10):
            captured.append(payload)
            return {"message": {"content": json.dumps({"answer": "Le query di output sono parametri appresi.", "action": {"scene": "qkv", "step": 6, "focus": "output"}})}}

        with patch.object(tutor, "ollama_request", reply):
            response = self.question(message="Nel decoder Perceiver IO le output queries sono apprese?", lab={"query": [7, 9]})
        context = captured[0]["messages"][0]["content"]
        self.assertNotIn('"query": [7.0, 9.0]', context)
        self.assertIn("self.output_queries = nn.Parameter", context)
        self.assertIsNone(response.json()["action"])

    def test_tutor_sees_lab_mistake_recomputed_instead_of_client_correct_flag(self):
        captured = []

        def reply(path, payload=None, timeout=10):
            captured.append(payload)
            return {"message": {"content": "Guarda q_proj: la sorgente è x_latent. Da dove nasce Q?"}}

        interaction = {"stage": 0, "focus": "q", "finished": False, "lastAnswer": {"selected": "input", "correct": True, "attempts": 2}}
        with patch.object(tutor, "ollama_request", reply):
            response = self.question(message="Spiegami il mio errore", lab=tutor.LabState().model_dump(), interaction=interaction)
        self.assertEqual(response.status_code, 200)
        context = captured[0]["messages"][0]["content"]
        self.assertIn("INTERAZIONE DEL LABORATORIO", context)
        self.assertIn('"selected": "input", "correct": false', context)
        self.assertNotIn('"selected": "input", "correct": true', context)

    def test_other_topics_omit_previous_lab_mistake(self):
        captured = []

        def reply(path, payload=None, timeout=10):
            captured.append(payload)
            return {"message": {"content": "Le output queries sono parametri appresi."}}

        interaction = {"stage": 2, "focus": "v", "finished": False, "lastAnswer": {"selected": "weights", "correct": True, "attempts": 1}}
        with patch.object(tutor, "ollama_request", reply):
            self.question(message="Da dove vengono le output queries del decoder IO?", lab=tutor.LabState().model_dump(), interaction=interaction)
        self.assertNotIn("INTERAZIONE DEL LABORATORIO", captured[0]["messages"][0]["content"])

    def test_lab_interaction_math_and_value_answer_ignore_browser_grading(self):
        lab = tutor.LabState(query=(0, 1))
        math = tutor.LabInteraction(stage=1, focus="weights", finished=False, lastAnswer={"selected": "2", "correct": False, "attempts": 1})
        values = tutor.LabInteraction(stage=2, focus="v", finished=False, lastAnswer={"selected": "weights", "correct": True, "attempts": 1})
        self.assertTrue(tutor.verified_interaction(math, lab)["lastAnswer"]["correct"])
        self.assertFalse(tutor.verified_interaction(values, lab)["lastAnswer"]["correct"])

    def test_invalid_interaction_cannot_reach_model(self):
        invalid = [
            {"stage": 3, "focus": "q", "finished": False},
            {"stage": 0, "focus": "shell", "finished": False},
            {"stage": 0, "focus": "q", "finished": False, "lastAnswer": {"selected": "1", "correct": True, "attempts": 1}},
            {"stage": 1, "focus": "weights", "finished": False, "lastAnswer": {"selected": "1", "correct": True, "attempts": 1}},
        ]
        with patch.object(tutor, "ollama_request") as model:
            for interaction in invalid:
                with self.subTest(interaction=interaction):
                    self.assertEqual(self.question(interaction=interaction).status_code, 422)
            model.assert_not_called()

    def test_guided_lab_quiz_does_not_retrieve_experiment_registry_from_button_text(self):
        response = self.question(message="Fammi una sola domanda sull'esperimento attuale", mode="quiz", lab=tutor.LabState().model_dump()).json()
        self.assertFalse(any(source["path"] == "/progetto/experiments.py" for source in response["sources"]))
        self.assertTrue(any(source["path"] == "/progetto/src/perceiver/attention.py" for source in response["sources"]))

    def test_experiment_question_uses_actual_cifar_setup_and_no_pe_result(self):
        captured = []

        def reply(path, payload=None, timeout=10):
            captured.append(payload)
            return {"message": {"content": "La configurazione sperimentale usa 96 latenti; senza PE il risultato è 32,36%."}}

        with patch.object(tutor, "ollama_request", reply):
            response = self.question(message="Su CIFAR-10 quanti latenti e dimensioni uso negli esperimenti e cosa succede senza positional encoding e29_no_pe?")
        context = captured[0]["messages"][0]["content"]
        self.assertIn("32,36", context)
        self.assertIn("96", context)
        self.assertIn("384", context)
        self.assertTrue(any("experiments.py" in s["path"] for s in response.json()["sources"]))
        self.assertIn('"--lr", "0.004"', context)
        self.assertNotIn("BASE_IO_CIFAR", context)

    def test_explicit_ui_request_has_action_even_if_model_only_returns_text(self):
        with patch.object(tutor, "ollama_request") as model:
            response = self.question(message="Mostra V")
            model.assert_not_called()
        self.assertEqual(response.json()["action"]["focus"], "v")
        self.assertEqual(response.json()["engine"], "activity")

    def test_guided_quiz_wrong_answer_gets_explanation_from_actual_lab(self):
        lab = {"query": [0, 1], "keys": [[1, 0], [0, 1], [-1, 0]], "values": [8, 3, 1], "scale": True}
        with patch.object(tutor, "ollama_request") as model:
            quiz_response = self.question(message="Fammi una domanda", mode="quiz", lab=lab).json()
            feedback = self.question(message="Input 1", mode="feedback", quiz_answer="1", quiz_id=quiz_response["quiz"]["id"], session_id=quiz_response["session_id"]).json()
            model.assert_not_called()
        self.assertFalse(feedback["feedback"]["correct"])
        self.assertEqual(feedback["feedback"]["expected"], ["Input 2"])
        self.assertIn("0.707", feedback["feedback"]["explanation"])
        self.assertIn("softmax", feedback["answer"])
        self.assertEqual(feedback["action"]["focus"], "weights")

    def test_quiz_uses_saved_values_and_accepts_equal_maxima(self):
        lab = {"query": [1, 0], "keys": [[1, 0], [1, 0], [-1, 0]], "values": [8, 3, 1], "scale": True}
        quiz = self.question(message="Una domanda", mode="quiz", lab=lab).json()
        # Cambiare il laboratorio dopo la domanda non cambia la sua soluzione.
        changed = {**lab, "query": [-1, 0]}
        feedback = self.question(message="2", mode="feedback", quiz_answer="2", quiz_id=quiz["quiz"]["id"], session_id=quiz["session_id"], lab=changed).json()
        self.assertTrue(feedback["feedback"]["correct"])
        self.assertEqual(feedback["feedback"]["expected"], ["Input 1", "Input 2"])

    def test_only_values_change_output_not_attention_weights(self):
        first = tutor.lab_result(tutor.LabState())
        second = tutor.lab_result(tutor.LabState(values=(0, 20, -20)))
        self.assertEqual(first["weights"], second["weights"])
        self.assertNotAlmostEqual(first["output"], second["output"])
        self.assertAlmostEqual(sum(first["weights"]), 1)
        self.assertAlmostEqual(first["scores"][0], 2 ** -0.5)
        self.assertLessEqual(first["output"], 8)
        self.assertGreaterEqual(first["output"], 1)

    def test_quiz_answer_requires_current_session_question(self):
        response = self.question(message="1", mode="feedback", quiz_answer="1", quiz_id="invented")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "QUIZ_EXPIRED")

    def test_lab_rejects_nonfinite_wrong_shape_and_fake_weights(self):
        invalid = [
            {"query": [1]}, {"keys": [[1, 0]]}, {"values": [float("inf"), 3, 1]},
            {"query": [21, 0]}, {"weights": [1, 0, 0]},
        ]
        with patch.object(tutor, "ollama_request") as model:
            for lab in invalid:
                with self.subTest(lab=lab):
                    # Raw JSON lets the server exercise nonfinite number rejection.
                    response = self.client.post("/api/chat", content=json.dumps({"message": "Spiega", "lab": lab}), headers={"Content-Type": "application/json"})
                    self.assertEqual(response.status_code, 422)
            model.assert_not_called()

    def test_schema_requested_and_actions_cannot_execute_or_mutate_lab(self):
        unsafe = [
            {"scene": "qkv", "step": 1, "focus": "eval"},
            {"scene": "qkv", "step": 1, "focus": "q", "query": [20, 20]},
            {"scene": "qkv", "step": 1, "focus": {}},
            {"scene": "qkv", "step": True, "focus": "q"},
        ]
        for action in unsafe:
            with self.subTest(action=action), patch.object(tutor, "ollama_request", return_value={"message": {"content": json.dumps({"answer": "Osserva Q.", "action": action})}}) as model:
                response = self.question()
                self.assertEqual(response.status_code, 200)
                self.assertIsNone(response.json()["action"])
                self.assertEqual(model.call_args.args[1]["format"]["type"], "object")

    def test_model_cannot_fabricate_line_number_attributions(self):
        answer = "Q viene dai latenti (vedi righe 999–1003 di `attention.py`), come nel capitolo 88."
        with patch.object(tutor, "ollama_request", return_value={"message": {"content": answer}}):
            response = self.question().json()
        self.assertNotIn("999", response["answer"])
        self.assertNotIn("1003", response["answer"])
        self.assertNotIn("88", response["answer"])
        self.assertTrue(any("righe 41–70" in s["label"] for s in response["sources"]))

    def test_retrieval_covers_all_chapters_without_private_material(self):
        corpus = tutor.corpus()
        self.assertEqual({p["chapter"] for p in corpus if p["kind"] == "lesson"}, set(range(1, 53)))
        paths = {p["path"] for p in corpus}
        self.assertFalse(any(part in path for path in paths for part in ("_copione", "backups", "transcript", "/progetto/logs/", "/progetto/data/")))
        self.assertIn("/progetto/src/perceiver_io/perceiver_io.py", paths)
        self.assertEqual(self.client.get("/progetto/experiments.py").status_code, 200)
        self.assertEqual(self.client.get("/progetto/README.md").status_code, 200)

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
