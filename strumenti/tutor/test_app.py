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

    def course_question(self, chapter=1, **fields):
        return self.question(**{"message": "Guidami in questo passo del mio progetto", "scene": "course", "chapter": chapter, **fields})

    def test_course_grounding_uses_current_original_chapter_and_code(self):
        from course_content import get_chapter, load_course
        course = load_course(tutor.ROOT)
        captured = []

        def reply(path, payload=None, timeout=10):
            captured.append(payload)
            return {"message": {"content": json.dumps({"answer": "Partiamo dal testo originale. Quale idea riconosci?", "action": None})}}

        with patch.object(tutor, "ollama_request", reply):
            for number in (1, 5, 15, 29, 52):
                with self.subTest(chapter=number):
                    chapter = get_chapter(course, number)
                    step = chapter["steps"][0]
                    response = self.course_question(number, course_step=step["id"])
                    self.assertEqual(response.status_code, 200)
                    context = captured[-1]["messages"][0]["content"]
                    self.assertIn(chapter["title"], context)
                    self.assertIn(step["text"], context)
                    self.assertIn(f"PASSO ATTUALE: {step['id']}", context)
                    self.assertNotIn("LABORATORIO DIDATTICO ATTUALE", context)
                    sources = response.json()["sources"]
                    original = [source for source in sources if "/sito/lezione" in source["path"]]
                    self.assertTrue(original)
                    self.assertTrue(all(f"cap-{number}" in source["path"] or source.get("anchor") == f"cap-{number}" for source in original))
                    if chapter.get("code"):
                        self.assertTrue(any(all(line.strip() in context for line in item["text"].splitlines() if line.strip()) for item in chapter["code"]))
                    schema = captured[-1]["format"]["properties"]["action"]["anyOf"][1]["properties"]
                    self.assertEqual(schema["chapter"]["const"], number)
                    self.assertEqual(set(schema["step"]["enum"]), {item["id"] for item in chapter["steps"][:3]})

    def test_course_rejects_unknown_cross_chapter_steps_and_lab_state_before_model(self):
        invalid = [
            {"chapter": 0}, {"chapter": 53}, {"chapter": True}, {"chapter": "5"},
            {"chapter": 1, "course_step": "c5-s1"}, {"chapter": 15, "course_step": "invented"},
            {"chapter": 5, "lab": tutor.LabState().model_dump()},
        ]
        with patch.object(tutor, "ollama_request") as model:
            for fields in invalid:
                with self.subTest(fields=fields):
                    self.assertEqual(self.course_question(**fields).status_code, 422)
            model.assert_not_called()

    def test_course_model_actions_are_limited_to_real_steps_in_current_chapter(self):
        from course_content import get_chapter, load_course
        chapter = get_chapter(load_course(tutor.ROOT), 5)
        step = chapter["steps"][0]["id"]
        actions = [
            {"scene": "course", "chapter": 5, "step": step},
            {"scene": "course", "chapter": 1, "step": "c1-s1"},
            {"scene": "course", "chapter": 5, "step": "c5-invented"},
            {"scene": "course", "chapter": True, "step": step},
            {"scene": "course", "chapter": 5, "step": step, "play_video": True},
            {"scene": "qkv", "step": 0, "focus": "q"},
        ]
        for index, action in enumerate(actions):
            with self.subTest(action=action), patch.object(tutor, "ollama_request", return_value={"message": {"content": json.dumps({"answer": "Osserva questo passo.", "action": action})}}):
                response = self.course_question(5, course_step=step).json()
                self.assertEqual(response["action"], action if index == 0 else None)

    def test_course_quiz_is_original_and_open_feedback_has_no_fake_verdict(self):
        from course_content import get_chapter, load_course
        chapter = get_chapter(load_course(tutor.ROOT), 5)
        original = chapter["quiz"][0]
        captured = []

        def reply(path, payload=None, timeout=10):
            captured.append(payload)
            return {"message": {"content": json.dumps({"answer": "La tua risposta va confrontata con la formula: considera anche K e V.", "action": None})}}

        with patch.object(tutor, "ollama_request", reply):
            asked = self.course_question(5, mode="quiz", course_question=original["id"]).json()
            self.assertEqual(captured, [])
            self.assertEqual(asked["quiz"]["question"], original["question"])
            self.assertEqual(asked["quiz"]["type"], "open")
            self.assertEqual(asked["quiz"]["options"], [])
            feedback = self.course_question(5, message="La mia risposta", mode="feedback", quiz_answer="Dipende solo da V", quiz_id=asked["quiz"]["id"], session_id=asked["session_id"]).json()
        self.assertIsNone(feedback["feedback"]["correct"])
        self.assertEqual(feedback["feedback"]["assessment"], "review")
        self.assertEqual(feedback["feedback"]["expected"], [original["expected"]])
        self.assertEqual(feedback["engine"], "local_model")
        self.assertIn(original["expected"], captured[0]["messages"][0]["content"])
        self.assertIn("Dipende solo da V", captured[0]["messages"][0]["content"])

    def test_course_feedback_keeps_question_step_snapshot_and_rejects_other_chapter(self):
        from course_content import get_chapter, load_course
        chapter = get_chapter(load_course(tutor.ROOT), 5)
        steps = chapter["steps"]
        asked = self.course_question(5, course_step=steps[0]["id"], mode="quiz").json()
        fields = {"mode": "feedback", "quiz_answer": "Una risposta aperta", "quiz_id": asked["quiz"]["id"], "session_id": asked["session_id"]}
        with patch.object(tutor, "ollama_request") as model:
            self.assertEqual(self.course_question(15, **fields).status_code, 409)
            model.assert_not_called()
        with patch.object(tutor, "ollama_request", return_value={"message": {"content": "Confrontiamo la tua risposta con quella originale."}}) as model:
            response = self.course_question(5, course_step=steps[-1]["id"], **fields)
        self.assertEqual(response.status_code, 200)
        context = model.call_args.args[1]["messages"][0]["content"]
        self.assertIn(f"PASSO ATTUALE: {steps[0]['id']}", context)
        self.assertIsNone(response.json()["action"])

    def test_course_history_does_not_mix_chapters_or_legacy_lab(self):
        captured = []

        def reply(path, payload=None, timeout=10):
            captured.append(payload)
            return {"message": {"content": json.dumps({"answer": "Seguiamo il capitolo attuale.", "action": None})}}

        with patch.object(tutor, "ollama_request", reply):
            first = self.course_question(1, message="Messaggio esclusivo capitolo uno").json()
            second = self.course_question(5, message="Messaggio esclusivo capitolo cinque", session_id=first["session_id"]).json()
            self.assertEqual(second["session_id"], first["session_id"])
            self.assertEqual(len(captured[1]["messages"]), 2)
            self.course_question(1, message="Riprendi il primo capitolo", session_id=first["session_id"])
            self.assertIn("Messaggio esclusivo capitolo uno", captured[2]["messages"][1]["content"])
            self.assertNotIn("Messaggio esclusivo capitolo cinque", json.dumps(captured[2]["messages"][1:]))
            self.question(session_id=first["session_id"])
            self.assertEqual(len(captured[3]["messages"]), 2)

    def test_course_open_feedback_model_failure_is_not_marked_correct(self):
        asked = self.course_question(5, mode="quiz").json()
        with patch.object(tutor, "ollama_request", side_effect=tutor.TutorError(503, "MODEL_UNAVAILABLE", "Ollama non risponde.")):
            response = self.course_question(5, mode="feedback", quiz_answer="Una risposta", quiz_id=asked["quiz"]["id"], session_id=asked["session_id"])
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("feedback", response.json())

    def test_course_practice_reviews_named_original_question_without_quiz_preflight(self):
        from course_content import get_chapter, load_course
        chapter = get_chapter(load_course(tutor.ROOT), 15)
        original = chapter["quiz"][1]
        with patch.object(tutor, "ollama_request", return_value={"message": {"content": "Confronta la tua spiegazione con le query originali del decoder."}}) as model:
            response = self.course_question(15, course_question=original["id"], quiz_id=original["id"], mode="feedback", quiz_answer="Le query leggono i latenti.")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["feedback"]["assessment"], "review")
        self.assertIsNone(response.json()["feedback"]["correct"])
        self.assertEqual(response.json()["feedback"]["expected"], [original["expected"]])
        self.assertIn(original["question"], model.call_args.args[1]["messages"][0]["content"])
        with patch.object(tutor, "ollama_request") as model:
            self.assertEqual(self.course_question(5, course_question=original["id"], quiz_id=original["id"], mode="feedback").status_code, 422)
            self.assertEqual(self.course_question(5, quiz_id=original["id"], mode="feedback").status_code, 409)
            model.assert_not_called()

    def test_course_selected_quiz_is_named_original_and_bibliography_has_no_fake_quiz(self):
        from course_content import get_chapter, load_course
        chapter = get_chapter(load_course(tutor.ROOT), 29)
        original = chapter["quiz"][-1]
        with patch.object(tutor, "ollama_request") as model:
            response = self.course_question(29, mode="quiz", course_question=original["id"])
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["quiz"]["question"], original["question"])
            self.assertEqual(response.json()["quiz"]["lesson_id"], original["id"])
            self.assertEqual(self.course_question(52, mode="quiz").status_code, 404)
            model.assert_not_called()

    def test_course_guide_returns_one_original_open_quiz_alongside_real_model_answer(self):
        from course_content import get_chapter, load_course
        chapter = get_chapter(load_course(tutor.ROOT), 5)
        model_answer = "Le Fourier features aggiungono al contenuto l'informazione sulla posizione."
        with patch.object(tutor, "ollama_request", return_value={"message": {"content": json.dumps({"answer": model_answer, "action": None})}}) as model:
            response = self.course_question(5, message="Guidami nel passo, poi fammi una domanda breve.").json()
        self.assertEqual(response["answer"], model_answer)
        self.assertEqual(response["engine"], "local_model")
        self.assertEqual(response["quiz"]["type"], "open")
        self.assertIn(response["quiz"]["question"], [quiz["question"] for quiz in chapter["quiz"]])
        self.assertEqual(response["quiz"]["options"], [])
        prompt = model.call_args.args[1]["messages"][0]["content"]
        self.assertIn("DOMANDA ORIGINALE CHE L'APP MOSTRA SEPARATAMENTE", prompt)
        self.assertIn("non ripeterla e non porre un'altra domanda", prompt)
        self.assertEqual(tutor.sessions[response["session_id"]]["course_quiz"]["public"]["id"], response["quiz"]["id"])

    def test_course_feedback_retains_original_hypothesis_limit_when_model_omits_it(self):
        from course_content import get_chapter, load_course
        chapter = get_chapter(load_course(tutor.ROOT), 45)
        original = chapter["quiz"][0]
        self.assertIn("È un'ipotesi, non una misura.", original["expected"])
        with patch.object(tutor, "ollama_request", return_value={"message": {"content": "La differenza deriva dall'ordine delle somme in bfloat16."}}):
            response = self.course_question(45, mode="feedback", course_question=original["id"], quiz_id=original["id"], quiz_answer="Cambiano i latenti").json()
        self.assertIn("Nei tuoi appunti: «È un'ipotesi, non una misura.»", response["answer"])
        self.assertEqual(response["feedback"]["explanation"], response["answer"])
        self.assertIsNone(response["feedback"]["correct"])
        self.assertEqual(response["feedback"]["expected"], [original["expected"]])
        qualifier = "È un'ipotesi, non una misura."
        self.assertEqual(tutor.preserve_reference_limits(qualifier, original["expected"]), qualifier)
        contradicted = "La fonte dice che non è un'ipotesi, ma una misura."
        self.assertIn("Nei tuoi appunti: «" + qualifier + "»", tutor.preserve_reference_limits(contradicted, original["expected"]))

    def test_course_schema_has_only_nearby_steps_while_server_rejects_unknown_actions(self):
        from course_content import get_chapter, load_course
        chapter = get_chapter(load_course(tutor.ROOT), 45)
        current = chapter["steps"][30]
        schema = tutor.course_turn_schema(chapter, current)
        step_ids = schema["properties"]["action"]["anyOf"][1]["properties"]["step"]["enum"]
        self.assertEqual(step_ids, [item["id"] for item in chapter["steps"][28:33]])
        self.assertLessEqual(len(step_ids), 5)
        self.assertIsNone(tutor.valid_course_action({"scene": "course", "chapter": 45, "step": "c45-invented"}, chapter))

    def test_course_request_uses_gpu_workspace_budget_and_bounds_its_history(self):
        with patch.object(tutor, "ollama_request", return_value={"message": {"content": "Una spiegazione breve del passo attuale."}}) as model:
            response = self.course_question(5, message="Spiegami questo passo").json()
            course_options = model.call_args.args[1]["options"]
            self.assertEqual(course_options["num_ctx"], 4096)
            self.assertEqual(course_options["num_batch"], 128)
            self.assertEqual(course_options["num_gpu"], tutor.NUM_GPU)
            self.question()
            self.assertEqual(model.call_args.args[1]["options"]["num_ctx"], 6144)
            self.assertNotIn("num_batch", model.call_args.args[1]["options"])
        session = tutor.sessions[response["session_id"]]
        for index in range(10):
            tutor.remember_course(session, 5, "Una domanda lunga " * 100, "Una risposta lunga " * 100)
        self.assertLessEqual(sum(len(turn["content"]) for turn in session["course_history"][5]), 1400)

    def test_direct_public_clients_are_rejected(self):
        client = TestClient(tutor.app, client=("8.8.8.8", 50123))
        self.assertEqual(client.get("/api/status").status_code, 403)


if __name__ == "__main__":
    unittest.main()
