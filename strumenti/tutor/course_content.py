"""Derive the study course from the project's original lesson, without media generation.

Only whitespace and HTML entities are normalized. Questions and reference answers
remain the original open questions: this module never invents choices or grades.
Run this module to refresh the committed JSON manifest after editing the lesson.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Iterable

LESSON_PATH = "sito/lezione/index.html"
MANIFEST_PATH = "sito/studio/media/course-project.json"
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}


@dataclass
class Node:
    tag: str
    attrs: dict[str, str]
    line: int
    order: int
    parent: "Node | None" = None
    children: list["Node | str"] = field(default_factory=list)

    @property
    def classes(self) -> set[str]:
        return set(self.attrs.get("class", "").split())

    def descendants(self) -> Iterable["Node"]:
        for child in self.children:
            if isinstance(child, Node):
                yield child
                yield from child.descendants()

    def ancestors(self) -> Iterable["Node"]:
        node = self.parent
        while node is not None:
            yield node
            node = node.parent


class LessonParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node("document", {}, 1, 0)
        self.stack = [self.root]
        self.counter = 0

    def handle_starttag(self, tag, attrs):
        self.counter += 1
        node = Node(tag, dict((key, value or "") for key, value in attrs), self.getpos()[0], self.counter, self.stack[-1])
        self.stack[-1].children.append(node)
        if tag not in VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                return

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def plain_text(node: Node) -> str:
    """Plain source wording, including inline code, with only whitespace normalized."""
    def collect(current: Node) -> str:
        if current.tag in {"script", "style"}:
            return ""
        result = []
        for child in current.children:
            if isinstance(child, str):
                result.append(child)
            elif child.tag == "br":
                result.append(" ")
            else:
                separator = " " if child.tag in {"td", "th", "p", "li", "tr", "div"} else ""
                result.append(separator + collect(child) + separator)
        return "".join(result)
    return re.sub(r"\s+", " ", collect(node)).strip()


def parse_lesson(text: str) -> list[Node]:
    parser = LessonParser()
    parser.feed(text)
    chapters = [node for node in parser.root.descendants() if node.tag == "section" and "data-chapter" in node.attrs]
    ids = [int(node.attrs["data-chapter"]) for node in chapters]
    if sorted(ids) != list(range(1, 53)) or len(ids) != 52:
        raise ValueError("La lezione deve contenere esattamente i capitoli 1–52.")
    # HTML chapter order is not numeric for the theory references; the source
    # chapter numbering defines the original guided navigation order.
    return sorted(chapters, key=lambda node: int(node.attrs["data-chapter"]))


ATTN = "progetto/src/perceiver/attention.py"
ENC = "progetto/src/perceiver/encoder.py"
MODEL = "progetto/src/perceiver/perceiver.py"
IO = "progetto/src/perceiver_io/perceiver_io.py"
PE = "progetto/src/utils/positional_encoding.py"
INPUT = "progetto/src/perceiver/input_pe.py"
TRAIN = "progetto/train.py"
BASELINE = "progetto/baseline_cnn.py"
EXPERIMENTS = "progetto/experiments.py"
CFG = "progetto/src/config/base_cfg.py"
SCHEDULE = "progetto/src/utils/scheduler.py"

# A source symbol means the actual implementation of that concept. Empty
# mappings are intentional for theory architectures absent from this project.
# Numeric spans are small, contiguous source excerpts from larger training files.
CODE_BY_CHAPTER = {
    1: [(ENC, "PerceiverEncoder.forward")],
    2: [(ATTN, "CrossAttention.forward"), (ATTN, "SelfAttention.forward")],
    3: [(MODEL, "Perceiver.forward"), (ENC, "PerceiverEncoder.forward")],
    4: [(INPUT, "InputPositionalEncoding.forward")],
    5: [(PE, "FourierPositionalEncoding.forward"), (PE, "FourierPositionalEncoding.out_dim")],
    6: [(MODEL, (40, 55)), (MODEL, "Perceiver.forward")],
    7: [(ATTN, "CrossAttention.forward")],
    8: [(ATTN, "SelfAttention.forward"), (ATTN, "FeedForward")],
    9: [(ENC, (91, 113)), (ENC, "PerceiverEncoder.forward")],
    10: [(MODEL, "Perceiver.forward")],
    11: [(TRAIN, (385, 413)), (SCHEDULE, None)],
    12: [(INPUT, "InputPositionalEncoding.forward")],
    13: [(EXPERIMENTS, "BASE"), (ENC, (86, 113)), (EXPERIMENTS, (200, 243))],
    14: [(TRAIN, (629, 675))],
    15: [(IO, "PerceiverIO.forward")],
    16: [(IO, (80, 101)), (IO, "PerceiverIO.forward")],
    17: [(MODEL, "Perceiver.forward"), (ENC, "PerceiverEncoder.forward")],
    18: [(MODEL, "Perceiver.forward"), (IO, "PerceiverIO.forward")],
    19: [(ATTN, (57, 64))],
    20: [(PE, "FourierPositionalEncoding.forward"), (INPUT, "InputPositionalEncoding.forward")],
    21: [(TRAIN, (405, 413)), (TRAIN, (629, 650))],
    22: [(ATTN, (40, 47)), (ATTN, "CrossAttention.forward")],
    23: [(ATTN, "FeedForward")],
    24: [(ATTN, (63, 73))],
    25: [(TRAIN, (385, 413)), (SCHEDULE, None)],
    27: [(ATTN, "FeedForward")],
    31: [(BASELINE, "build_resnet18_for_cifar")],
    32: [(BASELINE, "build_resnet18_for_cifar")],
    33: [(BASELINE, "build_resnet18_for_cifar")],
    34: [(ATTN, "SelfAttention.forward")],
    36: [(ATTN, "FeedForward"), (ATTN, (87, 101))],
    37: [(MODEL, (40, 55))],
    38: [(TRAIN, (385, 394))],
    39: [(TRAIN, (64, 88))],
    40: [(IO, "PerceiverIO.forward")],
    41: [(ATTN, "CrossAttention.forward"), (PE, "FourierPositionalEncoding.forward")],
    42: [(MODEL, "Perceiver.forward"), (IO, "PerceiverIO.forward")],
    43: [(MODEL, "Perceiver.forward"), (BASELINE, "build_resnet18_for_cifar")],
    44: [(EXPERIMENTS, "BASE"), (EXPERIMENTS, "EXPERIMENTS[e01_baseline]"), (EXPERIMENTS, "EXPERIMENTS[e02_permuted]"), (EXPERIMENTS, (164, 187))],
    45: [(EXPERIMENTS, "BASE"), (EXPERIMENTS, "EXPERIMENTS[e01_baseline]"), (EXPERIMENTS, "EXPERIMENTS[e02_permuted]"), (INPUT, "InputPositionalEncoding.forward"), (EXPERIMENTS, (200, 243))],
    46: [(IO, "PerceiverIO.forward"), (TRAIN, (629, 650))],
    48: [(IO, "PerceiverIO.forward"), (TRAIN, (629, 650))],
    49: [(CFG, None), (EXPERIMENTS, (164, 200))],
    50: [(TRAIN, (405, 413)), (TRAIN, (629, 650))],
    51: [(IO, "PerceiverIO.forward")],
}
CODE_NOTES = {
    4: "Questo estratto aggiunge la posizione ai token già preparati; non contiene il reshape della griglia.",
    11: "La lezione distingue la ricetta ImageNet del paper dalla configurazione del tuo training.",
    26: "Il capitolo spiega il perceptrone come teoria: non c'è un modello Perceptron separato nel progetto.",
    28: "Il capitolo spiega le RNN come teoria; il progetto riusa pesi nei blocchi Perceiver, senza implementare una RNN separata.",
    29: "Il capitolo spiega LSTM come teoria: non c'è un modulo LSTM nel modello del progetto.",
    30: "Il capitolo spiega GRU come teoria: non c'è un modulo GRU nel modello del progetto.",
    31: "La baseline del progetto costruisce ResNet-18 con torchvision; il codice delle convoluzioni è nella libreria.",
    32: "La baseline del progetto è ResNet-18; le altre ConvNet del capitolo sono riferimenti teorici.",
    33: "Il progetto adatta ResNet-18 di torchvision a CIFAR-10.",
    35: "Il Vision Transformer è un confronto teorico della lezione, non un modello implementato nel progetto.",
    38: "Nel training Perceiver LAMB usa il default weight decay 0; L1/L2 sono spiegate anche come teoria.",
    44: "BASE è la configurazione delle run CIFAR-10: 96 latenti, 4 letture e 4 blocchi sono grandezze diverse. Gli override di ciascuna run si aggiungono alla sua base.",
    45: "Le run CIFAR-10 ereditano BASE (96 latenti, 4 letture, 4 blocchi), salvo override espliciti. e01 non ha override; e02 aggiunge solo permutazione e seed della permutazione. ModelNet40 usa BASE_MODELNET.",
    47: "La scheda riporta modalità d'esame e priorità di studio, non un modulo del modello.",
    48: "Il tuo codice implementa MLM su byte con Perceiver IO; BERT e gli altri metodi restano confronti teorici.",
    49: "Il progetto sceglie le configurazioni a mano nel registro esperimenti; non implementa un ottimizzatore HPO.",
    51: "Il codice collegato è il decoder Perceiver IO parallelo; U-Net, beam search e MoE sono confronti teorici.",
    52: "La bibliografia conserva le fonti originali; non ha autoverifiche o codice da valutare.",
}


def code_excerpt(root: Path, path: str, symbol: str | tuple[int, int] | None) -> dict:
    file_path = (root / path).resolve()
    if not file_path.is_relative_to((root / "progetto").resolve()) or file_path.suffix != ".py":
        raise ValueError("I collegamenti del corso devono puntare al codice Python del progetto.")
    text = file_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if isinstance(symbol, tuple):
        start, end = symbol
    elif symbol is None:
        start, end = 1, len(lines)
    else:
        tree = ast.parse(text)
        registry_item = re.fullmatch(r"EXPERIMENTS\[([a-z0-9_]+)\]", symbol)
        if registry_item:
            assignment = next((node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "EXPERIMENTS" for target in node.targets)), None)
            values = assignment.value.elts if assignment is not None and isinstance(assignment.value, ast.List) else []
            target = next((node for node in values if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == registry_item.group(1)), None)
            if target is None:
                raise ValueError(f"Run originale non trovata: {path}:{symbol}")
            start, end = target.lineno, target.end_lineno
            return {"path": path, "start": start, "end": end, "text": "\n".join(lines[start - 1:end]), "symbol": symbol}
        parts = symbol.split(".")
        collection = tree.body
        target = None
        for part in parts:
            target = next((node for node in collection if (isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == part) or (isinstance(node, ast.Assign) and any(isinstance(assigned, ast.Name) and assigned.id == part for assigned in node.targets))), None)
            if target is None:
                raise ValueError(f"Simbolo originale non trovato: {path}:{symbol}")
            collection = getattr(target, "body", [])
        start, end = target.lineno, target.end_lineno
    if not 1 <= start <= end <= len(lines):
        raise ValueError(f"Intervallo sorgente non valido: {path}:{start}–{end}")
    return {"path": path, "start": start, "end": end, "text": "\n".join(lines[start - 1:end]), "symbol": symbol if isinstance(symbol, str) else None}


def _is_content(node: Node) -> bool:
    if node.tag in {"p", "li", "pre", "tr", "figcaption", "blockquote", "dd"}:
        return True
    if node.tag == "a" and "book-card" in node.classes:
        return True
    return node.tag == "div" and bool(node.classes & {"display-eq", "nota", "widget-note", "lab-readout", "warning", "comparison-note", "architecture-answer", "formula-meta", "code-block", "prio-legend", "vit-cost-card", "cnn-calc-result"})


def _excluded(node: Node) -> bool:
    lineage = [node, *node.ancestors()]
    if any(current.classes & {"qa-item", "qa-group", "nav-bar", "collegamenti", "lab-toggle", "lab-step-rail"} for current in lineage):
        return True
    return any(current.tag in {"script", "style", "button", "select", "option"} for current in lineage)


def _source(node: Node, chapter: int) -> dict:
    return {"path": LESSON_PATH, "line": node.line, "anchor": f"cap-{chapter}"}


def _experiment_code(root: Path, node: Node, chapter_id: int) -> tuple[str | None, list[dict]]:
    """For a run's source card, use its actual base and only its own overrides.

    Adjacent cards in Tab. 2 share a heading but e03 changes T while e02 does
    not. Mixing their code would provide contradictory context to the tutor.
    """
    if chapter_id != 45:
        return None, []
    card = next((parent for parent in node.ancestors() if "dataset-card" in parent.classes), None)
    name = next((child for child in card.descendants() if "dataset-name" in child.classes), None) if card else None
    code = next((child for child in name.descendants() if child.tag == "code"), None) if name else None
    if code is None and node.tag == "tr":
        first_cell = next((child for child in node.children if isinstance(child, Node) and child.tag == "td"), None)
        code = next((child for child in first_cell.descendants() if child.tag == "code"), None) if first_cell else None
    run_id = plain_text(code) if code else ""
    if not re.fullmatch(r"(?:e|mn)\d+[a-z0-9_]+", run_id):
        return None, []
    base = "BASE_MODELNET" if run_id.startswith("mn") else "BASE"
    snippets = [code_excerpt(root, EXPERIMENTS, base), code_excerpt(root, EXPERIMENTS, f"EXPERIMENTS[{run_id}]")]
    if run_id in {"e02_permuted", "e03_learned_pe", "e04_learned_pe_permuted", "e29_no_pe"}:
        snippets.append(code_excerpt(root, INPUT, "InputPositionalEncoding.forward"))
    return run_id, snippets


def _animation_for(node: Node, chapter: int, lab_nodes: list[Node]) -> dict:
    # Only use a lab inside the same original heading section. Nearness alone
    # must not attach a Fourier diagram to an unrelated explanation.
    owner = next((parent for parent in [node, *node.ancestors()] if "data-lab" in parent.attrs), None)
    if owner is None:
        chapter_node = next(parent for parent in node.ancestors() if "data-chapter" in parent.attrs)
        headings = [item for item in chapter_node.descendants() if item.tag in {"h2", "h3"} and not any("data-lab" in parent.attrs for parent in item.ancestors())]
        before = max((item.order for item in headings if item.order < node.order), default=0)
        after = min((item.order for item in headings if item.order > node.order), default=10**9)
        candidates = [lab for lab in lab_nodes if before < lab.order < after and abs(lab.line - node.line) <= 28]
        owner = min(candidates, key=lambda lab: abs(lab.line - node.line), default=None)
    animation = {"kind": "original", "href": f"../lezione/#cap-{chapter}"}
    if owner is not None:
        animation["labId"] = owner.attrs["data-lab"]
        if owner.attrs.get("data-preset"):
            animation["preset"] = owner.attrs["data-preset"]
    else:
        widget = next((parent for parent in [node, *node.ancestors()] if parent.classes & {"widget", "flow-lab"}), None)
        if widget is not None:
            first_id = next((item.attrs["id"] for item in widget.descendants() if "id" in item.attrs), None)
            if first_id:
                animation["targetId"] = first_id
        elif chapter == 42:
            # Chapter 42 uses an older native flow widget rather than data-lab.
            animation["targetId"] = "flowPanel"
    return animation


def build_course(root: str | Path) -> dict:
    root = Path(root).resolve()
    source = (root / LESSON_PATH).read_text(encoding="utf-8")
    chapters = []
    for chapter_node in parse_lesson(source):
        chapter_id = int(chapter_node.attrs["data-chapter"])
        nodes = list(chapter_node.descendants())
        heading = next(node for node in nodes if node.tag == "h1")
        title = plain_text(heading)
        labs = [node for node in nodes if "data-lab" in node.attrs]
        excerpts = [code_excerpt(root, *spec) for spec in CODE_BY_CHAPTER.get(chapter_id, [])]
        steps = []
        current_title = title
        for node in nodes:
            if node.tag in {"h2", "h3"} and not _excluded(node):
                current_title = plain_text(node) or current_title
            if not _is_content(node) or _excluded(node):
                continue
            # Pick the outer contiguous block once, so nested paragraphs are
            # not repeated and a list item's introductory words aren't lost.
            if any(_is_content(parent) and not _excluded(parent) for parent in node.ancestors() if parent is not chapter_node):
                continue
            text = plain_text(node)
            if not text:
                continue
            animation = _animation_for(node, chapter_id, labs)
            run_id, run_code = _experiment_code(root, node, chapter_id)
            step = {"id": f"c{chapter_id}-s{len(steps) + 1}", "title": current_title, "text": text, "source": _source(node, chapter_id), "code": run_code or (excerpts if not steps else []), "animation": animation}
            if run_id:
                step["experimentId"] = run_id
            links = [{"href": linked.attrs["href"], "label": plain_text(linked)} for linked in [node, *node.descendants()] if linked.tag == "a" and linked.attrs.get("href")]
            if links:
                step["links"] = links
            steps.append(step)
        quizzes = []
        for qa in [node for node in nodes if "qa-item" in node.classes]:
            summary = next((child for child in qa.descendants() if child.tag == "summary"), None)
            answers = [child for child in qa.descendants() if "qa-a" in child.classes]
            if summary is None:
                raise ValueError(f"Domanda senza testo originale nel capitolo {chapter_id}.")
            expected = " ".join(plain_text(answer) for answer in answers).strip()
            quiz = {"id": f"c{chapter_id}-q{len(quizzes) + 1}", "type": "open", "question": plain_text(summary), "options": [], "correct": [], "expected": expected, "explanation": expected, "source": _source(qa, chapter_id), "evaluation": "model-reference" if expected else "self-review"}
            if steps:
                previous = [step for step in steps if step["source"]["line"] <= qa.line]
                quiz["stepId"] = (previous[-1] if previous else steps[0])["id"]
            quizzes.append(quiz)
        if not steps:
            raise ValueError(f"Capitolo {chapter_id} privo di testo di studio originale.")
        animations = [{"kind": "original", "labId": node.attrs["data-lab"], "preset": node.attrs.get("data-preset"), "source": _source(node, chapter_id), "href": f"../lezione/#cap-{chapter_id}"} for node in labs]
        widgets = [node for node in nodes if node.classes & {"widget", "flow-lab"}]
        for widget in widgets:
            target_id = next((node.attrs["id"] for node in widget.descendants() if "id" in node.attrs), None)
            if target_id:
                animations.append({"kind": "original", "targetId": target_id, "source": _source(widget, chapter_id), "href": f"../lezione/#cap-{chapter_id}"})
        chapter = {"id": chapter_id, "title": title, "anchor": f"cap-{chapter_id}", "originalHref": f"../lezione/#cap-{chapter_id}", "source": _source(chapter_node, chapter_id), "steps": steps, "quiz": quizzes, "code": excerpts, "animations": animations, "codeNote": CODE_NOTES.get(chapter_id, ""), "quizNote": "Domande aperte e risposte di riferimento originali; il feedback del tutor è una valutazione AI." if quizzes else "Questo capitolo non contiene autoverifiche originali. Puoi ripassarlo e parlarne con il tutor."}
        chapters.append(chapter)
    code_paths = sorted({code["path"] for chapter in chapters for code in chapter["code"]} | {code["path"] for chapter in chapters for step in chapter["steps"] for code in step["code"]})
    code_hashes = {path: hashlib.sha256((root / path).read_text(encoding="utf-8").encode("utf-8")).hexdigest() for path in code_paths}
    return {"version": 1, "title": "Il tuo progetto, capitolo per capitolo", "source": {"path": LESSON_PATH, "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(), "code_sha256": code_hashes}, "chapterCount": len(chapters), "chapters": chapters, "media": {"generated": False, "autoplay": False, "note": "Il manifesto contiene testo, codice e collegamenti alle animazioni originali; non genera o riproduce video."}}


def load_course(root: str | Path) -> dict:
    root = Path(root).resolve()
    file = root / MANIFEST_PATH
    if file.is_file():
        course = json.loads(file.read_text(encoding="utf-8"))
        source = (root / LESSON_PATH).read_text(encoding="utf-8")
        hashes = course.get("source", {}).get("code_sha256", {})
        code_current = bool(hashes) and all((root / path).is_file() and hashlib.sha256((root / path).read_text(encoding="utf-8").encode("utf-8")).hexdigest() == digest for path, digest in hashes.items())
        if code_current and course.get("source", {}).get("sha256") == hashlib.sha256(source.encode("utf-8")).hexdigest():
            return course
    # A changed lesson must not leave the tutor grounded in stale explanations.
    # Building here is read-only: writing the committed manifest is explicit CLI.
    return build_course(root)


def get_chapter(course: dict, chapter_id: int) -> dict:
    if isinstance(chapter_id, bool) or not isinstance(chapter_id, int):
        raise ValueError("Capitolo non valido.")
    chapter = next((item for item in course["chapters"] if item["id"] == chapter_id), None)
    if chapter is None:
        raise ValueError("Capitolo non valido.")
    return chapter


def chapter_grounding(course: dict, chapter_id: int, step_id: str | None = None, max_chars: int = 6000) -> tuple[str, list[dict]]:
    chapter = get_chapter(course, chapter_id)
    step = next((item for item in chapter["steps"] if item["id"] == step_id), None) if step_id else chapter["steps"][0]
    if step_id and step is None:
        raise ValueError("Passo non appartenente al capitolo.")
    max_chars = max(1000, min(16000, max_chars))
    context = [f"Capitolo {chapter_id}: {chapter['title']}", f"Passo originale {step['id']}: {step['title']}\n{step['text']}"]
    sources = [{"path": LESSON_PATH, "label": f"Capitolo {chapter_id} · {chapter['title']}", "line_start": step["source"]["line"], "anchor": chapter["anchor"]}]
    if chapter["codeNote"]:
        context.append("Limite del collegamento al codice: " + chapter["codeNote"])
    # Active step gets priority; neighbouring original blocks explain its
    # immediate context. Reference answers are supplied by the chat quiz mode.
    index = chapter["steps"].index(step)
    neighbours = chapter["steps"][max(0, index - 1):index] + chapter["steps"][index + 1:index + 3]
    if step.get("experimentId"):
        context.append(f"Run attuale negli appunti: {step['experimentId']}. La base e gli override seguenti appartengono a questa run.")
        neighbours = [neighbour for neighbour in neighbours if neighbour.get("experimentId") == step["experimentId"]]
    for neighbour in neighbours:
        candidate = f"Contesto originale {neighbour['id']}: {neighbour['text']}"
        if sum(len(part) for part in context) + len(candidate) < max_chars - 1500:
            context.append(candidate)
    for code in step["code"] or chapter["code"]:
        numbered = "\n".join(f"{number}: {line}" for number, line in enumerate(code["text"].splitlines(), code["start"]))
        snippet = f"Codice reale {code['path']}:{code['start']}–{code['end']}\n{numbered}"
        if sum(len(part) for part in context) + len(snippet) <= max_chars:
            context.append(snippet)
            sources.append({"path": code["path"], "label": f"Codice originale · {code['path'].rsplit('/', 1)[-1]}", "line_start": code["start"], "line_end": code["end"]})
    return "\n\n".join(context), sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Estrae il percorso completo senza generare audio o video.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--check", action="store_true", help="Controlla il manifesto salvato senza scrivere.")
    args = parser.parse_args()
    course = build_course(args.root)
    target = args.root / MANIFEST_PATH
    if args.check:
        if not target.is_file() or json.loads(target.read_text(encoding="utf-8")) != course:
            raise SystemExit("Il manifesto non è aggiornato; eseguire course_content.py.")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(course, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"chapters": len(course["chapters"]), "steps": sum(len(item["steps"]) for item in course["chapters"]), "questions": sum(len(item["quiz"]) for item in course["chapters"]), "original_labs": sum("labId" in animation for item in course["chapters"] for animation in item["animations"]), "original_widgets": sum("targetId" in animation for item in course["chapters"] for animation in item["animations"]), "video_generated": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
