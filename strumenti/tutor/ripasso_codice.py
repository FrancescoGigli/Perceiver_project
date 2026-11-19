"""Prepara gli estratti Python mostrati nel ripasso illustrato su GitHub Pages."""

import argparse
import json
from pathlib import Path

from course_content import code_excerpt


ROOT = Path(__file__).resolve().parents[2]
LESSONS = Path(__file__).with_suffix(".json")
OUTPUT = ROOT / "sito/studio/ripasso-codice.js"


def extract_step(step):
    path = step["path"]
    block = code_excerpt(ROOT, path, step.get("scope"))
    lines = block["text"].splitlines()
    first = step["first"]
    last = step["last"]
    starts = [i for i, line in enumerate(lines) if line.strip() == first]
    if len(starts) != 1:
        raise ValueError(f"{path} ({step['label']}): trovati {len(starts)} inizi per {first!r}")
    start = starts[0]
    end = next((i for i in range(start, len(lines)) if lines[i].strip() == last), None)
    if end is None:
        raise ValueError(f"{path} ({step['label']}): manca la fine {last!r}")
    return {
        "label": step["label"],
        "path": path,
        "start": block["start"] + start,
        "end": block["start"] + end,
        "code": "\n".join(lines[start:end + 1]),
        "input": step["input"],
        "output": step["output"],
        "explain": step["explain"],
    }


def build():
    stages = {}
    for lesson in json.loads(LESSONS.read_text(encoding="utf-8")):
        slug = lesson["slug"]
        if slug in stages:
            raise ValueError(f"Tappa ripetuta: {slug}")
        stages[slug] = {"orale": lesson["orale"], "steps": [extract_step(step) for step in lesson["steps"]]}
    return ("// Generato da strumenti/tutor/ripasso_codice.py.\n"
            "window.perceiverRipassoCodice = " + json.dumps(stages, indent=2, ensure_ascii=False) + ";\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verifica gli estratti salvati senza scrivere.")
    args = parser.parse_args()
    output = build()
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != output:
            raise SystemExit("Estratti da aggiornare con python strumenti/tutor/ripasso_codice.py")
    else:
        OUTPUT.write_text(output, encoding="utf-8")
    print("Estratti del ripasso allineati ai sorgenti Python.")
