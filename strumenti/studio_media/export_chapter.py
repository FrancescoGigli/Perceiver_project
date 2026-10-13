"""Prepare a source-verified chapter video; render only after explicit --render.

The default is a JSON plan: no voice model, browser, audio, video or FFmpeg starts.
All generated files must stay outside the repository. Optional media libraries are
imported only by render_plan(), using the existing local Kokoro runtime.
"""
from __future__ import annotations

import argparse
import hashlib
from html import escape
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO / "sito/studio/media/course-project.json"
WIDTH, HEIGHT = 1280, 720
FONT = "Segoe UI"
CODE_FONT = "Consolas"
INDIGO, ORANGE, MUTED = "#1a237e", "#e65100", "#6c7081"
ALLOWED_CODE_FILES = {
    "progetto/train.py", "progetto/baseline_cnn.py", "progetto/experiments.py",
}


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class PlainHTML(HTMLParser):
    """Match the visible lesson text, excluding executable/style contents."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        elif not self.hidden and tag in {"p", "li", "div", "section", "h1", "h2", "h3", "br", "tr", "td", "th"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        elif not self.hidden and tag in {"p", "li", "div", "section", "h1", "h2", "h3", "tr", "td", "th"}:
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def source_path(root: Path, relative: str, *, code: bool = False) -> Path:
    """Read only lesson/source code paths, never archives or personal drafts."""
    if not isinstance(relative, str) or not relative:
        raise ValueError("Una fonte deve avere un percorso relativo valido.")
    relative = relative.replace("\\", "/")
    if relative.startswith("/") or ":" in relative or ".." in relative.split("/"):
        raise ValueError(f"Percorso fonte non consentito: {relative}")
    if code:
        allowed = relative in ALLOWED_CODE_FILES or (
            relative.startswith("progetto/src/") and relative.endswith(".py")
        )
    else:
        allowed = relative == "sito/lezione/index.html"
    if not allowed:
        raise ValueError(f"Fonte fuori dalla lezione e dal codice autorizzati: {relative}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Fonte esterna al progetto: {relative}")
    return resolved


def verify_sources(manifest: dict[str, Any], chapter: dict[str, Any], root: Path) -> None:
    source = manifest.get("source", {})
    lesson_path = source_path(root, source.get("path", ""))
    # The course builder hashes UTF-8 text with universal newlines, so the same
    # source verifies on Windows (CRLF) and GitHub Pages (LF).
    lesson = lesson_path.read_text(encoding="utf-8")
    digest = hashlib.sha256(lesson.encode("utf-8")).hexdigest()
    if not source.get("sha256") or digest != source["sha256"]:
        raise ValueError("La lezione è cambiata: rigenera il manifest del corso prima di esportare.")
    parser = PlainHTML()
    parser.feed(lesson)
    plain = normalise("".join(parser.parts))
    for step in chapter.get("steps", []):
        quote = normalise(step.get("text", ""))
        if not quote or quote not in plain:
            raise ValueError(f"Citazione non presente nella lezione: {step.get('id', '?')}")
        if step.get("source", {}).get("path") != source["path"]:
            raise ValueError(f"Fonte di capitolo inattesa: {step.get('id', '?')}")
        for snippet in step.get("code", []):
            file_path = source_path(root, snippet.get("path", ""), code=True)
            start, end = snippet.get("start"), snippet.get("end")
            lines = file_path.read_text(encoding="utf-8-sig").splitlines()
            if type(start) is not int or type(end) is not int or not (1 <= start <= end <= len(lines)):
                raise ValueError(f"Intervallo di codice non valido: {snippet.get('path')}")
            actual = "\n".join(lines[start - 1:end])
            if actual != snippet.get("text", "").replace("\r\n", "\n"):
                raise ValueError(f"Codice cambiato: {snippet['path']}:{start}")
    for quiz in chapter.get("quiz", []):
        for field in ("question", "expected"):
            value = normalise(quiz.get(field, ""))
            if not value or value not in plain:
                raise ValueError(f"{field} non verificabile nella lezione: {quiz.get('id', '?')}")


def split_quote(text: str, limit: int = 260) -> list[str]:
    """Split at whitespace only: joining chunks restores the exact normalised text."""
    text = normalise(text)
    chunks: list[str] = []
    pending = ""
    for word in text.split():
        if pending and len(pending) + len(word) + 1 > limit:
            chunks.append(pending)
            pending = ""
        pending = f"{pending} {word}".strip()
    if pending:
        chunks.append(pending)
    return chunks


def code_pages(snippets: list[dict[str, Any]], row_limit: int = 8) -> list[dict[str, Any]]:
    """Paginate actual code, preserving line numbers and all characters."""
    pages: list[dict[str, Any]] = []
    for snippet in snippets:
        rows: list[dict[str, Any]] = []
        for offset, line in enumerate(snippet["text"].splitlines()):
            # Whitespace remains verbatim in the JSON. SVG uses xml:space=preserve.
            parts = textwrap.wrap(line, 90, expand_tabs=False, replace_whitespace=False, drop_whitespace=False) or [""]
            for part_index, part in enumerate(parts):
                rows.append({"line": snippet["start"] + offset, "text": part, "continuation": part_index > 0})
        for start in range(0, len(rows), row_limit):
            pages.append({"path": snippet["path"], "rows": rows[start:start + row_limit]})
    return pages


def build_plan(manifest: dict[str, Any], chapter_id: int, root: Path = REPO,
               voice: str = "im_nicola", speed: float = .86) -> dict[str, Any]:
    if type(chapter_id) is not int:
        raise ValueError("Il capitolo deve essere un numero intero.")
    chapter = next((item for item in manifest.get("chapters", []) if item.get("id") == chapter_id), None)
    if chapter is None:
        raise ValueError(f"Capitolo {chapter_id} non presente nel manifest.")
    if voice not in {"im_nicola", "if_sara"} or not (.5 <= speed <= 1.3):
        raise ValueError("Voce o velocità non valida.")
    if not chapter.get("steps"):
        raise ValueError("Il capitolo non contiene spiegazioni da esportare.")
    verify_sources(manifest, chapter, root)
    scenes: list[dict[str, Any]] = []
    for step in chapter["steps"]:
        quotes = split_quote(step["text"])
        pages = code_pages(step.get("code", []))
        for index, quote in enumerate(quotes):
            page = pages[index % len(pages)] if pages else None
            scenes.append({
                "id": f"{step['id']}-part-{index + 1}", "kind": "explanation",
                "title": step["title"], "text": quote, "speechText": quote,
                "source": step["source"], "links": step.get("links", []),
                "code": step.get("code", []), "codePage": page,
                "originalAnimation": step.get("animation", {"kind": "text"}),
                "animationExport": "code_highlight" if page else "text_highlight",
                "pauseAfterSeconds": .7,
            })
        # Additional code pages get their own silent reading panels, instead of truncation.
        for index in range(len(quotes), len(pages)):
            scenes.append({
                "id": f"{step['id']}-code-{index + 1}", "kind": "code",
                "title": step["title"], "text": "", "speechText": "",
                "source": step["source"], "links": step.get("links", []),
                "code": step.get("code", []), "codePage": pages[index],
                "originalAnimation": step.get("animation", {"kind": "code"}),
                "animationExport": "code_highlight", "readingSeconds": 8,
                "pauseAfterSeconds": 0,
            })
    for quiz in chapter.get("quiz", []):
        for kind, field in (("question", "question"), ("answer", "expected")):
            for index, quote in enumerate(split_quote(quiz[field])):
                scenes.append({
                    "id": f"{quiz['id']}-{kind}-{index + 1}", "kind": kind,
                    "title": "Fermati e rispondi" if kind == "question" else "Confronta con la tua spiegazione",
                    "text": quote, "speechText": quote, "source": quiz["source"],
                    "code": [], "codePage": None, "originalAnimation": {"kind": "text"},
                    "animationExport": "text_highlight", "pauseAfterSeconds": 8 if kind == "question" else 1,
                })
    return {
        "version": 1, "chapter": chapter_id, "title": chapter["title"],
        "originalHref": chapter.get("originalHref"),
        "source": manifest["source"], "sourceVerified": True, "mediaGenerated": False,
        "mode": "plan", "voice": voice, "speechSpeed": speed,
        "format": {"width": WIDTH, "height": HEIGHT, "fps": 20, "videoCodec": "h264",
                   "audioCodec": "aac", "pixelFormat": "yuv420p", "fastStart": True,
                   "subtitles": ["vtt", "srt", "mp4-mov_text"]},
        "style": {"font": FONT, "codeFont": CODE_FONT, "accent": INDIGO, "highlight": ORANGE},
        "animationNote": "Evidenziazioni animate delle citazioni e del codice originali. I widget interattivi originali restano nella piattaforma e sono indicati in originalAnimation; non sono simulati dal video.",
        "interactionNote": "Il video è lineare: le domande hanno una pausa per rispondere e poi la spiegazione. Dialogo col tutor, correzione e controlli si usano nella piattaforma.",
        "timingsVerified": False, "scenes": scenes,
    }


def output_directory(directory: Path, root: Path = REPO) -> Path:
    resolved = directory.expanduser().resolve()
    if resolved.is_relative_to(root.resolve()):
        raise ValueError("Scegli una cartella fuori dal repository, per esempio Downloads\\Perceiver-video.")
    return resolved


def stamp(seconds: float, sep: str = ".") -> str:
    ms = round(seconds * 1000)
    hour, ms = divmod(ms, 3600000)
    minute, ms = divmod(ms, 60000)
    second, ms = divmod(ms, 1000)
    return f"{hour:02}:{minute:02}:{second:02}{sep}{ms:03}"


def subtitle_text(scenes: list[dict[str, Any]]) -> tuple[str, str]:
    vtt, srt = "WEBVTT\n\n", ""
    cue = 0
    for scene in scenes:
        if not scene["text"]:
            continue
        cue += 1
        text = "\n".join(textwrap.wrap(scene["text"], width=66, break_long_words=False, break_on_hyphens=False))
        vtt += f"{cue}\n{stamp(scene['start'])} --> {stamp(scene['end'])}\n{text}\n\n"
        srt += f"{cue}\n{stamp(scene['start'], ',')} --> {stamp(scene['end'], ',')}\n{text}\n\n"
    return vtt, srt


def scene_svg(plan: dict[str, Any], scene: dict[str, Any], index: int) -> str:
    """Pure vector reading panel, using the existing Segoe UI/Consolas lesson style."""
    def text(x, y, value, size=24, fill=INDIGO, family=FONT, weight=400):
        return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
                f'font-weight="{weight}" fill="{fill}" xml:space="preserve">{escape(value)}</text>')

    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720" role="img" aria-labelledby="title desc">',
           f'<title id="title">{escape(scene["title"])}</title>',
           '<desc id="desc">Citazione e codice originali. Le evidenziazioni aiutano la lettura; non mostrano pesi o esecuzioni del modello.</desc>',
           '<rect width="1280" height="720" fill="#f7f8fc"/>',
           '<rect width="1280" height="10" fill="#1a237e"/>',
           text(54, 44, f"PERCEIVER · CAPITOLO {plan['chapter']}", 18, MUTED, weight=600),
           text(1070, 44, f"{index + 1} / {len(plan['scenes'])}", 20, ORANGE, weight=700)]
    for row, title in enumerate(textwrap.wrap(scene["title"], width=67, break_long_words=False)):
        if row >= 2:
            break
        svg.append(text(54, 91 + row * 34, title, 29, weight=700))
    svg.append('<rect x="42" y="150" width="1196" height="180" rx="12" fill="#fff" stroke="#dce0ee" stroke-width="2"/>')
    for row, quote in enumerate(textwrap.wrap(scene["text"], width=82, break_long_words=False, break_on_hyphens=False)):
        svg.append(text(65, 187 + row * 33, quote, 25, weight=600))
    page = scene.get("codePage")
    svg.append('<rect x="42" y="350" width="1196" height="268" rx="12" fill="#e8eaf6" stroke="#d7dcee" stroke-width="2"/>')
    if page:
        svg.append(text(65, 379, page["path"], 17, weight=600))
        for row, code in enumerate(page["rows"]):
            y = 411 + row * 25
            label = "" if code["continuation"] else str(code["line"])
            svg += [text(65, y, label, 19, ORANGE, CODE_FONT, 600), text(122, y, code["text"], 19, family=CODE_FONT)]
    else:
        label = "Rispondi a voce o metti in pausa." if scene["kind"] == "question" else "Segui questo passaggio nella tua lezione."
        svg.append(text(65, 398, label, 26, weight=600))
        svg.append(text(65, 445, "Le domande e il tutor restano interattivi nella piattaforma.", 22, MUTED))
        original = scene.get("originalAnimation", {})
        if original.get("kind") == "original":
            target = original.get("targetId") or original.get("labId") or "capitolo"
            svg.append(text(65, 490, f"Animazione originale nella lezione: {target}", 20, MUTED))
    source = scene["source"]
    svg.append(text(54, 658, f"Testo originale · {source['path']}:{source['line']}", 17, MUTED))
    svg.append(text(54, 688, "Evidenziazione per la lettura · nessuna esecuzione simulata del modello", 16, MUTED))
    svg.append('<rect x="0" y="710" width="1280" height="10" fill="#dce0ee"/></svg>')
    return "".join(svg)


def ffmpeg_command(ffmpeg: str, audio: Path, subtitle: Path, video: Path, fps: int) -> list[str]:
    return [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{WIDTH}x{HEIGHT}", "-r", str(fps), "-i", "pipe:0", "-i", str(audio),
            "-i", str(subtitle), "-map", "0:v", "-map", "1:a", "-map", "2:0",
            "-c:v", "libx264", "-preset", "fast", "-crf", "24", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "96k", "-c:s", "mov_text", "-metadata:s:s:0", "language=ita",
            "-movflags", "+faststart", str(video)]


def runtime_directory(explicit: Path | None) -> Path:
    if explicit is None:
        config = Path(os.environ.get("LOCALAPPDATA", "")) / "PerceiverTutor/voice-runtime.json"
        if not config.is_file():
            raise ValueError("Indica --runtime oppure configura il runtime Kokoro locale del tutor.")
        explicit = Path(json.loads(config.read_text(encoding="utf-8-sig"))["root"])
    resolved = output_directory(explicit)
    for filename in ("kokoro-v1.0.onnx", "voices-v1.0.bin"):
        if not (resolved / filename).is_file():
            raise ValueError(f"File del runtime locale mancante: {filename}")
    return resolved


def render_plan(plan: dict[str, Any], output: Path, runtime: Path, fps: int = 20,
                overwrite: bool = False) -> dict[str, Any]:
    """Explicit future export. No caller invokes this in default/--plan mode."""
    output = output_directory(output)
    runtime = runtime_directory(runtime)
    if fps not in {15, 20, 25, 30}:
        raise ValueError("FPS consentiti: 15, 20, 25, 30.")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("FFmpeg non è disponibile nel PATH.")
    stem = f"cap-{plan['chapter']:02}"
    names = {extension: output / f"{stem}.{extension}" for extension in ("mp4", "vtt", "srt")}
    review = output / f"{stem}-piano.json"
    if not overwrite and any(path.exists() for path in [*names.values(), review]):
        raise ValueError("Un file di questo capitolo esiste già; scegli un'altra cartella o usa --overwrite.")
    # Same local neural voice and SVG rasterisation used by generate_lesson.py.
    import io
    import numpy as np
    from PIL import Image, ImageDraw
    import resvg_py
    import soundfile as sf
    from neural_tts import synthesize

    fonts = [Path("C:/Windows/Fonts") / name for name in ("segoeui.ttf", "segoeuib.ttf", "consola.ttf")]
    if not all(path.is_file() for path in fonts):
        raise ValueError("Sono richiesti i font originali Segoe UI e Consolas.")
    output.mkdir(parents=True, exist_ok=True)
    rendered = json.loads(json.dumps(plan, ensure_ascii=False))
    with tempfile.TemporaryDirectory(prefix=f"perceiver-{stem}-", dir=runtime) as temporary_name:
        temporary = Path(temporary_name)
        samples_list, position, rate = [], 0., 24000
        frames = []
        for index, scene in enumerate(rendered["scenes"]):
            scene["start"] = round(position, 3)
            if scene["speechText"]:
                wav = temporary / f"segment-{index:04}.wav"
                synthesize(scene["speechText"], wav, rendered["voice"], rendered["speechSpeed"], runtime)
                samples, rate = sf.read(str(wav), dtype="float32")
                samples_list.append(samples)
                position += len(samples) / rate
            pause = scene.get("readingSeconds", 0) + scene["pauseAfterSeconds"]
            samples_list.append(np.zeros(round(rate * pause), dtype=np.float32))
            position += pause
            scene["end"] = round(position, 3)
            png = resvg_py.svg_to_bytes(svg_string=scene_svg(rendered, scene, index),
                                       font_files=[str(path) for path in fonts], skip_system_fonts=True)
            frames.append(Image.open(io.BytesIO(png)).convert("RGB"))
        audio = temporary / "master.wav"
        sf.write(str(audio), np.concatenate(samples_list), rate)
        rendered.update(mode="render", mediaGenerated=True, timingsVerified=True,
                        duration=round(position, 3), voiceEngine="Kokoro 82M ONNX · voce italiana locale")
        rendered["format"]["fps"] = fps
        vtt, srt = subtitle_text(rendered["scenes"])
        temporary_vtt = temporary / f"{stem}.vtt"
        temporary_srt = temporary / f"{stem}.srt"
        temporary_vtt.write_text(vtt, encoding="utf-8")
        temporary_srt.write_text(srt, encoding="utf-8")
        temporary_video = temporary / f"{stem}.mp4"
        process = subprocess.Popen(ffmpeg_command(ffmpeg, audio, temporary_srt, temporary_video, fps), stdin=subprocess.PIPE)
        try:
            scene_index = 0
            for frame_number in range(math.ceil(position * fps)):
                time = frame_number / fps
                while scene_index < len(rendered["scenes"]) - 1 and time >= rendered["scenes"][scene_index]["end"]:
                    scene_index += 1
                scene = rendered["scenes"][scene_index]
                frame = frames[scene_index].copy()
                draw = ImageDraw.Draw(frame)
                local = max(0, time - scene["start"])
                # This marker follows the reading order; it is not a model activation.
                rows = scene.get("codePage", {}).get("rows", []) if scene.get("codePage") else []
                y = 403 + 25 * (int(local / 1.4) % len(rows)) if rows else 183 + 33 * (int(local / 2.4) % max(1, len(textwrap.wrap(scene["text"], 82))))
                draw.ellipse((47, y - 5, 57, y + 5), fill=ORANGE)
                draw.rectangle((0, 710, WIDTH * min(1, time / position), HEIGHT), fill=INDIGO)
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("FFmpeg non ha completato l'esportazione del capitolo.")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        shutil.copyfile(temporary_video, names["mp4"])
        shutil.copyfile(temporary_vtt, names["vtt"])
        shutil.copyfile(temporary_srt, names["srt"])
        review.write_text(json.dumps(rendered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"chapter": plan["chapter"], "video": str(names["mp4"]), "duration": rendered["duration"],
            "bytes": names["mp4"].stat().st_size, "plan": str(review),
            "subtitles": [str(names["vtt"]), str(names["srt"])]}


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chapter", type=int, required=True, help="Numero del capitolo (1–52).")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Solo piano JSON, nessun audio/video (predefinito).")
    mode.add_argument("--render", action="store_true", help="Genera esplicitamente MP4 e sottotitoli usando il PC.")
    parser.add_argument("--output", type=Path, help="Cartella esterna al repository; senza flag il piano va su stdout.")
    parser.add_argument("--runtime", type=Path, help="Runtime Kokoro locale, necessario solo per --render.")
    parser.add_argument("--voice", choices=["im_nicola", "if_sara"], default="im_nicola")
    parser.add_argument("--speed", type=float, default=.86)
    parser.add_argument("--fps", type=int, choices=[15, 20, 25, 30], default=20)
    parser.add_argument("--overwrite", action="store_true", help="Sostituisce soltanto i file del capitolo nella cartella scelta.")
    args = parser.parse_args(argv)
    try:
        if args.render and args.output is None:
            raise ValueError("--render richiede --output con una cartella esterna al repository.")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
        plan = build_plan(manifest, args.chapter, voice=args.voice, speed=args.speed)
        if args.render:
            result = render_plan(plan, args.output, runtime_directory(args.runtime), args.fps, args.overwrite)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.output:
            output = output_directory(args.output)
            path = output / f"cap-{args.chapter:02}-piano.json"
            if path.exists() and not args.overwrite:
                raise ValueError("Il piano esiste già; scegli un'altra cartella o usa --overwrite.")
            output.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"plan": str(path), "mediaGenerated": False}, ensure_ascii=False))
        else:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"Errore: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
