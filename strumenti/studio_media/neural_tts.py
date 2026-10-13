"""Local neural Italian voice. Model/dependencies are stored outside the repository.

Runtime: Kokoro ONNX (MIT) + Kokoro 82M model (Apache-2.0).
This helper does not use SAPI and never sends text to an external service.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

_pipelines = {}


def normalize_pronunciation(text: str) -> str:
    """Read mathematical symbols aloud without changing displayed source text."""
    replacements = {
        "QKᵀ": "cu per kappa trasposta",
        "N×M": "enne per emme",
        "O(M²)": "o di emme al quadrato",
        "O(MN)": "o di emme per enne",
        "≪": " molto minore di ",
        "×": " per ",
        "²": " al quadrato ",
    }
    for original, spoken in replacements.items():
        text = text.replace(original, spoken)
    for letter, spoken in {"Q": "cu", "K": "kappa", "V": "vi", "L": "elle", "N": "enne", "M": "emme", "X": "ics"}.items():
        text = re.sub(rf"\b{letter}\b", spoken, text)
    return text.replace("=", " è ")


def synthesize(text: str, output: str | Path, voice: str = "im_nicola", speed: float = .92, runtime: str | Path | None = None):
    import numpy as np
    import soundfile as sf
    from kokoro_onnx import Kokoro
    from misaki.espeak import EspeakG2P
    import onnxruntime as ort

    if not runtime:
        raise ValueError("Specify runtime directory containing kokoro-v1.0.onnx and voices-v1.0.bin")
    runtime = Path(runtime)
    key = str(runtime.resolve())
    if key not in _pipelines:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(runtime / "kokoro-v1.0.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
        _pipelines[key] = (Kokoro.from_session(session, str(runtime / "voices-v1.0.bin")), EspeakG2P(language="it"))
    model, phonemizer = _pipelines[key]
    if voice not in {"if_sara", "im_nicola"}:
        raise ValueError("Italian voice must be if_sara or im_nicola")
    chunks = []
    pending = ""
    for word in normalize_pronunciation(text).split():
        if len(pending) + len(word) + 1 > 300:
            chunks.append(pending)
            pending = ""
        pending = f"{pending} {word}".strip()
        if len(pending) > 100 and pending[-1:] in ".!?;":
            chunks.append(pending)
            pending = ""
    if pending:
        chunks.append(pending)
    if not chunks:
        raise ValueError("Text must not be empty")
    audio = []
    rate = 24000
    for chunk in chunks:
        phonemes, _ = phonemizer(chunk.strip())
        samples, rate = model.create(phonemes, voice, speed=float(speed), is_phonemes=True)
        audio.append(samples)
        audio.append(np.zeros(int(rate * .12), dtype=np.float32))
    samples = np.concatenate(audio)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), samples, rate)
    return {"path": str(output), "duration": len(samples) / rate, "sample_rate": rate, "voice": voice}


def main():
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--voice", choices=["if_sara", "im_nicola"], default="im_nicola")
    parser.add_argument("--speed", type=float, default=.92)
    args = parser.parse_args()
    print(json.dumps(synthesize(args.text_file.read_text(encoding="utf-8-sig"), args.output, args.voice, args.speed, args.runtime)))


if __name__ == "__main__":
    main()
