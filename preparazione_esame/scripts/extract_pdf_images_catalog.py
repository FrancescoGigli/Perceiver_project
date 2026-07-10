from __future__ import annotations

import csv
import hashlib
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader


PDF_DIR = Path(r"C:\Users\gigli\Desktop\PDF_Bibliografia_Perceiver")
OUT_DIR = Path(r"C:\Users\gigli\Desktop\PDF_Bibliografia_Perceiver_immagini_estratte")
PDFIMAGES = Path(r"C:\Users\gigli\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdfimages.exe")


@dataclass
class ImageInfo:
    page: int
    num: int
    kind: str
    width: int
    height: int
    color: str
    comp: str
    bpc: str
    enc: str
    object_id: str
    x_ppi: str
    y_ppi: str
    size: str
    ratio: str


def clean_text(value: str, max_len: int = 900) -> str:
    value = re.sub(r"\s+", " ", value or "").strip()
    return value[:max_len].rstrip()


def safe_name(value: str, max_len: int = 70) -> str:
    value = re.sub(r"[\\/:*?\"<>|]+", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value[:max_len].strip() or "pdf"


def parse_pdfimages_list(output: str) -> list[ImageInfo]:
    rows: list[ImageInfo] = []
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 15 or not parts[0].isdigit():
            continue
        try:
            rows.append(
                ImageInfo(
                    page=int(parts[0]),
                    num=int(parts[1]),
                    kind=parts[2],
                    width=int(parts[3]),
                    height=int(parts[4]),
                    color=parts[5],
                    comp=parts[6],
                    bpc=parts[7],
                    enc=parts[8],
                    object_id=f"{parts[11]} {parts[12]}",
                    x_ppi=parts[13],
                    y_ppi=parts[14],
                    size=parts[15] if len(parts) > 15 else "",
                    ratio=parts[16] if len(parts) > 16 else "",
                )
            )
        except ValueError:
            continue
    return rows


def page_captions(text: str) -> list[str]:
    lines = [clean_text(line, 500) for line in (text or "").splitlines()]
    lines = [line for line in lines if line]
    captions: list[str] = []
    caption_start = re.compile(r"^(figure|fig\.?|table|tab\.?)\s*[\dIVXivxA-Za-z][\w.\-()]*\s*[:.\-]?\s+", re.I)
    stop = re.compile(
        r"^(\d+(\.\d+)*\s+[A-Z]|[A-Z][A-Z ]{6,}$|abstract\b|introduction\b|references\b|appendix\b)",
        re.I,
    )

    i = 0
    while i < len(lines):
        line = lines[i]
        if caption_start.search(line):
            chunk = [line]
            j = i + 1
            while j < len(lines) and len(" ".join(chunk)) < 700:
                nxt = lines[j]
                if caption_start.search(nxt) or stop.search(nxt):
                    break
                chunk.append(nxt)
                if nxt.endswith(".") and len(" ".join(chunk)) > 120:
                    break
                j += 1
            captions.append(clean_text(" ".join(chunk), 800))
            i = j
        else:
            i += 1
    return captions


def fallback_description(text: str, title: str, page: int) -> str:
    text = clean_text(text, 1200)
    if not text:
        return f"Nessuna didascalia testuale rilevata; immagine estratta da pagina {page} di {title}."
    for token in ("figure", "fig.", "image", "diagram", "architecture", "plot", "table"):
        pos = text.lower().find(token)
        if pos >= 0:
            start = max(0, pos - 120)
            return "Testo pagina vicino a riferimenti visivi: " + clean_text(text[start : start + 650], 700)
    return "Nessuna didascalia esplicita rilevata. Contesto pagina: " + clean_text(text[:650], 700)


def read_page_texts(pdf: Path) -> dict[int, str]:
    texts: dict[int, str] = {}
    try:
        reader = PdfReader(str(pdf))
        for idx, page in enumerate(reader.pages, 1):
            try:
                texts[idx] = page.extract_text() or ""
            except Exception:
                texts[idx] = ""
    except Exception:
        pass
    return texts


def find_extracted_file(raw_dir: Path, prefix: str, page: int, num: int) -> Path | None:
    base = f"{prefix}-{page:03d}-{num:03d}"
    matches = sorted(raw_dir.glob(base + ".*"))
    return matches[0] if matches else None


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_pdfimages(args: Iterable[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(PDFIMAGES), *args],
        text=True,
        capture_output=True,
        errors="replace",
        check=False,
    )


def main() -> None:
    if not PDF_DIR.exists():
        raise SystemExit(f"Cartella PDF non trovata: {PDF_DIR}")
    if not PDFIMAGES.exists():
        raise SystemExit(f"pdfimages non trovato: {PDFIMAGES}")

    images_dir = OUT_DIR / "immagini"
    raw_dir = OUT_DIR / "_raw_pdfimages"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    global_idx = 0

    for pdf_idx, pdf in enumerate(sorted(PDF_DIR.glob("*.pdf")), 1):
        title = pdf.stem
        pdf_slug = safe_name(title, 62)
        list_proc = run_pdfimages(["-list", str(pdf)])
        infos = parse_pdfimages_list(list_proc.stdout)
        main_images = [info for info in infos if info.kind == "image"]
        masks = [info for info in infos if info.kind == "smask"]

        page_texts = read_page_texts(pdf)
        captions_by_page = {page: page_captions(text) for page, text in page_texts.items()}
        per_page_seen: dict[int, int] = {}

        raw_prefix = str(raw_dir / f"pdf{pdf_idx:02d}")
        extract_proc = run_pdfimages(["-png", "-p", str(pdf), raw_prefix])
        if extract_proc.returncode != 0:
            summary.append(
                {
                    "PDF": pdf.name,
                    "Immagini principali trovate": len(main_images),
                    "Maschere escluse": len(masks),
                    "Immagini estratte": 0,
                    "Note": clean_text(extract_proc.stderr, 400),
                }
            )
            continue

        extracted_count = 0
        for info in main_images:
            src = find_extracted_file(raw_dir, f"pdf{pdf_idx:02d}", info.page, info.num)
            if not src:
                continue
            global_idx += 1
            extracted_count += 1
            page_count = per_page_seen.get(info.page, 0)
            per_page_seen[info.page] = page_count + 1
            captions = captions_by_page.get(info.page, [])
            caption = captions[page_count] if page_count < len(captions) else (" | ".join(captions) if captions else "")
            description = caption or fallback_description(page_texts.get(info.page, ""), title, info.page)

            area = info.width * info.height
            fragment_note = ""
            if info.width < 80 or info.height < 80 or area < 10_000:
                fragment_note = "Possibile frammento o icona piccola"
            elif len([x for x in main_images if x.page == info.page]) >= 8:
                fragment_note = "Pagina con molte immagini: possibile figura spezzata in più oggetti"

            pdf_image_dir = images_dir / f"{pdf_idx:02d}_{pdf_slug}"
            pdf_image_dir.mkdir(parents=True, exist_ok=True)
            ext = src.suffix.lower() or ".png"
            out_name = f"img_{global_idx:04d}_p{info.page:03d}_n{info.num:03d}{ext}"
            out_path = pdf_image_dir / out_name
            shutil.copy2(src, out_path)

            rows.append(
                {
                    "ID": global_idx,
                    "PDF": pdf.name,
                    "PDF_index": pdf_idx,
                    "Pagina": info.page,
                    "Numero_immagine_pdf": info.num,
                    "File_immagine": str(out_path),
                    "Nome_file_immagine": out_name,
                    "Descrizione": description,
                    "Caption_rilevata": caption,
                    "Note": fragment_note,
                    "Larghezza_px": info.width,
                    "Altezza_px": info.height,
                    "Area_px": area,
                    "Colore": info.color,
                    "BPC": info.bpc,
                    "Encoding": info.enc,
                    "Object_ID": info.object_id,
                    "X_PPI": info.x_ppi,
                    "Y_PPI": info.y_ppi,
                    "Dimensione_pdfimages": info.size,
                    "SHA1": file_sha1(out_path),
                }
            )

        summary.append(
            {
                "PDF": pdf.name,
                "Immagini principali trovate": len(main_images),
                "Maschere escluse": len(masks),
                "Immagini estratte": extracted_count,
                "Note": "",
            }
        )

    csv_path = OUT_DIR / "tabella_immagini_pdf.csv"
    summary_path = OUT_DIR / "riepilogo_pdf.csv"
    fieldnames = list(rows[0].keys()) if rows else ["ID", "PDF", "Descrizione"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with summary_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print(f"Output: {OUT_DIR}")
    print(f"PDF processati: {len(summary)}")
    print(f"Immagini estratte: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"Riepilogo: {summary_path}")


if __name__ == "__main__":
    main()
