from __future__ import annotations

import csv
import html
import re
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image


ROOT = Path(r"N:\Perceiver_project\preparazione_esame")
APP_DIR = ROOT / "perceiver_condivisibile" / "perceiver_interattivo"
INDEX_PATH = APP_DIR / "index.html"
ASSET_DIR = APP_DIR / "paper_figures"
CATALOG_CSV = Path(r"C:\Users\gigli\Desktop\PDF_Bibliografia_Perceiver_immagini_estratte\tabella_immagini_pdf.csv")
MANIFEST_CSV = Path(r"C:\Users\gigli\Desktop\PDF_Bibliografia_Perceiver\manifest_pdf_bibliografia.csv")


CATEGORY_BY_INDEX = [
    (1, 5, "Perceiver e Transformer"),
    (6, 11, "Visione e CNN"),
    (12, 17, "Sequenze e NLP"),
    (18, 22, "Self-supervised e rappresentazioni"),
    (23, 26, "Ottimizzazione"),
    (27, 31, "Regolarizzazione e normalizzazione"),
    (32, 36, "Inizializzazione e iperparametri"),
    (37, 39, "Fondamenti e attivazioni"),
]


def category_for(index: int) -> str:
    for start, end, label in CATEGORY_BY_INDEX:
        if start <= index <= end:
            return label
    return "Altri riferimenti"


def clean(value: str, max_len: int = 420) -> str:
    value = re.sub(r"\s+", " ", value or "").strip()
    return value[:max_len].rstrip()


def norm_caption(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", " ", (value or "").lower())
    return " ".join(value.split())[:160]


def safe_attr(value: str) -> str:
    return html.escape(value, quote=True)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def resize_to_web(src: Path, dst: Path) -> tuple[int, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im.load()
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            bg = Image.new("RGB", im.size, "white")
            bg.paste(im.convert("RGBA"), mask=im.convert("RGBA").getchannel("A"))
            im = bg
        else:
            im = im.convert("RGB")
        im.thumbnail((920, 640), Image.Resampling.LANCZOS)
        im.save(dst, "JPEG", quality=84, optimize=True, progressive=True)
        return im.size


def select_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    seen_hashes: set[str] = set()
    for row in rows:
        try:
            index = int(row["PDF_index"])
            width = int(row["Larghezza_px"])
            height = int(row["Altezza_px"])
            area = int(row["Area_px"])
        except (KeyError, ValueError):
            continue
        src = Path(row.get("File_immagine", ""))
        if not src.exists() or src.stat().st_size < 8_000:
            continue
        sha1 = row.get("SHA1", "")
        if sha1 and sha1 in seen_hashes:
            continue
        if sha1:
            seen_hashes.add(sha1)
        note = row.get("Note", "")
        row["_index"] = index
        row["_width"] = width
        row["_height"] = height
        row["_area"] = area
        if width >= 180 and height >= 120 and area >= 45_000 and "frammento" not in note.lower():
            grouped[index].append(row)

    selected: list[dict[str, str]] = []
    for index in sorted(grouped):
        by_caption: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in grouped[index]:
            key = norm_caption(row.get("Caption_rilevata") or row.get("Descrizione") or f"p{row.get('Pagina')}")
            by_caption[key].append(row)

        representatives = []
        for caption_rows in by_caption.values():
            caption_rows.sort(key=lambda r: (-int(r["_area"]), int(r["Pagina"]), int(r["Numero_immagine_pdf"])))
            representatives.append(caption_rows[0])

        representatives.sort(key=lambda r: (int(r["Pagina"]), -int(r["_area"]), int(r["Numero_immagine_pdf"])))
        chosen = representatives[:3]

        if len(chosen) < 2:
            fill = sorted(grouped[index], key=lambda r: (-int(r["_area"]), int(r["Pagina"])))
            for row in fill:
                if row not in chosen:
                    chosen.append(row)
                if len(chosen) >= 2:
                    break

        selected.extend(chosen)

    return selected


def build_html(cards: list[dict[str, str]], manifest: dict[int, dict[str, str]]) -> str:
    sections: dict[str, list[str]] = defaultdict(list)
    for row in cards:
        index = int(row["PDF_index"])
        meta = manifest.get(index, {})
        title = clean(meta.get("Title") or row["PDF"], 120)
        ref = clean(meta.get("Ref") or f"PDF {index}", 60)
        authors = clean(meta.get("Authors") or "", 90)
        href = meta.get("OriginalUrl") or meta.get("PdfUrl") or "#"
        page = row["Pagina"]
        caption = clean(row.get("Caption_rilevata") or row.get("Descrizione") or "Figura estratta dal paper.")
        img_src = row["_web_src"]
        width = row["_web_width"]
        height = row["_web_height"]
        category = category_for(index)
        alt = clean(f"{title}, pagina {page}: {caption}", 180)

        sections[category].append(
            f'''          <figure class="paper-figure-card">
            <a class="paper-figure-media" href="{safe_attr(href)}" target="_blank" rel="noopener" aria-label="Apri il paper: {safe_attr(title)}">
              <img src="{safe_attr(img_src)}" alt="{safe_attr(alt)}" loading="lazy" decoding="async" width="{width}" height="{height}">
            </a>
            <figcaption>
              <span class="paper-figure-source">{safe_attr(ref)} · p. {safe_attr(page)}{(" · " + safe_attr(authors)) if authors else ""}</span>
              <strong>{safe_attr(title)}</strong>
              <span class="paper-figure-caption">{safe_attr(caption)}</span>
            </figcaption>
          </figure>'''
        )

    parts = [
        '      <!-- ============ Atlante visivo dai paper ============ -->',
        '      <div class="paper-figure-atlas" id="atlante-figure-paper">',
        '        <div class="paper-figure-atlas-head">',
        '          <span class="book-sigla">Didascalie estratte</span>',
        '          <h2>Atlante visivo dai paper</h2>',
        '          <p>Una selezione leggera delle figure raster estratte dai PDF, scelta per ripasso visivo: poche immagini rappresentative per paper, con pagina e caption originale quando disponibile.</p>',
        '        </div>',
    ]
    for _, _, category in CATEGORY_BY_INDEX:
        if category not in sections:
            continue
        parts.extend(
            [
                '        <section class="paper-figure-group">',
                f'          <h3>{safe_attr(category)}</h3>',
                '          <div class="paper-figure-grid">',
                *sections[category],
                '          </div>',
                '        </section>',
            ]
        )
    parts.append("      </div>")
    return "\n".join(parts)


def main() -> None:
    rows = load_csv(CATALOG_CSV)
    manifest = {int(row["Index"]): row for row in load_csv(MANIFEST_CSV)}
    selected = select_rows(rows)

    if ASSET_DIR.exists():
        shutil.rmtree(ASSET_DIR)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    for counter, row in enumerate(selected, 1):
        src = Path(row["File_immagine"])
        out_name = f"fig_{counter:03d}_pdf{int(row['PDF_index']):02d}_p{int(row['Pagina']):03d}.jpg"
        dst = ASSET_DIR / out_name
        width, height = resize_to_web(src, dst)
        row["_web_src"] = f"paper_figures/{out_name}"
        row["_web_width"] = str(width)
        row["_web_height"] = str(height)

    atlas_html = build_html(selected, manifest)
    html_text = INDEX_PATH.read_text(encoding="utf-8-sig")
    pattern = re.compile(
        r"\n\s*<!-- ============ Atlante visivo dai paper ============ -->[\s\S]*?\n\s*</div>\s*\n\s*<div class=\"idea\">",
        re.M,
    )
    replacement = "\n" + atlas_html + "\n\n      <div class=\"idea\">"
    if pattern.search(html_text):
        html_text = pattern.sub(replacement, html_text)
    else:
        marker = '      <div class="idea">\n        <div class="idea-label">Come usarla</div>'
        if marker not in html_text:
            raise SystemExit("Marker di inserimento non trovato.")
        html_text = html_text.replace(marker, atlas_html + "\n\n" + marker, 1)
    INDEX_PATH.write_text(html_text, encoding="utf-8")

    print(f"Selected figures: {len(selected)}")
    print(f"Assets: {ASSET_DIR}")
    print(f"Updated: {INDEX_PATH}")


if __name__ == "__main__":
    main()
