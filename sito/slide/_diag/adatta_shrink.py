# Alza lo shrink dei frame che sforano, finche' non sfora piu' nessuno.
#
#   python _diag/adatta_shrink.py         # dalla cartella sito/slide/
#
# Serve perche' togliere i riquadri dei block ha fatto crescere i frame: senza
# la cornice il testo occupa piu' righe. Beamer non lo segnala (nessun Overfull
# vbox), quindi il controllo e' sul PDF renderizzato: se resta inchiostro nella
# fascia appena sopra la barra del footer, quella pagina sfora.
#
# I frame corrispondono alle pagine 1:1 (nessun overlay, nessuna slide di
# sezione), quindi la pagina N si mappa sull'N-esimo \begin{frame}.

import glob
import io
import os
import re
import subprocess

QUI = os.path.dirname(os.path.abspath(__file__))
SLIDE = os.path.abspath(os.path.join(QUI, ".."))
TEX = "perceiver_presentation_eng.tex"
PDF = "perceiver_presentation_eng.pdf"
PDFLATEX = r"C:\Users\gigli\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"
RENDER = os.path.join(os.environ.get("TEMP", "."), "tex_shrink")

PASSO = 6          # di quanto alzare lo shrink a ogni giro
TETTO = 40         # oltre, il testo diventa illeggibile: meglio saperlo


def compila():
    for _ in range(2):
        r = subprocess.run([PDFLATEX, "-interaction=nonstopmode", "-halt-on-error", TEX],
                           cwd=SLIDE, capture_output=True, text=True)
    if not os.path.exists(os.path.join(SLIDE, PDF)):
        raise SystemExit("compilazione fallita:\n" + r.stdout[-2000:])


def pagine_che_sforano():
    if os.path.isdir(RENDER):
        for f in glob.glob(os.path.join(RENDER, "*.png")):
            os.remove(f)
    else:
        os.makedirs(RENDER)
    subprocess.run(["pdftocairo", "-png", "-r", "90", PDF, os.path.join(RENDER, "p")],
                   cwd=SLIDE, check=True)
    from PIL import Image
    fuori = []
    for f in glob.glob(os.path.join(RENDER, "p-*.png")):
        n = int(re.search(r"-(\d+)\.png", f).group(1))
        im = Image.open(f).convert("L")
        w, h = im.size
        striscia = im.crop((0, int(h * 0.925), w, int(h * 0.955)))
        if sum(1 for p in striscia.get_flattened_data() if p < 120) > 20:
            fuori.append(n)
    return sorted(fuori)


def alza(pagine):
    """Alza lo shrink dei frame corrispondenti. Ritorna quanti ne ha toccati."""
    percorso = os.path.join(SLIDE, TEX)
    testo = io.open(percorso, encoding="utf-8").read()
    inizi = [m.start() for m in re.finditer(r"\\begin\{frame\}", testo)]
    toccati = 0
    for n in sorted(pagine, reverse=True):     # dal fondo, per non spostare gli offset
        if n > len(inizi):
            continue
        i = inizi[n - 1]
        fine = testo.index("{", i + len("\\begin{frame}") - 1) if False else None
        testa = testo[i:i + 60]
        m = re.match(r"\\begin\{frame\}\[shrink=(\d+)\]", testa)
        if m:
            nuovo = min(int(m.group(1)) + PASSO, TETTO)
            testo = testo[:i] + f"\\begin{{frame}}[shrink={nuovo}]" + testo[i + m.end():]
        elif testa.startswith("\\begin{frame}{") or testa.startswith("\\begin{frame}\n"):
            testo = (testo[:i] + f"\\begin{{frame}}[shrink={PASSO}]"
                     + testo[i + len("\\begin{frame}"):])
        else:
            continue
        toccati += 1
    io.open(percorso, "w", encoding="utf-8").write(testo)
    return toccati


def main():
    for giro in range(1, 6):
        compila()
        fuori = pagine_che_sforano()
        if not fuori:
            print(f"giro {giro}: nessuna pagina sfora piu'")
            return
        print(f"giro {giro}: sforano {fuori}")
        if alza(fuori) == 0:
            print("nessun frame modificabile, mi fermo")
            return
    compila()
    print("residuo dopo 5 giri:", pagine_che_sforano())


if __name__ == "__main__":
    main()
