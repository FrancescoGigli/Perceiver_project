# tools/make_consegna.py
# Costruisce la cartella di consegna: SOLO il codice, pronto per la repo privata.
#
#   python tools/make_consegna.py                 # cartella + zip
#   python tools/make_consegna.py --keep-comments # senza spogliare i commenti
#
# Cosa fa, in ordine:
#   1. copia da perceiver_project/ i soli file TRACCIATI da git
#      (esclude per costruzione data/ ~19 GB, logs/, __pycache__, output rigenerabili);
#   2. rimuove commenti e docstring dai .py (default), lasciando le stringhe di
#      argparse: il --help continua a funzionare;
#   3. verifica che tutto compili e crea lo zip.
#
# NON fa parte del codice da consegnare: e' uno strumento di build.

import argparse
import ast
import io
import re
import shutil
import subprocess
import sys
import tokenize
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # tools/ -> radice del repo
SOURCE = ROOT / "perceiver_project"


def tracked_files():
    """File tracciati da git dentro perceiver_project/. E' il filtro piu' affidabile:
    quello che git ignora (dati, log, cache) non finisce nella consegna."""
    out = subprocess.check_output(["git", "ls-files", "perceiver_project/"],
                                  cwd=str(ROOT), text=True)
    return [line.strip() for line in out.splitlines() if line.strip()]


def check_nothing_is_left_behind():
    """File .py presenti ma non tracciati da git verrebbero esclusi in SILENZIO:
    la consegna partirebbe senza uno script che il registro invoca. Meglio fermarsi."""
    out = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "perceiver_project/"],
        cwd=str(ROOT), text=True)
    untracked = [line.strip() for line in out.splitlines()
                 if line.strip().endswith(".py")]
    if untracked:
        raise SystemExit(
            "Questi file .py non sono tracciati da git e resterebbero fuori dalla "
            "consegna:\n  " + "\n  ".join(untracked) +
            "\n\nAggiungili (git add <file>) oppure ignorali esplicitamente, poi rilancia."
        )


def strip_comments(source, path):
    """Toglie commenti e docstring preservando la formattazione (tokenize).
    Se cosi' resterebbe un blocco vuoto, ripiega su AST inserendo `pass`."""
    def _via_tokenize():
        out, prev, last_line, last_col = "", tokenize.INDENT, -1, 0
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            ttype, tstr, (sline, scol), (eline, ecol), _ = tok
            if sline > last_line:
                last_col = 0
            if scol > last_col:
                out += " " * (scol - last_col)
            is_docstring = ttype == tokenize.STRING and prev in (tokenize.INDENT, tokenize.NEWLINE)
            if ttype != tokenize.COMMENT and not is_docstring:
                out += tstr
            prev, last_col, last_line = ttype, ecol, eline
        return out

    def _via_ast():
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                body = node.body
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(getattr(body[0], "value", None), ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    body.pop(0)
                    if not body:
                        body.append(ast.Pass())
        return ast.unparse(tree)

    try:
        cleaned = _via_tokenize()
        compile(cleaned, str(path), "exec")
    except SyntaxError:
        cleaned = _via_ast()
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip("\n") + "\n"
    compile(cleaned, str(path), "exec")   # deve compilare comunque
    return cleaned


def main():
    parser = argparse.ArgumentParser(description="Crea la cartella di consegna")
    parser.add_argument("--dest", default=str(ROOT.parent / "Perceiver_consegna"))
    parser.add_argument("--keep-comments", action="store_true",
                        help="non rimuovere commenti e docstring")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()

    dest = Path(args.dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    check_nothing_is_left_behind()
    files = tracked_files()
    if not files:
        raise SystemExit("git ls-files non ha trovato nulla sotto perceiver_project/")

    n_py = 0
    for rel in files:
        src = ROOT / rel
        target = dest / Path(rel).relative_to("perceiver_project")
        target.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix == ".py" and not args.keep_comments:
            target.write_text(strip_comments(src.read_text(encoding="utf-8"), src),
                              encoding="utf-8")
            n_py += 1
        else:
            shutil.copy2(src, target)

    print(f"Copiati {len(files)} file tracciati in {dest}")
    if not args.keep_comments:
        print(f"  {n_py} file .py spogliati di commenti e docstring")

    # Verifica: tutto compila e il registro e' importabile dalla cartella pulita.
    subprocess.check_call([sys.executable, "-m", "compileall", "-q", str(dest)])
    subprocess.check_call([sys.executable, "-c",
                           "import experiments; print('  registro:', len(experiments.EXPERIMENTS), 'run')"],
                          cwd=str(dest))
    for cache in dest.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)

    if not args.no_zip:
        zip_path = dest.with_suffix(".zip")
        zip_path.unlink(missing_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(dest.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(dest))
        print(f"Zip: {zip_path} ({zip_path.stat().st_size / 1024:.0f} KB)")

    total = sum(1 for f in dest.rglob("*") if f.is_file())
    print(f"Pronto: {total} file in {dest}")


if __name__ == "__main__":
    main()
