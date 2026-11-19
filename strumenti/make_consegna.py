# strumenti/make_consegna.py
# Costruisce la cartella di consegna: SOLO il codice, pronto per la repo privata.
#
#   python strumenti/make_consegna.py                 # cartella + zip
#   python strumenti/make_consegna.py --keep-comments # senza spogliare i commenti
#
# Cosa fa, in ordine:
#   1. copia da progetto/ i soli file TRACCIATI da git
#      (esclude per costruzione data/ ~19 GB, logs/, __pycache__, output rigenerabili);
#   2. rimuove commenti e docstring dai .py (default), lasciando le stringhe di
#      argparse: il --help continua a funzionare;
#   3. verifica che tutto compili e crea lo zip.
#
# NON fa parte del codice da consegnare: e' uno strumento di build.

import argparse
import ast
import errno
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tokenize
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # strumenti/ -> radice del repo
SOURCE = ROOT / "progetto"


def tracked_files():
    """File tracciati da git dentro progetto/. E' il filtro piu' affidabile:
    quello che git ignora (dati, log, cache) non finisce nella consegna."""
    out = subprocess.check_output(["git", "ls-files", "progetto/"],
                                  cwd=str(ROOT), text=True)
    return [line.strip() for line in out.splitlines() if line.strip()]


def check_nothing_is_left_behind():
    """File .py presenti ma non tracciati da git verrebbero esclusi in SILENZIO:
    la consegna partirebbe senza uno script che il registro invoca. Meglio fermarsi."""
    out = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "progetto/"],
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


def copy_files(files, dest, keep_comments):
    n_py = 0
    for rel in files:
        src = ROOT / rel
        target = dest / Path(rel).relative_to("progetto")
        target.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix == ".py" and not keep_comments:
            target.write_text(strip_comments(src.read_text(encoding="utf-8"), src),
                              encoding="utf-8")
            n_py += 1
        else:
            shutil.copy2(src, target)
        if src.suffix == ".py":
            compile(target.read_bytes(), str(target), "exec")
    subprocess.check_call([sys.executable, "-B", "-c",
                           "import experiments; print('  registro:', len(experiments.EXPERIMENTS), 'run')"],
                          cwd=str(dest))
    return n_py


def publish_new(source, target):
    """Rinomina sullo stesso filesystem solo se il nome finale è ancora libero."""
    if os.name == "nt":
        # Su Windows rename fallisce anche se il target è una directory vuota.
        os.rename(source, target)
        return

    # Su POSIX os.rename può sostituire una directory vuota comparsa in corsa.
    # I flag esclusivi del kernel applicano controllo e pubblicazione insieme.
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        at_fdcwd = -100
        result = libc.renameat2(at_fdcwd, os.fsencode(source), at_fdcwd, os.fsencode(target), 1)
    elif sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        result = libc.renamex_np(os.fsencode(source), os.fsencode(target), 4)
    else:
        raise OSError(errno.ENOTSUP, "Rinomina esclusiva non disponibile su questa piattaforma")
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(target))


def build(dest, keep_comments=False, no_zip=False):
    # Risolvi i link dei genitori ma conserva l'ultima componente per rilevare
    # anche un link di destinazione rotto.
    dest = Path(dest).absolute()
    dest = dest.parent.resolve() / dest.name
    source = SOURCE.resolve()
    git_dir = (ROOT / ".git").resolve()
    if source.is_relative_to(dest) or dest.is_relative_to(source) or dest.is_relative_to(git_dir):
        raise ValueError(f"Destinazione incompatibile con i sorgenti del progetto: {dest}")
    archive = None if no_zip else dest.with_suffix(".zip")
    if archive == dest:
        raise ValueError(f"Cartella e ZIP avrebbero lo stesso nome: {dest}")
    for target in (dest, archive):
        if target is not None and os.path.lexists(target):
            raise FileExistsError(errno.EEXIST, "Scegli un nome di consegna libero", str(target))

    check_nothing_is_left_behind()
    files = tracked_files()
    if not files:
        raise ValueError("git ls-files non ha trovato nulla sotto progetto/")

    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".consegna-", dir=dest.parent) as work:
        staged = Path(work) / "codice"
        staged.mkdir()
        n_py = copy_files(files, staged, keep_comments)
        if archive is None:
            publish_new(staged, dest)
        else:
            staged_zip = Path(work) / "consegna.zip"
            with zipfile.ZipFile(staged_zip, "x", zipfile.ZIP_DEFLATED) as zf:
                for f in sorted(staged.rglob("*")):
                    if f.is_file():
                        zf.write(f, f.relative_to(staged))
            # Pubblicando prima lo ZIP, un conflitto sulla cartella richiede di
            # ritirare solo il nostro file, mai di eliminare un albero pubblico.
            written = staged_zip.stat()
            publish_new(staged_zip, archive)
            try:
                publish_new(staged, dest)
            except BaseException:
                try:
                    if os.path.samestat(written, archive.lstat()):
                        archive.unlink()
                except FileNotFoundError:
                    pass
                raise
    print(f"Pronto: {len(files)} file in {dest}")
    if not keep_comments:
        print(f"  {n_py} file .py spogliati di commenti e docstring")
    if archive is not None:
        print(f"Zip: {archive}")


def main():
    parser = argparse.ArgumentParser(description="Crea una nuova cartella di consegna")
    parser.add_argument("--dest", default=str(ROOT.parent / "Perceiver_consegna"),
                        help="nome libero per la cartella e il relativo .zip")
    parser.add_argument("--keep-comments", action="store_true",
                        help="non rimuovere commenti e docstring")
    parser.add_argument("--no-zip", action="store_true", help="crea solo la cartella")
    args = parser.parse_args()
    try:
        build(args.dest, args.keep_comments, args.no_zip)
    except (OSError, ValueError, SyntaxError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Consegna interrotta: {exc}\n")


if __name__ == "__main__":
    main()
