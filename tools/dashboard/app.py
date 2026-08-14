# tools/dashboard/app.py
# Pannello desktop per lanciare/monitorare gli esperimenti Perceiver / Perceiver IO.
# Tkinter (standard library) + psutil. Nessun server, nessun browser:
#
#     python tools/dashboard/app.py
#
# NON fa parte del codice da consegnare: e' solo uno strumento di controllo locale.
# Lancia train.py esattamente come farebbe `experiments.py --run <id>`.
#
# Reattivita': le letture costose (nvidia-smi, CPU) girano su un THREAD di
# background e la UI legge solo una stringa gia' pronta. Lo stato del training
# si controlla con Popen.poll() (istantaneo), niente scansione dei processi.

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

import psutil

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "perceiver_project"
LOGS = PROJECT / "logs"
VIZ = PROJECT / "perceiver_visualizations_v2"

sys.path.insert(0, str(PROJECT))
import experiments as exp  # noqa: E402  -> EXPERIMENTS, command_for

# Oltre questa eta', una cartella con residui ma senza risultato non e' piu'
# "interrotta" (qualcosa che hai fermato poco fa) ma un vecchio residuo = "da fare".
INTERROTTA_MAX_ETA_SEC = 30 * 60

GROUP_PAPER = {
    "tab1": "Tab.1 baseline", "tab2": "Tab.2 permut./PE", "tab5": "Tab.5 no latent-T",
    "tab6": "Tab.6 n. cross-attend", "tab7": "Tab.7 weight sharing",
    "fig6": "Fig.6 bande/freq/init", "noise": "banda di rumore (seed)",
    "modelnet": "Tab.4 ModelNet40",
    # Perceiver IO
    "io_image": "IO — CIFAR (decoder a query)",
    "io_mlm": "IO — MLM byte-level",
    "io_glue": "IO — Tab.1 GLUE",
}


def run_status(experiment):
    d = LOGS / experiment["id"]
    path = d / "results.json"
    if not path.exists():
        # cartella con residui ma senza risultato: "interrotta" solo se recente
        # (qualcosa fermato poco fa); un vecchio residuo torna "da fare".
        # "in corso" lo assegna la UI solo alla run che gira davvero adesso.
        if not d.is_dir():
            return ("da fare", None)
        try:
            mtime = max(f.stat().st_mtime for f in d.rglob("*") if f.is_file())
        except (OSError, ValueError):
            return ("da fare", None)
        recente = (time.time() - mtime) < INTERROTTA_MAX_ETA_SEC
        return ("interrotta", None) if recente else ("da fare", None)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return ("illeggibile", None)
    acc = data.get("test_accuracy")
    if acc is None:
        acc = data.get("val_accuracy")
    if acc is None or acc != acc:
        return ("DIVERGITA", None)
    return ("ok", acc)


def _gpu_str():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return "GPU: n/d"
    try:
        out = subprocess.check_output(
            [exe, "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=3)
        u, used, tot, temp = (x.strip() for x in out.strip().splitlines()[0].split(","))
        return f"GPU {float(u):.0f}%   VRAM {float(used)/1024:.1f}/{float(tot)/1024:.1f} GB   {float(temp):.0f}°C"
    except (subprocess.SubprocessError, OSError, ValueError, IndexError):
        return "GPU: errore"


class App:
    def __init__(self, root):
        self.root = root
        self.proc = None          # Popen del training in corso (o None)
        self.queue = []           # esperimenti in coda per la modalita' sequenziale
        self.run_id = None
        self.done = 0
        self.sysline = "lettura…"  # aggiornata dal thread, letta dalla UI
        self._log_size = -1
        self._logf = None
        self._running_id = None    # esperimento rilevato in esecuzione (anche esterno), dal thread
        self._watch_last = None    # ultimo log seguito, per redraw su cambio run
        root.title("Perceiver — pannello esperimenti")
        root.geometry("1180x720")
        root.protocol("WM_DELETE_WINDOW", self._close)

        top = ttk.Frame(root, padding=8)
        top.pack(fill="x")
        self.sys_lbl = ttk.Label(top, text="…", font=("Consolas", 10))
        self.sys_lbl.pack(side="left")
        self.run_lbl = ttk.Label(top, text="", font=("Consolas", 10, "bold"))
        self.run_lbl.pack(side="right")

        cols = ("id", "paper", "config", "mod", "stato", "acc")
        widths = (170, 150, 470, 90, 110, 70)
        mid = ttk.Frame(root, padding=(8, 0))
        mid.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(mid, columns=cols, show="headings", height=16)
        for c, w in zip(cols, widths):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="w")
        for e in exp.EXPERIMENTS:
            self.tree.insert("", "end", iid=e["id"], values=(
                e["id"], GROUP_PAPER.get(e["group"], e["group"]),
                " ".join(e["overrides"]) or "baseline",
                e.get("modality", "image"), "…", "—"))
        self.tree.tag_configure("ok", foreground="#127c12")
        self.tree.tag_configure("todo", foreground="#777")
        self.tree.tag_configure("warn", foreground="#b06000")
        self.tree.tag_configure("run", foreground="#1560d0")
        self.tree.tag_configure("bad", foreground="#c02020")
        sb = ttk.Scrollbar(mid, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        bar = ttk.Frame(root, padding=8)
        bar.pack(fill="x")
        ttk.Button(bar, text="▶ Lancia selezionato", command=self.launch).pack(side="left")
        ttk.Button(bar, text="▶▶ Mancanti in sequenza", command=self.run_missing).pack(side="left", padx=4)
        ttk.Button(bar, text="■ Ferma", command=self.stop).pack(side="left", padx=4)
        ttk.Button(bar, text="↻ Aggiorna", command=self.refresh).pack(side="left", padx=4)
        ttk.Button(bar, text="🖼 Genera mappe", command=self.gen_maps).pack(side="left", padx=4)
        ttk.Button(bar, text="📂 Mappe", command=lambda: self.open_dir(VIZ)).pack(side="left", padx=4)
        ttk.Button(bar, text="📁 Logs", command=lambda: self.open_dir(LOGS)).pack(side="left", padx=4)

        self.cmd_lbl = ttk.Label(root, text="", font=("Consolas", 9), foreground="#555", padding=(8, 0))
        self.cmd_lbl.pack(fill="x")
        self.log = tk.Text(root, height=12, bg="#0e0e0e", fg="#d0d0d0", font=("Consolas", 9), wrap="none")
        self.log.pack(fill="both", expand=False, padx=8, pady=(4, 8))

        self.tree.bind("<<TreeviewSelect>>", lambda ev: self.show_cmd())
        self.refresh()
        threading.Thread(target=self._stats_loop, daemon=True).start()  # CPU/GPU fuori dalla UI
        self.tick()

    # -- thread di background: solo calcoli, nessun widget --
    def _stats_loop(self):
        while getattr(self, "_alive", True):
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            self.sysline = f"CPU {cpu:4.0f}%   RAM {ram:3.0f}%   {_gpu_str()}"
            self._running_id = self._detect_running()   # scan processi fuori dal thread UI
            time.sleep(2)

    # -- azioni --
    def _sel(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Nessuna selezione", "Seleziona un esperimento nella tabella.")
            return None
        return next(e for e in exp.EXPERIMENTS if e["id"] == sel[0])

    def show_cmd(self):
        e = next((x for x in exp.EXPERIMENTS if x["id"] in self.tree.selection()), None)
        self.cmd_lbl.config(text=(" ".join(exp.command_for(e)) if e else ""))

    def _busy(self):
        return self.proc is not None and self.proc.poll() is None

    @staticmethod
    def _detect_running():
        """id dell'esperimento con un train.py VERO in esecuzione, o None.
        Preciso: 'train.py' come argomento script (non dentro un `-c`) + --experiment_name,
        cosi' non si confonde con comandi tipo `python -c \"...train.py...\"`."""
        for pr in psutil.process_iter(["cmdline"]):
            try:
                cl = pr.info["cmdline"] or []
                if "-c" in cl or "--experiment_name" not in cl:
                    continue
                if any(str(c).replace("\\", "/").endswith("train.py") for c in cl):
                    return cl[cl.index("--experiment_name") + 1]
            except (psutil.Error, ValueError, IndexError):
                pass
        return None

    def _active(self):
        """L'esperimento che gira davvero: la nostra Popen, o uno rilevato (esterno/dopo restart)."""
        return self.run_id if self._busy() else self._running_id

    def _any_running(self):
        return self._busy() or self._running_id is not None

    def _start(self, e):
        run_dir = LOGS / e["id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        self._logf = open(run_dir / "train_stdout.log", "w", encoding="utf-8")  # tenuto su self: non GC finche' gira
        cmd = exp.command_for(e)
        cmd.insert(1, "-u")                                # python -u: stdout NON bufferizzato -> log live
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        self.proc = subprocess.Popen(cmd, cwd=str(PROJECT), stdout=self._logf,
                                     stderr=subprocess.STDOUT, env=env, creationflags=flags)
        self.run_id = e["id"]
        self._log_size = -1
        self.log.delete("1.0", "end")

    def launch(self):
        e = self._sel()
        if not e:
            return
        if self._any_running():
            messagebox.showwarning("Occupato", f"Un training è già in corso: {self._active()}.")
            return
        self.queue = []          # lancio singolo: nessuna coda
        self._start(e)

    def run_missing(self):
        if self._any_running():
            messagebox.showwarning("Occupato", f"Un training è già in corso: {self._active()}.")
            return
        missing = [e for e in exp.EXPERIMENTS if run_status(e)[0] != "ok"]
        if not missing:
            messagebox.showinfo("Sequenza", "Tutti gli esperimenti sono già completi.")
            return
        if not messagebox.askyesno(
                "Sequenza",
                f"Lancio i {len(missing)} esperimenti mancanti, uno alla volta.\n"
                f"Può durare a lungo (ogni run è ~120 epoche). Procedo?"):
            return
        self.queue = missing[1:]
        self._start(missing[0])

    def _start_next(self):
        if self.queue:
            self._start(self.queue.pop(0))

    def stop(self):
        active = self._active()
        if not active:
            messagebox.showinfo("Niente da fermare", "Nessun training in corso.")
            return
        n = len(self.queue)
        msg = f"Fermare {active}?"
        if n:
            msg += f"\nInterrompe anche la sequenza ({n} ancora in coda)."
        if not messagebox.askyesno("Ferma", msg):
            return
        self.queue = []          # ferma anche la sequenza
        self._kill_training(active)
        self.proc = None
        self._running_id = None

    @staticmethod
    def _kill_training(eid):
        """Uccide il train.py di questo esperimento e i suoi figli (worker), anche se
        non l'abbiamo lanciato noi. Filtro come _detect_running: niente `-c` di passaggio."""
        for pr in psutil.process_iter(["cmdline"]):
            try:
                cl = pr.info["cmdline"] or []
                if "-c" in cl or "--experiment_name" not in cl:
                    continue
                if cl[cl.index("--experiment_name") + 1] == eid \
                        and any(str(c).replace("\\", "/").endswith("train.py") for c in cl):
                    for ch in pr.children(recursive=True):
                        ch.terminate()
                    pr.terminate()
            except (psutil.Error, ValueError, IndexError):
                pass

    def gen_maps(self):
        e = self._sel()
        if not e:
            return
        if not (LOGS / e["id"] / "checkpoints" / "best_model.pt").exists():
            messagebox.showwarning("Manca il checkpoint", f"{e['id']} non ha best_model.pt: prima addestralo.")
            return
        subprocess.Popen([sys.executable, "visualize_v2_attention.py", "--experiment", e["id"],
                          "--out", str(VIZ)], cwd=str(PROJECT))
        messagebox.showinfo("Mappe", f"Generazione avviata per {e['id']} → cartella {VIZ.name}/.")

    def open_dir(self, path):
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(str(path)) if sys.platform == "win32" else subprocess.Popen(["xdg-open", str(path)])

    # -- refresh tabella (letture leggere: 26 json) --
    def refresh(self):
        tagmap = {"ok": "ok", "da fare": "todo", "interrotta": "warn", "in corso": "run",
                  "DIVERGITA": "bad", "illeggibile": "bad"}
        n_ok = 0
        for e in exp.EXPERIMENTS:
            stato, acc = run_status(e)
            n_ok += stato == "ok"
            self.tree.set(e["id"], "stato", stato)
            self.tree.set(e["id"], "acc", f"{acc*100:.2f}%" if acc is not None else "—")
            self.tree.item(e["id"], tags=(tagmap.get(stato, ""),))
        self.done = n_ok
        active = self._active()
        if active and self.tree.exists(active):   # la run che gira resta 'in corso'
            self.tree.set(active, "stato", "in corso")
            self.tree.item(active, tags=("run",))

    # -- loop UI: tutto leggero (label + poll + tail solo se cambia) --
    def tick(self):
        self.sys_lbl.config(text=self.sysline)
        # transizione: la NOSTRA run e' finita?
        if self.proc is not None and self.proc.poll() is not None:
            self.proc = None
            self.refresh()                 # ricalcola lo stato reale (ok / DIVERGITA / interrotta)
            if self.queue:
                self._start_next()         # incatena il prossimo mancante
        active = self._active()            # nostro training o uno rilevato (esterno / dopo restart)
        if active:
            if self.tree.exists(active):
                self.tree.set(active, "stato", "in corso")
                self.tree.item(active, tags=("run",))
            pid = self.proc.pid if self._busy() else "esterno"
            q = f"   coda: {len(self.queue)} rimasti" if self.queue else ""
            self.run_lbl.config(text=f"● training: {active} ({pid}){q}", foreground="#1560d0")
            self._tail_into_log(active)
        else:
            self.run_lbl.config(text=f"completi {self.done}/{len(exp.EXPERIMENTS)}", foreground="#127c12")
        self.root.after(1000, self.tick)

    def _tail_into_log(self, watch, force=False):
        if not watch:
            return
        p = LOGS / watch / "train_stdout.log"
        try:  # leggo solo la coda (ultimi 64 KB): costo costante anche con log enormi
            with open(p, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 65536))
                raw = f.read().decode("utf-8", errors="replace")
        except OSError:
            return
        if not force and watch == self._watch_last and size == self._log_size:
            return                          # niente di nuovo: non ridisegnare
        self._watch_last = watch
        self._log_size = size
        # emula il terminale: \n = nuova riga; \r = torna a inizio riga e sovrascrive
        # (cosi' la barra di tqdm resta UNA riga che si aggiorna, non centinaia)
        raw = raw.replace("\r\n", "\n")
        lines = [next((x for x in reversed(seg.split("\r")) if x), "") for seg in raw.split("\n")]
        self.log.delete("1.0", "end")
        self.log.insert("end", "\n".join(lines[-300:]))
        self.log.see("end")

    def _close(self):
        self._alive = False
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
