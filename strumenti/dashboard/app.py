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
PROJECT = ROOT / "progetto"
LOGS = PROJECT / "logs"
VIZ = PROJECT / "perceiver_visualizations_v2"

sys.path.insert(0, str(PROJECT))
import experiments as exp  # noqa: E402  -> EXPERIMENTS, command_for

# Oltre questa eta', una cartella con residui ma senza risultato non e' piu'
# "interrotta" (qualcosa che hai fermato poco fa) ma un vecchio residuo = "da fare".
INTERROTTA_MAX_ETA_SEC = 30 * 60

# Non tutte le run passano da train.py: io_glue_multitask usa multitask_glue.py e
# cnn_baseline usa baseline_cnn.py. Filtrando solo su train.py la dashboard non
# vedeva girare quelle due -- e il pulsante Lancia ne avrebbe avviata una seconda
# sopra la prima, sullo stesso log. L'insieme si ricava dal registro, cosi' uno
# script nuovo e' coperto senza toccare questo file.
RUN_SCRIPTS = {e.get("script", "train.py") for e in exp.EXPERIMENTS}


def _e_uno_script_di_run(cmdline):
    return any(str(c).replace("\\", "/").rsplit("/", 1)[-1] in RUN_SCRIPTS for c in cmdline)


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

# Valore riportato dal paper per la riga che ogni run replica. Le fonti sono le
# tabelle gia' trascritte nel cap. 13 della lezione (sito/lezione/index.html):
# Tab.2 permutazione/PE, Tab.5 solo cross-attend, Tab.6 numero di cross-attend,
# Tab.7 weight sharing, Tab.4 ModelNet40.
#
# Dove il paper non riporta quella configurazione si lascia vuoto invece di
# stimare: Fig.6 e' un grafico senza numeri in tabella, l'at-start e' dato solo
# per T=8, e il ramo IO non ha ne' CIFAR-10 ne' l'accuratezza sui byte mascherati.
# GLUE il paper lo da' come media sugli 8 task (81.0 byte-level, BERT 81.1), non
# per singolo task: metterla su ogni riga sarebbe un confronto falso.
#
# ATTENZIONE al confronto diretto: il paper e' ImageNet a piena scala, queste run
# sono CIFAR-10 ridotto. Tab.5 e Tab.6 vengono per giunta dal modello ablativo
# ridotto dell'App. B (~60-76%), non dal modello principale al 78.0%. La colonna
# serve a sapere quale riga si sta replicando, non a leggere la differenza.
PAPER_ACC = {
    "e01_baseline":            "78.0%",   # Tab.2, Perceiver con Fourier features
    "e02_permuted":            "78.0%",   # Tab.2, stesso modello sotto permutazione
    "e03_learned_pe":          "70.9%",   # Tab.2, learned position encoding
    "e04_learned_pe_permuted": "70.9%",   # Tab.2, learned PE sotto permutazione
    "e05_no_latent_T4":        "39.4%",   # Tab.5, 4 cross-attend senza self-attention
    "e06_no_latent_T8":        "45.3%",   # Tab.5, 8 cross-attend senza self-attention
    "e07_no_latent_T12":       "OOM",     # Tab.5, 12 va in out of memory nel paper
    "e08_T1_interleaved":      "76.7%",   # Tab.6, interleaved T=1
    "e09_T2_interleaved":      "76.5%",   # Tab.6, interleaved T=2
    "e10_T8_interleaved":      "78.0%",   # Tab.6, interleaved T=8 (configurazione finale)
    "e14_T8_at_start":         "73.7%",   # Tab.6, at-start T=8 (unico at-start riportato)
    "e16_no_weight_sharing":   "72.9%",   # Tab.7, val senza weight sharing (326M param)
    "mn01_baseline":           "85.7%",   # Tab.4, ModelNet40
}

# Un id sbagliato qui sparirebbe in silenzio: la colonna mostrerebbe "—" e
# sembrerebbe che il paper non riporti quella riga. Meglio non partire.
_IGNOTI = sorted(set(PAPER_ACC) - {e["id"] for e in exp.EXPERIMENTS})
assert not _IGNOTI, f"PAPER_ACC cita run che non esistono nel registro: {_IGNOTI}"

# Etichetta leggibile del dataset, per la riga di confronto.
DATASET_LABEL = {
    "cifar10": "CIFAR-10 32x32", "modelnet40": "ModelNet40 2048 punti",
    "wikitext103": "WikiText-103 byte-level",
    "glue_sst2": "GLUE SST-2", "glue_cola": "GLUE CoLA", "glue_mrpc": "GLUE MRPC",
    "glue_stsb": "GLUE STS-B", "glue_qqp": "GLUE QQP", "glue_mnli": "GLUE MNLI",
    "glue_qnli": "GLUE QNLI", "glue_rte": "GLUE RTE",
}


def setup_nostro(e):
    """Config EFFETTIVA della run, letta dal comando che verrebbe lanciato.
    Ricavarla qui invece di scriverla a mano evita che resti indietro quando un
    override la cambia: e10 gira a T=8, non al T=4 della config base."""
    cmd = exp.command_for(e)

    def val(flag, default=None):
        # ULTIMA occorrenza: command_for concatena base + override e argparse
        # tiene l'ultimo, quindi la prima darebbe il valore base (e10 -> T=4).
        if flag not in cmd:
            return default
        return cmd[len(cmd) - 1 - cmd[::-1].index(flag) + 1]

    ds = val("--dataset") or ("cifar10" if e.get("modality") == "baseline" else "?")
    pezzi = [DATASET_LABEL.get(ds, ds)]
    if e.get("modality") == "baseline":
        pezzi.append("CNN di riferimento, non Perceiver")
    if val("--model_type") == "perceiver_io":
        q = val("--num_output_queries")
        pezzi.append(f"IO, {q} query" if q else "IO, 1 query per posizione")

    dims = [f"{k}={v}" for k, v in (
        ("N", val("--num_latents")), ("D", val("--latent_dim")),
        ("T", val("--num_cross_attend_stages")), ("l", val("--num_transformer_blocks")),
        # Senza questi, le run di Fig. 6 sembrerebbero tutte identiche al baseline:
        # il loro unico override e' proprio qui.
        ("K", val("--fourier_num_bands")), ("f_max", val("--fourier_max_freq")),
        ("init", val("--latent_init_scale")),
    ) if v]
    if dims:
        pezzi.append(" ".join(dims))

    # Le varianti strutturali cambiano il senso della run: vanno viste subito.
    varianti = [etichetta for flag, etichetta in (
        ("--no_latent_transformer", "NO latent transformer"),
        ("--no_weight_sharing", "NO weight sharing"),
        ("--no_positional_encoding", "NO positional encoding"),
        ("--no_share_cross_attend", "cross-attend non condivise"),
    ) if flag in cmd]
    if val("--cross_attend_arrangement"):
        varianti.append(val("--cross_attend_arrangement").replace("_", " "))
    if val("--seed"):
        varianti.append(f"seed {val('--seed')}")
    if varianti:
        pezzi.append(", ".join(varianti))

    train = [t for t in (
        (val("--optimizer") or "").upper() or None,
        val("--lr"), val("--scheduler"),
        f"{val('--epochs')} ep" if val("--epochs") else None,
    ) if t]
    if train:
        pezzi.append(", ".join(train))
    return " | ".join(pezzi)


# Setup del PAPER per la riga replicata. Fonti: cap. 3 della lezione (config
# ImageNet del modello principale) e cap. 13 (avvertenza App. B + tabelle).
# Dove il paper non copre il caso lo si dice, invece di lasciare intendere che
# esista un termine di paragone.
SETUP_PAPER = {
    "tab1":     "ImageNet 224x224 (M=50.176, C_tot=261) | N=512 D=1024 T=8 l=6 H=8 K=64, weight sharing | 78.0%",
    "tab2":     "ImageNet 224x224 | stesso modello principale | Fourier 78.0% -> 78.0% permutato, learned PE 70.9% -> 70.9%",
    "tab5":     "ImageNet | modello ridotto App. B: D=512, l=4, 2 cross-attend, NO weight sharing, batch 64 (accuracy 60-76%)",
    "tab6":     "ImageNet | modello ridotto App. B (D=512, l=4, no weight sharing); la riga a 78.0% e' invece il modello principale",
    "tab7":     "ImageNet | 326M senza weight sharing vs 45M con | val 72.9% vs 78.0%, train 87.7% vs 79.5%",
    "fig6":     "ImageNet | Fig. 6 e' un grafico: il paper non da' numeri in tabella per bande, f_max e init scale",
    "noise":    "il paper non riporta la dispersione fra seed: la banda di rumore e' misurata solo qui",
    "modelnet": "ModelNet40 2048 punti (Tab. 4) | 85.7%",
    "io_image": "il paper Perceiver IO non valuta su CIFAR-10: nessun termine di paragone diretto",
    "io_mlm":   "il paper IO da' BPC 1.74 su language modeling byte-level (201M par), non l'accuratezza sui byte mascherati",
    "io_glue":  "GLUE byte-level, media sugli 8 task 81.0 (BERT 81.1) | il paper non da' il dettaglio per singolo task",
    "baseline": "in Tab. 1 il paper confronta con ResNet-50 (73.5%) e ViT-B/16 (76.7%) su ImageNet",
}

# Stessa ragione dell'assert sopra: un gruppo non mappato mostrerebbe una riga
# vuota, che si legge come "il paper non lo copre" ed e' un'altra cosa.
_GRP_SCOPERTI = sorted({e["group"] for e in exp.EXPERIMENTS} - set(SETUP_PAPER))
assert not _GRP_SCOPERTI, f"SETUP_PAPER non copre i gruppi: {_GRP_SCOPERTI}"



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

        cols = ("id", "paper", "config", "mod", "stato", "paper acc", "acc")
        widths = (170, 150, 430, 90, 110, 80, 70)
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
                e.get("modality", "image"), "…",
                PAPER_ACC.get(e["id"], "—"), "—"))
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

        conf = ttk.Frame(root, padding=(8, 4, 8, 0))
        conf.pack(fill="x")
        self.setup_nostro = ttk.Label(conf, text="", font=("Consolas", 9), foreground="#127c12", anchor="w")
        self.setup_nostro.pack(fill="x")
        self.setup_paper = ttk.Label(conf, text="", font=("Consolas", 9), foreground="#8a5000", anchor="w")
        self.setup_paper.pack(fill="x")

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
        nostro = setup_nostro(e) if e else ""
        paper = SETUP_PAPER.get(e["group"], "") if e else ""
        self.setup_nostro.config(text=f"NOSTRO  {nostro}" if e else "")
        self.setup_paper.config(text=f"PAPER   {paper}" if e else "")

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
                if _e_uno_script_di_run(cl):
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
                if cl[cl.index("--experiment_name") + 1] == eid and _e_uno_script_di_run(cl):
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
