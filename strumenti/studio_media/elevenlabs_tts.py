"""Official CLI voice, with one shared conservative Free budget and a local cache.

Credentials stay in the CLI's credential store. This module never plays media.
An uncertain request remains charged locally, so retries cannot spend twice.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

RUNTIME = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".cache"))) / "PerceiverTutor"
CONFIG = RUNTIME / "elevenlabs-runtime.json"
STATE = RUNTIME / "elevenlabs"
SETTINGS = {"stability": .5, "similarity_boost": .75}
MAX_AUDIO = 8 * 1024 * 1024
FREE_LIMIT = 10000


class ElevenLabsError(RuntimeError):
    def __init__(self, code: str, message: str, *, http_status: int | None = None, api_code: str | None = None,
                 return_code: int | None = None, stdout_length: int | None = None, stderr_length: int | None = None,
                 operation: str | None = None):
        self.code, self.message = code, message
        self.http_status, self.api_code = http_status, api_code
        self.return_code, self.stdout_length, self.stderr_length = return_code, stdout_length, stderr_length
        self.operation = operation
        super().__init__(message)


def _config() -> dict:
    try:
        value = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        raise ElevenLabsError("ELEVENLABS_DISABLED", "La voce ElevenLabs non è configurata sul PC.")
    if not isinstance(value, dict) or value.get("enabled") is not True:
        raise ElevenLabsError("ELEVENLABS_DISABLED", "La voce ElevenLabs è disattivata sul PC.")
    cli, voice, model = (value.get(key) for key in ("cli", "voice", "model"))
    if (not isinstance(cli, str) or not Path(cli).is_file()
            or not isinstance(voice, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", voice)
            or model != "eleven_v4"):
        raise ElevenLabsError("ELEVENLABS_CONFIG", "Controlla la configurazione della voce ElevenLabs sul PC.")
    return {"cli": cli, "voice": voice, "model": model}


def status() -> dict:
    """Local-only status: no account fields, paths, credentials or network requests."""
    try:
        _config()
        return {"enabled": True, "available": True, "engine": "ElevenLabs · italiano", "reason": None}
    except ElevenLabsError as error:
        return {"enabled": error.code != "ELEVENLABS_DISABLED", "available": False,
                "engine": "ElevenLabs · italiano", "reason": error.message}


@contextmanager
def _lock():
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "budget.lock").open("a+b") as handle:
        if not handle.seek(0, 2):
            handle.write(b"0")
            handle.flush()
        deadline = time.monotonic() + 30
        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise ElevenLabsError("ELEVENLABS_BUSY", "Un altro audio ElevenLabs è in preparazione. Attendi e riprova.")
                time.sleep(.1)
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _cli_failure(stdout: str, stderr: str, return_code: int | None = None, operation: str | None = None) -> ElevenLabsError:
    """Extract allowlisted diagnostics, never retain or expose the CLI response."""
    raw = re.sub(r"\x1b\[[0-9;]*m", "", (stderr + "\n" + stdout)[:65536])
    allowed = {"quota_exceeded", "insufficient_credits", "character_limit_exceeded", "max_character_limit_exceeded",
               "rate_limit_exceeded", "too_many_concurrent_requests", "system_busy", "unauthorized",
               "invalid_api_key", "authentication_error", "invalid_authorization_header", "insufficient_permissions",
               "missing_permissions", "payment_required", "subscription_required", "voice_not_found", "model_not_found",
               "model_not_available", "invalid_tts_model", "text_too_long", "invalid_request", "trial_exceeded",
               "detected_unusual_activity", "unusual_activity"}
    codes, statuses = [], []
    decoder = json.JSONDecoder()
    for match in list(re.finditer(r"\{", raw))[:30]:
        try:
            value, _ = decoder.raw_decode(raw[match.start():])
        except ValueError:
            continue
        if not isinstance(value, dict):
            continue
        nodes = [value] + [value[key] for key in ("detail", "error") if isinstance(value.get(key), dict)]
        for node in nodes:
            for key in ("code", "status", "error_code"):
                if isinstance(node.get(key), str) and node[key] in allowed:
                    codes.append(node[key])
            for key in ("http_status", "status_code", "status"):
                if type(node.get(key)) is int and 400 <= node[key] <= 599:
                    statuses.append(node[key])
    match = re.search(r"(?:HTTP(?:/\d(?:\.\d)?)?(?:\s+error)?|API\s+error|status(?:[_\s]+code)?)\s*[:=(]?\s*[\"']?([45]\d\d)\b", raw, re.I)
    http_status = statuses[0] if statuses else int(match[1]) if match else None
    api_code = codes[0] if codes else None
    if api_code in {"quota_exceeded", "insufficient_credits", "character_limit_exceeded", "max_character_limit_exceeded"}:
        code, message = "ELEVENLABS_QUOTA", "ElevenLabs segnala che i caratteri gratuiti non bastano. Puoi leggere il testo o scegliere una voce locale nelle Opzioni."
    elif api_code in {"rate_limit_exceeded", "too_many_concurrent_requests", "system_busy"} or http_status == 429:
        code, message = "ELEVENLABS_RATE_LIMIT", "ElevenLabs ha limitato temporaneamente le richieste. Non ho ripetuto la conversione; attendi prima di creare un altro audio."
    elif (api_code in {"unauthorized", "invalid_api_key", "authentication_error", "invalid_authorization_header"}
          or http_status == 401 or re.search(r"not authenticated|authentication required|not logged in|no credentials|missing API key|invalid API key", raw, re.I)):
        code, message = "ELEVENLABS_AUTH", "Il collegamento ElevenLabs deve essere rinnovato sul PC. La risposta scritta resta disponibile."
    elif http_status == 403 or api_code in {"insufficient_permissions", "missing_permissions"}:
        code, message = "ELEVENLABS_FORBIDDEN", "ElevenLabs ha rifiutato questa richiesta. Controlla l'accesso e il piano sul PC; non ho ripetuto la conversione."
    else:
        code, message = "ELEVENLABS_UNAVAILABLE", "ElevenLabs non ha creato l'audio. Non ho ripetuto la richiesta; controlla account e crediti sul PC."
    return ElevenLabsError(code, message, http_status=http_status, api_code=api_code,
                           return_code=return_code, stdout_length=len(stdout), stderr_length=len(stderr), operation=operation)


def _run(config: dict, arguments: list[str], payload: dict | None = None, timeout: int = 45) -> str:
    operation = "subscription" if arguments[:3] == ["user", "subscription", "get"] else "speech"
    try:
        result = subprocess.run([config["cli"], *arguments],
                                input=None if payload is None else json.dumps(payload, ensure_ascii=False),
                                text=True, encoding="utf-8", capture_output=True, timeout=timeout,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        raise ElevenLabsError("ELEVENLABS_TIMEOUT", "ElevenLabs non ha risposto in tempo. La risposta scritta resta disponibile.", operation=operation)
    except (OSError, UnicodeError):
        raise ElevenLabsError("ELEVENLABS_UNAVAILABLE", "Il programma ElevenLabs non risponde. Controlla il collegamento sul PC.", operation=operation)
    if result.returncode or len(result.stdout) > 65536:
        raise _cli_failure(result.stdout, result.stderr, result.returncode, operation)
    return result.stdout


def _subscription(config: dict) -> dict:
    arguments = ["user", "subscription", "get", "--format", "json", "--no-retry", "--query",
                 "{tier:tier,used:character_count,limit:character_limit,reset:next_character_count_reset_unix,can:can_extend_character_limit,allowed:allowed_to_extend_character_limit}"]
    for attempt in range(2):
        try:
            raw = _run(config, arguments)
            break
        except ElevenLabsError as error:
            error.operation = "subscription"
            transient = (error.code in {"ELEVENLABS_TIMEOUT", "ELEVENLABS_RATE_LIMIT"}
                         or error.code == "ELEVENLABS_UNAVAILABLE" and (error.http_status is None or error.http_status >= 500))
            if attempt or not transient:
                raise
            time.sleep(1)  # One retry of this read-only check; speech is never retried.
    try:
        account = json.loads(raw)
        if (not isinstance(account, dict) or account.get("tier") != "free"
                or account.get("can") is not False or account.get("allowed") is not False):
            raise ElevenLabsError("ELEVENLABS_FREE_REQUIRED", "Questa voce usa solo il piano gratuito senza estensioni a pagamento. Controlla il piano sul PC.", operation="subscription")
        if (any(type(account.get(key)) is not int for key in ("used", "limit", "reset"))
                or not 0 <= account["used"] <= account["limit"] <= FREE_LIMIT
                or account["limit"] <= 0 or account["reset"] <= int(time.time())):
            raise ValueError("Invalid Free allowance")
        return account
    except (ValueError, TypeError):
        raise ElevenLabsError("ELEVENLABS_ACCOUNT", "Non riesco a verificare i caratteri gratuiti. Non genero audio finché il piano non è verificabile.", operation="subscription")


def _save_budget(value: dict):
    temporary = STATE / "budget.tmp"
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(STATE / "budget.json")


def _budget(account: dict) -> dict:
    try:
        budget = json.loads((STATE / "budget.json").read_text(encoding="utf-8"))
        if (not isinstance(budget, dict) or budget.get("version") != 1
                or type(budget.get("period")) is not int or type(budget.get("reservedCharacters")) is not int
                or not 0 <= budget["reservedCharacters"] <= FREE_LIMIT or budget["period"] > account["reset"]):
            raise ValueError("Invalid local allowance")
        if budget["period"] < account["reset"]:
            # Reset only after the old period ended and the API confirms its successor.
            if budget["period"] > int(time.time()):
                raise ValueError("Premature reset")
            budget = {"version": 1, "period": account["reset"], "reservedCharacters": account["used"]}
        return budget
    except (OSError, ValueError, TypeError):
        raise ElevenLabsError("ELEVENLABS_BUDGET", "Il conteggio gratuito sul PC deve essere inizializzato o verificato prima di creare altri audio.")


def initialize_budget(initial_characters: int = 0) -> dict:
    """Record known previous usage before enabling the exporter and tutor (no speech)."""
    if type(initial_characters) is not int or not 0 <= initial_characters <= FREE_LIMIT:
        raise ValueError("Invalid initial usage")
    config = _config()
    with _lock():
        account = _subscription(config)
        if (STATE / "budget.json").exists():
            budget = _budget(account)
            budget["reservedCharacters"] = max(budget["reservedCharacters"], initial_characters, account["used"])
        else:
            budget = {"version": 1, "period": account["reset"], "reservedCharacters": max(initial_characters, account["used"])}
        _save_budget(budget)
        return {"remainingCharacters": max(0, account["limit"] - budget["reservedCharacters"])}


def _key(text: str, config: dict) -> str:
    body = {"text": text, "voice": config["voice"], "model": config["model"], "language": "it", "settings": SETTINGS}
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _valid_mp3(path: Path) -> bool:
    """Reject empty, oversized, JSON/HTML and invalid MP3 headers before serving."""
    try:
        if path.is_symlink() or not 256 <= path.stat().st_size <= MAX_AUDIO:
            return False
        with path.open("rb") as handle:
            first = handle.read(10)
            offset = 0
            if first.startswith(b"ID3"):
                if len(first) < 10 or any(value & 128 for value in first[6:10]):
                    return False
                offset = 10 + sum(value << shift for value, shift in zip(first[6:10], (21, 14, 7, 0)))
            handle.seek(offset)
            data = handle.read(16384)
        for index in range(max(0, len(data) - 3)):
            a, b, c = data[index:index + 3]
            if (a == 255 and b & 224 == 224 and b & 24 != 8 and b & 6 == 2
                    and c & 240 not in (0, 240) and c & 12 != 12):
                return True
        return False
    except OSError:
        return False


def _text(text: str):
    if not isinstance(text, str) or not text.strip() or len(text) > FREE_LIMIT:
        raise ElevenLabsError("ELEVENLABS_TEXT", "La frase da leggere è vuota o troppo lunga.")


def _quota(account: dict, budget: dict, characters: int) -> int:
    remaining = account["limit"] - max(account["used"], budget["reservedCharacters"])
    if characters > remaining:
        raise ElevenLabsError("ELEVENLABS_QUOTA", "Non bastano i caratteri gratuiti ElevenLabs per questo audio. Puoi leggere il testo o scegliere una voce locale nelle Opzioni.")
    return remaining


def preflight(texts: list[str]) -> dict:
    """Check the complete uncached batch before spending; do not reserve it."""
    if not isinstance(texts, list):
        raise ValueError("Expected a list of texts")
    config = _config()
    unique = {}
    for text in texts:
        _text(text)
        unique[_key(text, config)] = text
    with _lock():
        missing = [text for key, text in unique.items() if not _valid_mp3(STATE / "audio" / (key + ".mp3"))]
        account = _subscription(config)
        budget = _budget(account)
        characters = sum(map(len, missing))
        remaining = _quota(account, budget, characters)
        return {"engine": "ElevenLabs", "uncachedCharacters": characters, "remainingCharacters": remaining,
                "cachedRequests": len(unique) - len(missing)}


def synthesize(text: str, output: Path) -> dict:
    _text(text)
    config = _config()
    output = Path(output)
    if output.suffix.lower() != ".mp3":
        raise ElevenLabsError("ELEVENLABS_OUTPUT", "L'audio ElevenLabs deve essere salvato come MP3.")
    cache = STATE / "audio" / (_key(text, config) + ".mp3")
    with _lock():
        cached = _valid_mp3(cache)
        if not cached:
            account = _subscription(config)
            budget = _budget(account)
            _quota(account, budget, len(text))
            budget["reservedCharacters"] = max(account["used"], budget["reservedCharacters"]) + len(text)
            _save_budget(budget)  # Reserve durably before any generation, including timeouts.
            cache.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="request-", dir=STATE) as directory:
                generated = Path(directory) / "speech.mp3"
                payload = {"text": text, "model_id": config["model"], "language_code": "it", "voice_settings": SETTINGS}
                _run(config, ["text-to-speech", "convert", "--voice-id", config["voice"], "--output-format", "mp3_44100_128",
                              "--json", "-", "--output", str(generated), "--no-retry"], payload, timeout=180)
                if not _valid_mp3(generated):
                    raise ElevenLabsError("ELEVENLABS_AUDIO", "ElevenLabs non ha restituito un MP3 valido. Non ho ripetuto la richiesta.")
                generated.replace(cache)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.resolve() != cache.resolve():
            shutil.copyfile(cache, output)
        return {"path": str(output), "engine": "ElevenLabs", "voice": config["voice"], "model": config["model"],
                "cached": cached, "characters": len(text)}
