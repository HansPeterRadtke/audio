#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import queue
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText


DATA_ROOT = Path("/data")
BIN_ROOT = DATA_ROOT / "bin"
HOSTS_ROOT = DATA_ROOT / "etc" / "hosts"
PORTS_PATH = DATA_ROOT / "etc" / "ports.json"
TMP_ROOT = DATA_ROOT / "var" / "tmp" / "tts_gui"

ENGINE_OPTIONS = ("auto", "index", "fish", "cosy", "miotts", "piper")
LANGUAGE_OPTIONS = ("auto", "de", "en")


def bootstrap_audio_env() -> None:
    if not os.environ.get("XDG_RUNTIME_DIR"):
        runtime_dir = f"/run/user/{os.getuid()}"
        if os.path.isdir(runtime_dir):
            os.environ["XDG_RUNTIME_DIR"] = runtime_dir
    if os.environ.get("XDG_RUNTIME_DIR") and not os.environ.get("PULSE_SERVER"):
        pulse_native = Path(os.environ["XDG_RUNTIME_DIR"]) / "pulse" / "native"
        if pulse_native.exists():
            os.environ["PULSE_SERVER"] = f"unix:{pulse_native}"


def host_short() -> str:
    host = os.environ.get("INFRA_HOST") or socket.gethostname()
    host = host.split(".", 1)[0].lower()
    if host.startswith("nitro"):
        return "nitro"
    if host.startswith("jetson"):
        return "jetson"
    return host


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def host_config() -> dict[str, Any]:
    short = host_short()
    for candidate in (HOSTS_ROOT / f"{short}.json", HOSTS_ROOT / f"{socket.gethostname()}.json"):
        if candidate.exists():
            return load_json(candidate)
    return {}


def tts_endpoint() -> tuple[str, int, str]:
    cfg = host_config()
    ports = load_json(PORTS_PATH)
    host = str(((cfg.get("tts") or {}).get("host")) or "127.0.0.1")
    base = int(((ports.get("services") or {}).get("tts") or {}).get("base") or 15101)
    instance = int(((cfg.get("tts") or {}).get("instance")) or 0)
    secret = str(((cfg.get("tts") or {}).get("secret_key")) or "")
    return host, base + instance, secret


def server_health(timeout: float = 1.5) -> dict[str, Any] | None:
    host, port, _secret = tts_endpoint()
    url = f"http://{host}:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


def current_sink_name() -> str:
    bootstrap_audio_env()
    try:
        proc = subprocess.run(
            ["pactl", "info"],
            check=False,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
    except FileNotFoundError:
        return "unknown"
    if proc.returncode != 0:
        return "unknown"
    for line in proc.stdout.splitlines():
        if line.startswith("Default Sink:"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def play_wav(path: Path) -> None:
    bootstrap_audio_env()
    candidates = (
        ("paplay", [str(path)]),
        ("pw-play", [str(path)]),
        ("aplay", [str(path)]),
        ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "error", str(path)]),
        ("cvlc", ["--play-and-exit", "--intf", "dummy", str(path)]),
    )
    env = os.environ.copy()
    for cmd, args in candidates:
        if shutil.which(cmd):
            subprocess.run([cmd, *args], check=True, env=env)
            return
    raise RuntimeError("no audio playback command found")


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate() or 1
        frames = handle.getnframes()
    return float(frames) / float(rate)


def format_seconds(value: float) -> str:
    total = max(0, int(round(value)))
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def list_piper_models() -> list[str]:
    root = DATA_ROOT / "models" / "piper"
    if not root.is_dir():
        return []
    return [str(p) for p in sorted(root.rglob("*.onnx"))]


def list_reference_voices() -> list[str]:
    candidates: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        key = str(path)
        if key in seen or not path.is_file():
            return
        seen.add(key)
        candidates.append(path)

    fixed = [
        DATA_ROOT / "etc" / "tts" / "speaker.wav",
        DATA_ROOT / "tmp" / "speaker.wav",
        DATA_ROOT / "tmp" / "fish_ref.wav",
        DATA_ROOT / "var" / "tmp" / "tts_speaker_10s.wav",
    ]
    for p in fixed:
        add(p)

    voice_root = DATA_ROOT / "models" / "voices"
    if voice_root.is_dir():
        for p in sorted(voice_root.glob("*.wav")):
            add(p)

    return [str(p) for p in candidates]


class TTSGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("TTS GUI")
        self.geometry("1240x860")
        self.minsize(1100, 760)

        TMP_ROOT.mkdir(parents=True, exist_ok=True)

        self.log_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.busy = False

        self.engine_var = tk.StringVar(value="auto")
        self.lang_var = tk.StringVar(value="auto")
        self.mode_var = tk.StringVar(value="server")
        self.no_speaker_var = tk.BooleanVar(value=False)
        self.style_var = tk.StringVar()
        self.prompt_var = tk.StringVar()
        self.chunk_length_var = tk.StringVar()
        self.piper_model_var = tk.StringVar()
        self.piper_speaker_id_var = tk.StringVar(value="0")
        self.piper_length_scale_var = tk.StringVar()
        self.piper_noise_scale_var = tk.StringVar()
        self.piper_noise_w_var = tk.StringVar()
        self.piper_sentence_silence_var = tk.StringVar()
        self.speaker_var = tk.StringVar()
        self.server_status_var = tk.StringVar(value="Server: checking...")
        self.server_detail_var = tk.StringVar(value="")
        self.sink_var = tk.StringVar(value=f"Sink: {current_sink_name()}")
        self.audio_status_var = tk.StringVar(value="Audio: none")
        self.playback_status_var = tk.StringVar(value="Playback: stopped")
        self.timeline_label_var = tk.StringVar(value="00:00 / 00:00")
        self.health_data: dict[str, Any] | None = None
        self.current_audio_path: Path | None = None
        self.current_audio_is_temporary = False
        self.current_audio_duration = 0.0
        self.play_proc: subprocess.Popen[str] | None = None
        self.play_started_at = 0.0
        self.play_offset = 0.0
        self.playback_after_id: str | None = None

        self._build_ui()
        self._refresh_voice_lists()
        self.after(100, self._poll_log_queue)
        self.after(150, self.refresh_status)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        top = ttk.Frame(self, padding=12)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)

        ttk.Label(top, textvariable=self.server_status_var, font=("TkDefaultFont", 11, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(top, textvariable=self.server_detail_var).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Label(top, textvariable=self.sink_var).grid(row=2, column=0, sticky="w", pady=(4, 0))

        btns = ttk.Frame(top)
        btns.grid(row=0, column=1, rowspan=3, sticky="e")
        ttk.Button(btns, text="Refresh", command=self.refresh_status).grid(row=0, column=0, padx=4)
        ttk.Button(btns, text="Start Server", command=self.start_server).grid(row=0, column=1, padx=4)
        ttk.Button(btns, text="Stop Server", command=self.stop_server).grid(row=0, column=2, padx=4)

        opts = ttk.LabelFrame(self, text="Options", padding=12)
        opts.grid(row=1, column=0, sticky="ew", padx=12)
        for col in range(6):
            opts.columnconfigure(col, weight=1 if col in (1, 3, 5) else 0)

        ttk.Label(opts, text="Mode").grid(row=0, column=0, sticky="w")
        mode_box = ttk.Frame(opts)
        mode_box.grid(row=0, column=1, sticky="ew")
        ttk.Radiobutton(mode_box, text="Server", variable=self.mode_var, value="server").pack(side="left")
        ttk.Radiobutton(mode_box, text="One-shot", variable=self.mode_var, value="oneshot").pack(side="left", padx=(8, 0))

        ttk.Label(opts, text="Engine").grid(row=0, column=2, sticky="w", padx=(12, 0))
        ttk.Combobox(opts, textvariable=self.engine_var, values=ENGINE_OPTIONS, state="readonly").grid(
            row=0, column=3, sticky="ew"
        )

        ttk.Label(opts, text="Language").grid(row=0, column=4, sticky="w", padx=(12, 0))
        ttk.Combobox(opts, textvariable=self.lang_var, values=LANGUAGE_OPTIONS, state="readonly").grid(
            row=0, column=5, sticky="ew"
        )

        ttk.Label(opts, text="Reference voice").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.speaker_combo = ttk.Combobox(opts, textvariable=self.speaker_var, state="readonly")
        self.speaker_combo.grid(row=1, column=1, sticky="ew", pady=(8, 0))
        ttk.Checkbutton(opts, text="No speaker", variable=self.no_speaker_var).grid(row=1, column=2, sticky="w", padx=(12, 0), pady=(8, 0))
        ttk.Button(opts, text="Refresh Voices", command=self._refresh_voice_lists).grid(row=1, column=3, sticky="w", pady=(8, 0))

        ttk.Label(opts, text="Piper model").grid(row=1, column=4, sticky="w", padx=(12, 0), pady=(8, 0))
        self.piper_model_combo = ttk.Combobox(opts, textvariable=self.piper_model_var, state="readonly")
        self.piper_model_combo.grid(row=1, column=5, sticky="ew", pady=(8, 0))

        ttk.Label(opts, text="Style").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(opts, textvariable=self.style_var).grid(row=2, column=1, sticky="ew", pady=(8, 0))
        ttk.Label(opts, text="Prompt text").grid(row=2, column=2, sticky="w", padx=(12, 0), pady=(8, 0))
        ttk.Entry(opts, textvariable=self.prompt_var).grid(row=2, column=3, sticky="ew", pady=(8, 0))
        ttk.Label(opts, text="Chunk length").grid(row=2, column=4, sticky="w", padx=(12, 0), pady=(8, 0))
        ttk.Entry(opts, textvariable=self.chunk_length_var).grid(row=2, column=5, sticky="ew", pady=(8, 0))

        ttk.Label(opts, text="Piper speaker-id").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(opts, textvariable=self.piper_speaker_id_var).grid(row=3, column=1, sticky="ew", pady=(8, 0))
        ttk.Label(opts, text="Length scale").grid(row=3, column=2, sticky="w", padx=(12, 0), pady=(8, 0))
        ttk.Entry(opts, textvariable=self.piper_length_scale_var).grid(row=3, column=3, sticky="ew", pady=(8, 0))
        ttk.Label(opts, text="Noise scale").grid(row=3, column=4, sticky="w", padx=(12, 0), pady=(8, 0))
        ttk.Entry(opts, textvariable=self.piper_noise_scale_var).grid(row=3, column=5, sticky="ew", pady=(8, 0))

        ttk.Label(opts, text="Noise W").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(opts, textvariable=self.piper_noise_w_var).grid(row=4, column=1, sticky="ew", pady=(8, 0))
        ttk.Label(opts, text="Sentence silence").grid(row=4, column=2, sticky="w", padx=(12, 0), pady=(8, 0))
        ttk.Entry(opts, textvariable=self.piper_sentence_silence_var).grid(row=4, column=3, sticky="ew", pady=(8, 0))

        body = ttk.Panedwindow(self, orient="vertical")
        body.grid(row=2, column=0, sticky="nsew", padx=12, pady=(0, 12))

        text_frame = ttk.LabelFrame(body, text="Text", padding=8)
        log_frame = ttk.LabelFrame(body, text="Log", padding=8)
        body.add(text_frame, weight=4)
        body.add(log_frame, weight=1)

        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        self.text_box = ScrolledText(text_frame, wrap="word", font=("DejaVu Sans Mono", 12))
        self.text_box.grid(row=0, column=0, sticky="nsew")

        action_bar = ttk.Frame(text_frame)
        action_bar.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(action_bar, text="Speak Using Selected Mode", command=self.speak_selected_mode).pack(side="left")
        ttk.Button(action_bar, text="Speak Once", command=lambda: self.speak("oneshot")).pack(side="left", padx=(8, 0))
        ttk.Button(action_bar, text="Speak Via Server", command=lambda: self.speak("server")).pack(side="left", padx=(8, 0))
        ttk.Button(action_bar, text="Clear Text", command=lambda: self.text_box.delete("1.0", "end")).pack(side="right")

        player_frame = ttk.LabelFrame(text_frame, text="Player", padding=8)
        player_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        player_frame.columnconfigure(0, weight=1)

        ttk.Label(player_frame, textvariable=self.audio_status_var).grid(row=0, column=0, sticky="w")
        ttk.Label(player_frame, textvariable=self.playback_status_var).grid(row=0, column=1, sticky="e", padx=(12, 0))

        player_buttons = ttk.Frame(player_frame)
        player_buttons.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(player_buttons, text="Play", command=self.play_current_audio).pack(side="left")
        ttk.Button(player_buttons, text="Stop", command=self.stop_current_audio).pack(side="left", padx=(8, 0))
        ttk.Label(player_buttons, textvariable=self.timeline_label_var).pack(side="right")

        self.timeline_canvas = tk.Canvas(player_frame, height=24, highlightthickness=1, highlightbackground="#808080")
        self.timeline_canvas.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.timeline_canvas.bind("<Configure>", lambda _event: self._draw_timeline())
        self.timeline_canvas.bind("<Button-1>", self._seek_timeline)

        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_box = ScrolledText(log_frame, wrap="word", height=10, state="disabled")
        self.log_box.grid(row=0, column=0, sticky="nsew")

    def _refresh_voice_lists(self) -> None:
        speakers = list_reference_voices()
        self.speaker_combo["values"] = speakers
        if speakers and self.speaker_var.get() not in speakers:
            self.speaker_var.set(speakers[0])

        piper_models = list_piper_models()
        self.piper_model_combo["values"] = piper_models
        if piper_models and self.piper_model_var.get() not in piper_models:
            preferred = next((p for p in piper_models if "de_DE-thorsten-high" in p), piper_models[0])
            self.piper_model_var.set(preferred)

    def _log(self, message: str, level: str = "INFO") -> None:
        self.log_queue.put((level, message))

    def _poll_log_queue(self) -> None:
        try:
            while True:
                level, message = self.log_queue.get_nowait()
                self.log_box.configure(state="normal")
                ts = time.strftime("%H:%M:%S")
                self.log_box.insert("end", f"[{ts}] {level}: {message}\n")
                self.log_box.see("end")
                self.log_box.configure(state="disabled")
        except queue.Empty:
            pass
        self.after(120, self._poll_log_queue)

    def _set_busy(self, value: bool) -> None:
        self.busy = value

    def _current_play_position(self) -> float:
        position = self.play_offset
        if self.play_proc is not None and self.play_proc.poll() is None:
            position += max(0.0, time.monotonic() - self.play_started_at)
        return min(self.current_audio_duration, position)

    def _set_audio_status(self, path: Path | None) -> None:
        if path is None:
            self.audio_status_var.set("Audio: none")
            self.timeline_label_var.set("00:00 / 00:00")
            return
        self.audio_status_var.set(f"Audio: {path.name} ({format_seconds(self.current_audio_duration)})")
        self.timeline_label_var.set(f"{format_seconds(self._current_play_position())} / {format_seconds(self.current_audio_duration)}")

    def _clear_playback_poll(self) -> None:
        if self.playback_after_id is not None:
            self.after_cancel(self.playback_after_id)
            self.playback_after_id = None

    def _draw_timeline(self) -> None:
        canvas = self.timeline_canvas
        canvas.delete("all")
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        canvas.create_rectangle(0, 0, width, height, fill="#202020", outline="")
        if self.current_audio_duration > 0:
            progress = self._current_play_position() / self.current_audio_duration
            progress = min(1.0, max(0.0, progress))
            canvas.create_rectangle(0, 0, int(width * progress), height, fill="#4a90e2", outline="")
        canvas.create_rectangle(0, 0, width, height, outline="#808080")
        self.timeline_label_var.set(
            f"{format_seconds(self._current_play_position())} / {format_seconds(self.current_audio_duration)}"
        )

    def _delete_audio_file(self, path: Path | None, *, temporary: bool) -> None:
        if not temporary or path is None:
            return
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

    def _reset_audio(self) -> None:
        self.stop_current_audio(reset_position=True)
        old_path = self.current_audio_path
        old_temporary = self.current_audio_is_temporary
        self.current_audio_path = None
        self.current_audio_is_temporary = False
        self.current_audio_duration = 0.0
        self.play_offset = 0.0
        self._delete_audio_file(old_path, temporary=old_temporary)
        self._set_audio_status(None)
        self.playback_status_var.set("Playback: stopped")
        self._draw_timeline()

    def _load_audio(self, path: Path, *, temporary: bool) -> None:
        self._reset_audio()
        self.current_audio_path = path
        self.current_audio_is_temporary = temporary
        self.current_audio_duration = wav_duration(path)
        self.play_offset = 0.0
        self.playback_status_var.set("Playback: ready")
        self._set_audio_status(path)
        self._draw_timeline()

    def _player_command(self, path: Path, offset: float) -> list[str]:
        bootstrap_audio_env()
        if shutil.which("ffplay"):
            cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error"]
            if offset > 0.0:
                cmd.extend(["-ss", f"{offset:.3f}"])
            cmd.append(str(path))
            return cmd
        if offset <= 0.0:
            for name in ("paplay", "pw-play", "aplay"):
                if shutil.which(name):
                    return [name, str(path)]
        raise RuntimeError("no seekable audio playback command found")

    def _playback_finished(self) -> None:
        self._clear_playback_poll()
        self.play_proc = None
        self.play_started_at = 0.0
        self.play_offset = self.current_audio_duration
        if self.current_audio_path is not None:
            self.playback_status_var.set("Playback: finished")
            self._set_audio_status(self.current_audio_path)
        else:
            self.playback_status_var.set("Playback: stopped")
        self._draw_timeline()

    def _poll_playback(self) -> None:
        if self.play_proc is None:
            self._clear_playback_poll()
            self._draw_timeline()
            return
        if self.play_proc.poll() is not None:
            self._playback_finished()
            return
        self._draw_timeline()
        self.playback_after_id = self.after(150, self._poll_playback)

    def _start_playback(self, offset: float) -> None:
        if self.current_audio_path is None:
            raise RuntimeError("no rendered audio available")
        self.stop_current_audio(reset_position=False)
        offset = min(self.current_audio_duration, max(0.0, offset))
        cmd = self._player_command(self.current_audio_path, offset)
        self.play_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=os.environ.copy(),
        )
        self.play_offset = offset
        self.play_started_at = time.monotonic()
        self.playback_status_var.set("Playback: playing")
        self._draw_timeline()
        self._poll_playback()

    def play_current_audio(self) -> None:
        if self.current_audio_path is None:
            self._log("No rendered audio available", "WARN")
            return
        try:
            offset = self._current_play_position()
            if offset >= self.current_audio_duration:
                offset = 0.0
            self._start_playback(offset)
        except Exception as exc:  # noqa: BLE001
            self._log(str(exc), "ERROR")

    def stop_current_audio(self, *, reset_position: bool = False) -> None:
        position = 0.0 if reset_position else self._current_play_position()
        self._clear_playback_poll()
        if self.play_proc is not None and self.play_proc.poll() is None:
            self.play_proc.terminate()
            try:
                self.play_proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.play_proc.kill()
                self.play_proc.wait(timeout=1.0)
        self.play_proc = None
        self.play_started_at = 0.0
        self.play_offset = 0.0 if reset_position else min(self.current_audio_duration, position)
        if self.current_audio_path is None:
            self.playback_status_var.set("Playback: stopped")
        elif reset_position:
            self.playback_status_var.set("Playback: stopped")
        else:
            self.playback_status_var.set("Playback: paused")
        if self.current_audio_path is not None:
            self._set_audio_status(self.current_audio_path)
        else:
            self._set_audio_status(None)
        self._draw_timeline()

    def _seek_timeline(self, event: Any) -> None:
        if self.current_audio_path is None or self.current_audio_duration <= 0:
            return
        width = max(1, self.timeline_canvas.winfo_width())
        offset = self.current_audio_duration * min(1.0, max(0.0, event.x / width))
        was_playing = self.play_proc is not None and self.play_proc.poll() is None
        self.stop_current_audio(reset_position=False)
        self.play_offset = offset
        self.playback_status_var.set("Playback: paused")
        self._set_audio_status(self.current_audio_path)
        self._draw_timeline()
        if was_playing:
            try:
                self._start_playback(offset)
            except Exception as exc:  # noqa: BLE001
                self._log(str(exc), "ERROR")

    def _run_async(self, fn, *, success_message: str | None = None, refresh_after: bool = False) -> None:
        if self.busy:
            self._log("A TTS action is already running", "WARN")
            return
        self._set_busy(True)

        def worker() -> None:
            try:
                fn()
                if success_message:
                    self._log(success_message)
            except subprocess.CalledProcessError as exc:
                err = (exc.stderr or exc.stdout or str(exc)).strip()
                self._log(err or f"command failed: {exc}", "ERROR")
            except Exception as exc:  # noqa: BLE001
                self._log(str(exc), "ERROR")
            finally:
                self.after(0, lambda: self._set_busy(False))
                if refresh_after:
                    self.after(0, self.refresh_status)

        threading.Thread(target=worker, daemon=True).start()

    def refresh_status(self) -> None:
        self.sink_var.set(f"Sink: {current_sink_name()}")
        health = server_health()
        self.health_data = health
        if health:
            engines = ", ".join(health.get("available_engines") or [])
            loaded = [k for k, v in (health.get("loaded") or {}).items() if v]
            self.server_status_var.set(
                f"Server: running on {health.get('host')} pid={health.get('pid')} device={health.get('device')}"
            )
            self.server_detail_var.set(
                f"Available: {engines or '-'} | Loaded: {', '.join(loaded) or '-'}"
            )
        else:
            host, port, _ = tts_endpoint()
            self.server_status_var.set(f"Server: not running on {host}:{port}")
            self.server_detail_var.set("")

    def start_server(self) -> None:
        def run() -> None:
            proc = subprocess.run(
                [str(BIN_ROOT / "tts.sh")],
                check=False,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            if proc.returncode != 0:
                raise RuntimeError(out.strip() or "tts.sh failed")
            self._log((out.strip() or "tts server started").splitlines()[-1])

        self._run_async(run, refresh_after=True)

    def stop_server(self) -> None:
        def run() -> None:
            proc = subprocess.run(
                [str(BIN_ROOT / "stop_tts.sh")],
                check=False,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            if proc.returncode != 0:
                raise RuntimeError(out.strip() or "stop_tts.sh failed")
            self._log((out.strip() or "tts server stopped").splitlines()[-1])

        self._run_async(run, refresh_after=True)

    def _text_value(self) -> str:
        return self.text_box.get("1.0", "end").strip()

    def _base_args(self) -> list[str]:
        args: list[str] = []
        engine = self.engine_var.get().strip() or "auto"
        args.extend(["--engine", engine])
        lang = self.lang_var.get().strip() or "auto"
        args.extend(["--lang", lang])

        if self.no_speaker_var.get():
            args.extend(["--speaker", "none"])
        else:
            speaker = self.speaker_var.get().strip()
            if speaker:
                args.extend(["--speaker", speaker])

        if self.style_var.get().strip():
            args.extend(["--style", self.style_var.get().strip()])
        if self.prompt_var.get().strip():
            args.extend(["--prompt-text", self.prompt_var.get().strip()])
        if self.chunk_length_var.get().strip():
            args.extend(["--chunk-length", self.chunk_length_var.get().strip()])
        if self.piper_model_var.get().strip():
            args.extend(["--model", self.piper_model_var.get().strip()])
        if self.piper_speaker_id_var.get().strip():
            args.extend(["--speaker-id", self.piper_speaker_id_var.get().strip()])
        if self.piper_length_scale_var.get().strip():
            args.extend(["--length-scale", self.piper_length_scale_var.get().strip()])
        if self.piper_noise_scale_var.get().strip():
            args.extend(["--noise-scale", self.piper_noise_scale_var.get().strip()])
        if self.piper_noise_w_var.get().strip():
            args.extend(["--noise-w", self.piper_noise_w_var.get().strip()])
        if self.piper_sentence_silence_var.get().strip():
            args.extend(["--sentence-silence", self.piper_sentence_silence_var.get().strip()])
        return args

    def speak_selected_mode(self) -> None:
        self.speak(self.mode_var.get())

    def speak(self, mode: str) -> None:
        text = self._text_value()
        if not text:
            messagebox.showerror("TTS GUI", "Text is empty.")
            return

        if mode == "server":
            self._run_async(lambda: self._speak_server(text), success_message="Server synthesis completed", refresh_after=True)
        else:
            self._run_async(lambda: self._speak_oneshot(text), success_message="One-shot synthesis completed")

    def _speak_oneshot(self, text: str) -> None:
        fd, tmp_name = tempfile.mkstemp(prefix="tts_gui_", suffix=".wav", dir=TMP_ROOT)
        os.close(fd)
        tmp_path = Path(tmp_name)
        cmd = [str(BIN_ROOT / "tts_once.sh"), "-o", str(tmp_path), *self._base_args(), text]
        self._log("Running one-shot TTS")
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )
            if proc.returncode != 0:
                raise RuntimeError((proc.stderr or proc.stdout or "").strip() or "tts_once.sh failed")
            out = (proc.stdout or "").strip()
            if out:
                self._log(out)
            self.after(0, lambda path=tmp_path: self._load_and_play_audio(path))
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    def _speak_server(self, text: str) -> None:
        health = server_health()
        if not health:
            raise RuntimeError("TTS server is not running")

        fd, tmp_name = tempfile.mkstemp(prefix="tts_gui_", suffix=".wav", dir=TMP_ROOT)
        os.close(fd)
        tmp_path = Path(tmp_name)

        try:
            cmd = [str(BIN_ROOT / "call_tts.sh"), "-o", str(tmp_path), *self._base_args(), text]
            self._log("Running server TTS")
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )
            if proc.returncode != 0:
                raise RuntimeError((proc.stderr or proc.stdout or "").strip() or "call_tts.sh failed")
            out = (proc.stdout or "").strip()
            if out:
                self._log(out)
            self.after(0, lambda path=tmp_path: self._load_and_play_audio(path))
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    def _load_and_play_audio(self, path: Path) -> None:
        self._load_audio(path, temporary=True)
        try:
            self._start_playback(0.0)
        except Exception as exc:  # noqa: BLE001
            self.playback_status_var.set("Playback: ready")
            self._draw_timeline()
            self._log(str(exc), "ERROR")

    def _on_close(self) -> None:
        self._reset_audio()
        self.destroy()


def main() -> int:
    bootstrap_audio_env()
    app = TTSGui()
    app.mainloop()
    return 0
