#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

def _now_ns() -> int:
    return time.monotonic_ns()

def _ns_to_ms(ns: int) -> float:
    return ns / 1_000_000.0

def _die(msg: str, code: int = 2) -> None:
    print(msg, file=sys.stderr, flush=True)
    raise SystemExit(code)

def _read_exact(req, n: int) -> bytes:
    data = req.rfile.read(n)
    if data is None or len(data) != n:
        return b""
    return data

class PiperHandler(BaseHTTPRequestHandler):
    server_version = "piper-http/1.0"

    def log_message(self, fmt, *args):
        # keep logs deterministic to stdout
        sys.stdout.write("%s - - [%s] %s\n" % (
            self.address_string(),
            self.log_date_time_string(),
            fmt % args
        ))
        sys.stdout.flush()

    def _send_json(self, code: int, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"ok": True})
            return
        if self.path.startswith("/"):
            self._send_json(404, {"error": "not found"})
            return

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/tts":
            self._send_json(404, {"error": "not found"})
            return

        qs = parse_qs(parsed.query)

        # model can be provided as query param `model=/path/to/model.onnx`
        model = None
        if "model" in qs and len(qs["model"]) > 0:
            model = qs["model"][0]

        # voice dir shortcut: `voice=en_US-ryan-high` -> canonical path
        voice = None
        if "voice" in qs and len(qs["voice"]) > 0:
            voice = qs["voice"][0]

        if model is None and voice is not None:
            model = f"/data/dev/bin/piper/voices/{voice}/{voice}.onnx"

        if model is None:
            self._send_json(400, {"error": "missing model (query param model=... or voice=...)"})
            return

        # read request body (text/plain or application/json)
        clen = self.headers.get("Content-Length")
        if clen is None:
            self._send_json(411, {"error": "missing Content-Length"})
            return
        try:
            n = int(clen)
        except ValueError:
            self._send_json(400, {"error": "invalid Content-Length"})
            return

        raw = _read_exact(self, n)
        if not raw:
            self._send_json(400, {"error": "empty body"})
            return

        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        text = None
        if ctype == "application/json":
            try:
                obj = json.loads(raw.decode("utf-8"))
            except Exception as e:
                self._send_json(400, {"error": f"invalid json: {e.__class__.__name__}: {e}"})
                return
            if not isinstance(obj, dict) or "text" not in obj:
                self._send_json(400, {"error": "json must be object with field 'text'"})
                return
            text = str(obj["text"])
        else:
            # default: treat body as utf-8 text
            try:
                text = raw.decode("utf-8")
            except Exception as e:
                self._send_json(400, {"error": f"body must be utf-8 text: {e.__class__.__name__}: {e}"})
                return

        text = text.strip()
        if not text:
            self._send_json(400, {"error": "text is empty after stripping"})
            return

        piper_bin = self.server.piper_bin  # type: ignore[attr-defined]
        if not piper_bin:
            piper_bin = "piper-tts"

        # Build command:
        # piper-tts --model MODEL --output_raw
        # (stdin: text, stdout: audio bytes)
        cmd = [piper_bin, "--model", model, "--output_raw"]

        # Timing points
        t0 = _now_ns()
        try:
            p = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
        except FileNotFoundError:
            self._send_json(500, {"error": f"piper binary not found: {piper_bin}"})
            return
        except Exception as e:
            self._send_json(500, {"error": f"failed to spawn piper: {e.__class__.__name__}: {e}"})
            return
        t_spawn = _now_ns()

        # Send text, close stdin to signal EOF
        try:
            assert p.stdin is not None
            p.stdin.write((text + "\n").encode("utf-8"))
            p.stdin.flush()
            p.stdin.close()
        except Exception as e:
            try:
                p.kill()
            except Exception:
                pass
            self._send_json(500, {"error": f"failed to write to piper stdin: {e.__class__.__name__}: {e}"})
            return

        # Prepare response headers: WAV container around raw PCM16 requires knowing sample rate/channels.
        # Piper raw output is PCM S16LE, but sample rate depends on voice config; without parsing the .onnx.json,
        # we cannot truthfully declare a WAV header. Therefore we stream raw PCM only.
        #
        # Client must know how to play PCM16LE (typically mono) at the model's sample rate.
        #
        # If you want WAV, you must provide the exact sample rate and channels from the model json.
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("X-Audio-Format", "s16le-raw")
        self.send_header("X-Model", model)
        self.end_headers()

        first_byte_ns = None
        total_out = 0
        err_chunks = []

        try:
            assert p.stdout is not None
            while True:
                chunk = p.stdout.read(4096)
                if not chunk:
                    break
                if first_byte_ns is None:
                    first_byte_ns = _now_ns()
                total_out += len(chunk)
                self.wfile.write(chunk)
        except BrokenPipeError:
            # client disconnected; still reap process
            pass
        except Exception as e:
            # capture stderr for diagnostics
            err_chunks.append(f"stream error: {e.__class__.__name__}: {e}")
        finally:
            # read stderr fully (non-blocking not guaranteed; but process should be at EOF soon)
            try:
                assert p.stderr is not None
                serr = p.stderr.read()
            except Exception:
                serr = b""

            rc = None
            try:
                rc = p.wait(timeout=10)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
                try:
                    rc = p.wait(timeout=5)
                except Exception:
                    rc = None

            t_end = _now_ns()

            # Deterministic timing log to stdout
            fb_ms = None if first_byte_ns is None else _ns_to_ms(first_byte_ns - t0)
            sys.stdout.write(
                "PIPER_REQ "
                + json.dumps({
                    "cmd": cmd,
                    "model": model,
                    "text_len": len(text),
                    "spawn_ms": _ns_to_ms(t_spawn - t0),
                    "first_byte_ms": fb_ms,
                    "total_ms": _ns_to_ms(t_end - t0),
                    "bytes_out": total_out,
                    "rc": rc,
                    "stderr_tail": (serr.decode("utf-8", errors="replace")[-400:] if serr else ""),
                    "notes": err_chunks,
                }, ensure_ascii=False)
                + "\n"
            )
            sys.stdout.flush()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9850)
    ap.add_argument("--piper-bin", default=os.environ.get("PIPER_BIN", "piper-tts"))
    args = ap.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), PiperHandler)
    httpd.piper_bin = args.piper_bin  # attach

    print(f"LISTEN http://{args.host}:{args.port}", flush=True)
    print("ENDPOINTS: GET /health ; POST /tts?voice=en_US-ryan-high OR /tts?model=/abs/path/model.onnx", flush=True)
    print("RESPONSE: raw PCM S16LE bytes (application/octet-stream), not WAV", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
