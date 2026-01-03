#!/usr/bin/env python3
import argparse
import http.client
import subprocess
import sys
import time
from urllib.parse import urlsplit

def _die(msg: str, code: int = 2) -> None:
    print(msg, file=sys.stderr, flush=True)
    raise SystemExit(code)

def _ns_to_ms(ns: int) -> float:
    return ns / 1_000_000.0

def _path_with_query(u) -> str:
    path = u.path if u.path else "/"
    if u.query:
        path += "?" + u.query
    return path

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", help="text to synthesize (UTF-8)")
    ap.add_argument("--url", default="http://127.0.0.1:9850/tts?voice=en_US-ryan-high")
    ap.add_argument("--rate", type=int, default=22050)
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--device", default=None, help="optional ALSA device name for aplay (-D), e.g. default, hw:CARD=PCH,DEV=0")
    ap.add_argument("--aplay", default="aplay")
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--timeout", type=int, default=30)
    args = ap.parse_args()

    u = urlsplit(args.url)
    if u.scheme != "http":
        _die(f"unsupported URL scheme: {u.scheme!r} (only http)")
    if not u.hostname:
        _die("URL missing hostname")

    host = u.hostname
    port = u.port if u.port is not None else 80
    path = _path_with_query(u)

    body = (args.text + "\n").encode("utf-8")
    headers = {
        "Content-Type": "text/plain; charset=utf-8",
        "Content-Length": str(len(body)),
        "Connection": "close",
    }

    aplay_cmd = [
        args.aplay,
        "-q",
        "-f", "S16_LE",
        "-c", str(args.channels),
        "-r", str(args.rate),
    ]
    if args.device is not None:
        aplay_cmd[1:1] = ["-D", args.device]  # insert after aplay

    t0 = time.monotonic_ns()

    try:
        aplay = subprocess.Popen(
            aplay_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        _die(f"aplay not found: {args.aplay}")
    except Exception as e:
        _die(f"failed to start aplay: {e.__class__.__name__}: {e}")

    t_aplay_spawn = time.monotonic_ns()

    conn = http.client.HTTPConnection(host, port, timeout=args.timeout)
    t_conn_create = time.monotonic_ns()

    first_byte_ns = None
    bytes_in = 0

    resp_status = None
    resp_reason = None

    try:
        conn.request("POST", path, body=body, headers=headers)
        t_req_sent = time.monotonic_ns()

        resp = conn.getresponse()
        resp_status = resp.status
        resp_reason = resp.reason
        t_resp_hdr = time.monotonic_ns()

        if resp_status != 200:
            err_body = resp.read()
            try:
                err_text = err_body.decode("utf-8", errors="replace")
            except Exception:
                err_text = repr(err_body[:400])
            _die(f"http {resp_status} {resp_reason}: {err_text}")

        if aplay.stdin is None:
            _die("aplay stdin is not available")

        while True:
            chunk = resp.read(args.chunk)
            if not chunk:
                break
            if first_byte_ns is None:
                first_byte_ns = time.monotonic_ns()
            bytes_in += len(chunk)
            try:
                aplay.stdin.write(chunk)
            except BrokenPipeError:
                _die("aplay closed stdin (broken pipe)")
    finally:
        try:
            conn.close()
        except Exception:
            pass
        try:
            if aplay.stdin is not None:
                aplay.stdin.close()
                aplay.stdin = None  # critical: prevent communicate()/flush on a closed file
        except Exception:
            pass

    # Reap aplay without communicate() to avoid stdin flush behavior
    try:
        rc = aplay.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        try:
            aplay.kill()
        except Exception:
            pass
        rc = aplay.wait()

    try:
        aplay_err = b"" if aplay.stderr is None else aplay.stderr.read()
    except Exception:
        aplay_err = b""

    t_end = time.monotonic_ns()

    if rc != 0:
        _die(f"aplay failed (rc={rc}): {aplay_err.decode(errors='replace')}")

    fb_ms = None if first_byte_ns is None else _ns_to_ms(first_byte_ns - t0)

    print(
        f"spawn_ms={_ns_to_ms(t_aplay_spawn - t0):.3f} "
        f"first_audio_byte_ms={(f'{fb_ms:.3f}' if fb_ms is not None else 'null')} "
        f"bytes_in={bytes_in} "
        f"http_status={resp_status} "
        f"total_ms={_ns_to_ms(t_end - t0):.3f}",
        flush=True,
    )

if __name__ == "__main__":
    main()
