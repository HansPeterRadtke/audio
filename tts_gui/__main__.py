from __future__ import annotations

import os


def _sanitize_tts_env() -> None:
    for key in (
        "TTS_PRELOAD",
        "TTS_PIPER_MODEL",
        "TTS_DEVICE",
        "TTS_HOST",
        "TTS_SECRET_KEY",
    ):
        os.environ.pop(key, None)


if __name__ == "__main__":
    _sanitize_tts_env()
    from .app import main

    raise SystemExit(main())
