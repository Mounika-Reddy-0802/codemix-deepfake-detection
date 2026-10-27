"""Open a free HTTPS tunnel to the demo server, so a phone can be the receiver handset.

A phone's browser gives a page the microphone and WebRTC only over https (localhost
excepted), so the receiver page on the mentor's phone must reach the laptop through a
tunnel. This starts one, finds its public address, writes it to ``tunnel_url.txt``
beside the demo calls (the scene-3 page reads it through ``/api/tunnel`` and shows it
as a QR code), and keeps the tunnel open until Ctrl+C.

Two free tunnels work:

- ``cloudflared tunnel --url http://localhost:8000`` -- no account (the default);
- ``ngrok http 8000`` -- a free account with its authtoken already configured.

Run (in its own terminal, after the server is up)::

    python scripts/start_tunnel.py              # cloudflared
    python scripts/start_tunnel.py --ngrok      # ngrok

``cloudflared`` is found on PATH, in ``CLOUDFLARED`` or at ``C:\\dfdata\\tools\\cloudflared.exe``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

URL_PATTERN = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def url_file() -> Path:
    if os.environ.get("TUNNEL_URL_FILE"):
        return Path(os.environ["TUNNEL_URL_FILE"])
    clips = Path(os.environ.get("DEMO_CLIPS_DIR", r"C:\dfdata\demo_calls_v2"))
    return clips.parent / "tunnel_url.txt"


def find(name: str, env: str, fallback: str) -> str | None:
    for candidate in (os.environ.get(env), shutil.which(name), fallback):
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def find_url(line: str) -> str | None:
    """The public address in one line of cloudflared's log, if it is there."""
    match = URL_PATTERN.search(line)
    return match.group(0) if match else None


def cloudflared(port: int) -> tuple[subprocess.Popen, str]:
    exe = find("cloudflared", "CLOUDFLARED", r"C:\dfdata\tools\cloudflared.exe")
    if exe is None:
        sys.exit("cloudflared not found: install it, or set CLOUDFLARED, or run with --ngrok")
    proc = subprocess.Popen(
        [exe, "tunnel", "--no-autoupdate", "--url", f"http://localhost:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    deadline = time.time() + 60
    for line in proc.stdout:  # cloudflared logs the address once the tunnel is up
        url = find_url(line)
        if url:
            return proc, url
        if time.time() > deadline:
            break
    proc.terminate()
    sys.exit("cloudflared started but printed no trycloudflare.com address within 60 s")


def ngrok(port: int) -> tuple[subprocess.Popen, str]:
    exe = find("ngrok", "NGROK", r"C:\dfdata\tools\ngrok.exe")
    if exe is None:
        sys.exit("ngrok not found: install it and run `ngrok config add-authtoken <token>` once")
    proc = subprocess.Popen(
        [exe, "http", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=2) as r:
                tunnels = json.load(r)["tunnels"]
            https = [t["public_url"] for t in tunnels if t["public_url"].startswith("https://")]
            if https:
                return proc, https[0]
        except OSError:
            pass
        time.sleep(1)
    proc.terminate()
    sys.exit("ngrok started but its local API reported no https tunnel within 30 s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Open an https tunnel for the phone demo")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ngrok", action="store_true", help="use ngrok instead of cloudflared")
    args = parser.parse_args()

    proc, url = (ngrok if args.ngrok else cloudflared)(args.port)
    out = url_file()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(url + "\n", encoding="utf-8")
    print(f"tunnel up: {url}")
    print(f"  on the phone (receiver): {url}/phone/receiver")
    print(f"  on the laptop (caller):  http://localhost:{args.port}/live")
    print(f"  address written to {out}; Ctrl+C closes the tunnel")
    try:
        proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        out.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
