# /// script
# requires-python = ">=3.11"
# dependencies = ["ziglang>=0.13"]
# ///
"""Serve the lantern simulator on http://localhost:8800.

The firmware's programs (src/lantern.h, src/params.h) are compiled to WebAssembly with zig
whenever they change, and the open page reloads them. The page can also write its settings back
into src/params.h and flash the ESP32 with PlatformIO.

    uv run sim/serve.py
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SIM = Path(__file__).resolve().parent
ROOT = SIM.parent
WASM = SIM / "build" / "lantern.wasm"
PARAMS = ROOT / "src" / "params.h"
build_lock = threading.Lock()
build_error = ""


def sources() -> list[Path]:
    return [*(ROOT / "src").glob("*.h"), *SIM.glob("*.cpp"), *SIM.glob("*.h")]


def stamp() -> str:
    return str(max(p.stat().st_mtime_ns for p in sources()))


def build() -> str:
    """Rebuild the wasm if a source is newer; returns the compiler's errors, or ""."""
    global build_error
    with build_lock:
        newest = max(p.stat().st_mtime_ns for p in sources())
        if WASM.exists() and WASM.stat().st_mtime_ns >= newest and not build_error:
            return ""
        WASM.parent.mkdir(exist_ok=True)
        cmd = [sys.executable, "-m", "ziglang", "c++", "-target", "wasm32-wasi", "-mexec-model=reactor",
               "-O2", "-s", "-std=c++17", "-fno-exceptions", "-rdynamic", "-Wl,-z,stack-size=1048576",
               "-o", str(WASM), str(SIM / "sim.cpp")]
        result = subprocess.run(cmd, capture_output=True, text=True)
        build_error = result.stderr if result.returncode else ""
        if not build_error:
            WASM.touch()
        print("built" if not build_error else build_error, flush=True)
        return build_error


def save_params(values: dict[str, float]) -> list[str]:
    """Rewrites the boot values in params.h; returns the names it didn't find."""
    text = PARAMS.read_text()
    missing = []
    for name, value in values.items():
        pattern = re.compile(rf"X\((int|float), {re.escape(name)}, ([^,]+),")
        m = pattern.search(text)
        if not m:
            missing.append(name)
            continue
        new = str(round(value)) if m.group(1) == "int" else f"{value:.6g}"
        text = text[: m.start(2)] + new + text[m.end(2):]
    PARAMS.write_text(realign(text))
    return missing


def realign(text: str) -> str:
    """Keeps the macro's line-continuation backslashes in one column."""
    lines = text.split("\n")
    rows = [i for i, line in enumerate(lines) if line.rstrip().endswith("\\")]
    width = max(len(lines[i].rstrip()[:-1].rstrip()) for i in rows) + 1
    for i in rows:
        lines[i] = lines[i].rstrip()[:-1].rstrip().ljust(width) + "\\"
    return "\n".join(lines)


def pio() -> str:
    found = shutil.which("pio") or Path.home() / ".platformio" / "penv" / "bin" / "pio"
    return str(found)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SIM), **kwargs)

    def log_message(self, format, *args):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def reply(self, status: int, body: str | bytes, kind: str = "text/plain; charset=utf-8"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/stamp":
            return self.reply(HTTPStatus.OK, stamp())
        if path == "/lantern.wasm":
            error = build()
            if error:
                return self.reply(HTTPStatus.INTERNAL_SERVER_ERROR, error)
            return self.reply(HTTPStatus.OK, WASM.read_bytes(), "application/wasm")
        if path == "/params.h":
            return self.reply(HTTPStatus.OK, PARAMS.read_text())
        super().do_GET()

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path == "/params":
            missing = save_params(json.loads(body))
            return self.reply(HTTPStatus.OK, json.dumps({"missing": missing}), "application/json")
        if self.path == "/flash":
            result = subprocess.run([pio(), "run", "-t", "upload"], cwd=ROOT, capture_output=True, text=True)
            log = result.stdout[-4000:] + result.stderr[-4000:]
            return self.reply(HTTPStatus.OK if result.returncode == 0 else HTTPStatus.INTERNAL_SERVER_ERROR, log)
        self.reply(HTTPStatus.NOT_FOUND, "")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8800)
    args = parser.parse_args()
    error = build()
    if error:
        print("the first build failed; fix the error and reload the page", flush=True)
    print(f"lantern simulator on http://localhost:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
