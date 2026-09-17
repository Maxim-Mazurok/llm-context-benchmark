from __future__ import annotations

import json
import secrets
import shutil
import threading
import webbrowser
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def _unique_trash_destination(run_dir: Path, trash_dir: Path) -> Path:
    destination = trash_dir / f"llm-context-benchmark-{run_dir.name}"
    if not destination.exists():
        return destination
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return trash_dir / f"llm-context-benchmark-{run_dir.name}-{stamp}"


def move_run_to_trash(
    run_dir: Path,
    trash_dir: Path | None = None,
    *,
    runs_root: Path | None = None,
) -> Path:
    """Move one validated benchmark run to Trash and return its destination."""
    run_dir = run_dir.expanduser().resolve()
    project_root = Path(__file__).resolve().parents[2]
    allowed_root = (runs_root or project_root / "runs").expanduser().resolve()
    if run_dir.parent != allowed_root:
        raise ValueError(f"Run must be directly inside {allowed_root}")
    required = (run_dir / "summary.json", run_dir / "config.json")
    if not run_dir.is_dir() or not all(path.is_file() for path in required):
        raise ValueError("Directory does not look like a completed benchmark run")
    destination_root = (trash_dir or Path.home() / ".Trash").expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = _unique_trash_destination(run_dir, destination_root)
    shutil.move(str(run_dir), str(destination))
    return destination


def serve_run_dashboard(
    run_dir: Path,
    *,
    port: int = 0,
    open_browser: bool = False,
) -> int:
    run_dir = run_dir.expanduser().resolve()
    if not (run_dir / "dashboard.html").is_file():
        raise ValueError(f"No dashboard.html found in {run_dir}")
    token = secrets.token_urlsafe(24)

    class DashboardHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, directory=str(run_dir), **kwargs)

        def log_message(self, format: str, *args: object) -> None:
            return None

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            supplied = parse_qs(parsed.query).get("token", [""])[0]
            if parsed.path != "/api/delete-run" or not secrets.compare_digest(
                supplied, token
            ):
                self.send_error(403)
                return
            try:
                destination = move_run_to_trash(run_dir)
            except (OSError, ValueError) as exc:
                payload = json.dumps({"ok": False, "error": str(exc)}).encode()
                self.send_response(500)
            else:
                payload = json.dumps({"ok": True, "movedTo": str(destination)}).encode()
                self.send_response(200)
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", port), DashboardHandler)
    actual_port = server.server_address[1]
    url = f"http://127.0.0.1:{actual_port}/dashboard.html?token={token}"
    print(f"Dashboard: {url}", flush=True)
    print("Delete moves this run to macOS Trash. Press Ctrl-C to close.", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
