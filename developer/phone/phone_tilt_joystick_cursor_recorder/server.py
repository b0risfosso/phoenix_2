#!/usr/bin/env python3
"""
Phone Live Sync Movement Recorder - HTTP fallback edition

Run:
  python3 server.py

Open on computer:
  http://localhost:8765/dashboard.html

Open on phone on same Wi-Fi:
  http://YOUR_COMPUTER_LAN_IP:8765/phone.html

This version does not require WebSockets for the main connection path. The phone
uses normal HTTP POST requests and the dashboard polls the server. This makes the
Connect button much more reliable on local networks.
"""
from __future__ import annotations

import csv
import http.server
import json
import os
import socket
import socketserver
import time
import urllib.parse
from pathlib import Path
from typing import Dict, List

HOST = "0.0.0.0"
PORT = 8765
ROOT = Path(__file__).resolve().parent
RECORDINGS_DIR = ROOT / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)

recording_active = False
recording_name = "movement"
recording_started_at = None
current_recording: List[dict] = []
all_events: List[dict] = []
last_saved_files: Dict[str, str] = {}
phone_last_seen = 0.0
phone_connected_count = 0
sample_seq = 0


def local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def now_ms() -> int:
    return int(time.time() * 1000)


def sanitize_name(name: str) -> str:
    name = name or "movement"
    safe = "".join(ch for ch in name if ch.isalnum() or ch in "-_ ").strip()
    return safe or "movement"


def flatten_sample(s: dict) -> dict:
    dm = s.get("devicemotion") or {}
    do = s.get("deviceorientation") or {}
    a = dm.get("acceleration") or {}
    ag = dm.get("accelerationIncludingGravity") or {}
    r = dm.get("rotationRate") or {}
    return {
        "seq": s.get("seq", ""),
        "t": s.get("t", ""),
        "dt": s.get("dt", ""),
        "source": s.get("source", "phone"),
        "eventType": s.get("eventType", "sample"),
        "accelX": a.get("x", ""),
        "accelY": a.get("y", ""),
        "accelZ": a.get("z", ""),
        "accelGX": ag.get("x", ""),
        "accelGY": ag.get("y", ""),
        "accelGZ": ag.get("z", ""),
        "rotAlpha": r.get("alpha", ""),
        "rotBeta": r.get("beta", ""),
        "rotGamma": r.get("gamma", ""),
        "orientAlpha": do.get("alpha", ""),
        "orientBeta": do.get("beta", ""),
        "orientGamma": do.get("gamma", ""),
        "absolute": do.get("absolute", ""),
        "interval": dm.get("interval", ""),
        "label": s.get("label", ""),
    }


def save_recording() -> Dict[str, str]:
    global last_saved_files
    data = list(current_recording)
    safe = sanitize_name(recording_name).replace(" ", "_")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = f"{safe}_{stamp}"
    json_path = RECORDINGS_DIR / f"{base}.json"
    csv_path = RECORDINGS_DIR / f"{base}.csv"

    package = {
        "name": recording_name,
        "savedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sampleCount": len(data),
        "samples": data,
    }
    json_path.write_text(json.dumps(package, indent=2), encoding="utf-8")

    fields = [
        "seq", "t", "dt", "source", "eventType",
        "accelX", "accelY", "accelZ",
        "accelGX", "accelGY", "accelGZ",
        "rotAlpha", "rotBeta", "rotGamma",
        "orientAlpha", "orientBeta", "orientGamma",
        "absolute", "interval", "label",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in data:
            flat = flatten_sample(row)
            writer.writerow({k: flat.get(k, "") for k in fields})

    last_saved_files = {"json": f"/recordings/{json_path.name}", "csv": f"/recordings/{csv_path.name}"}
    return last_saved_files


def state(last_seq: int = 0) -> dict:
    phone_alive = (time.time() - phone_last_seen) < 8
    new_events = [e for e in all_events if e.get("seq", 0) > last_seq]
    if len(new_events) > 1000:
        new_events = new_events[-1000:]
    return {
        "ok": True,
        "serverTime": time.time(),
        "phoneAlive": phone_alive,
        "phoneLastSeenSecondsAgo": None if phone_last_seen == 0 else round(time.time() - phone_last_seen, 2),
        "phoneConnectCount": phone_connected_count,
        "recordingActive": recording_active,
        "recordingName": recording_name,
        "sampleCount": len(current_recording),
        "eventCount": len(all_events),
        "lastSeq": sample_seq,
        "events": new_events,
        "lastSavedFiles": last_saved_files,
    }


class Handler(http.server.SimpleHTTPRequestHandler):
    server_version = "PhoneMovementRecorderHTTP/2.0"

    def translate_path(self, path):
        parsed = urllib.parse.urlparse(path)
        clean = os.path.normpath(urllib.parse.unquote(parsed.path)).lstrip("/")
        if clean == "":
            clean = "dashboard.html"
        return str(ROOT / clean)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, obj, status=200):
        body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/ping":
            self.send_json({"ok": True, "serverTime": time.time(), "phoneUrl": f"http://{local_ip()}:{PORT}/phone.html"})
            return
        if parsed.path == "/api/state":
            qs = urllib.parse.parse_qs(parsed.query)
            try:
                last_seq = int(qs.get("lastSeq", ["0"])[0])
            except Exception:
                last_seq = 0
            self.send_json(state(last_seq))
            return
        super().do_GET()

    def do_POST(self):
        global recording_active, recording_name, recording_started_at, current_recording
        global all_events, last_saved_files, phone_last_seen, phone_connected_count, sample_seq
        parsed = urllib.parse.urlparse(self.path)
        msg = self.read_json()

        if parsed.path == "/api/phone_hello":
            phone_last_seen = time.time()
            phone_connected_count += 1
            self.send_json({"ok": True, "message": "phone connected", "serverTime": time.time(), "state": state(0)})
            return

        if parsed.path == "/api/heartbeat":
            phone_last_seen = time.time()
            self.send_json({"ok": True, "recordingActive": recording_active, "serverTime": time.time()})
            return

        if parsed.path == "/api/start":
            phone_last_seen = time.time()
            recording_name = sanitize_name(msg.get("name") or "movement")
            recording_active = True
            recording_started_at = time.time()
            current_recording = []
            all_events = []
            sample_seq = 0
            last_saved_files = {}
            self.send_json({"ok": True, "recordingActive": True, "name": recording_name})
            return

        if parsed.path == "/api/sample":
            phone_last_seen = time.time()
            sample = msg.get("sample") or msg
            sample_seq += 1
            sample["seq"] = sample_seq
            sample.setdefault("source", "phone")
            sample.setdefault("eventType", "sample")
            all_events.append(sample)
            if len(all_events) > 5000:
                del all_events[:-5000]
            if recording_active:
                current_recording.append(sample)
            self.send_json({"ok": True, "seq": sample_seq, "recordingActive": recording_active, "sampleCount": len(current_recording)})
            return

        if parsed.path == "/api/marker":
            phone_last_seen = time.time()
            sample_seq += 1
            marker = {"seq": sample_seq, "eventType": "marker", "source": "phone", "t": now_ms(), "label": msg.get("label", "marker")}
            all_events.append(marker)
            if recording_active:
                current_recording.append(marker)
            self.send_json({"ok": True, "seq": sample_seq, "marker": marker, "sampleCount": len(current_recording)})
            return

        if parsed.path == "/api/stop":
            phone_last_seen = time.time()
            recording_active = False
            files = save_recording()
            self.send_json({"ok": True, "recordingActive": False, "files": files, "sampleCount": len(current_recording)})
            return

        if parsed.path == "/api/save_now":
            files = save_recording()
            self.send_json({"ok": True, "files": files, "sampleCount": len(current_recording)})
            return

        self.send_json({"ok": False, "error": "unknown endpoint"}, status=404)

    def log_message(self, fmt, *args):
        print("[%s] %s" % (time.strftime("%H:%M:%S"), fmt % args))


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ip = local_ip()
    print("Phone Live Sync Movement Recorder - HTTP fallback edition")
    print("------------------------------------------------------")
    print(f"Computer dashboard: http://localhost:{PORT}/dashboard.html")
    print(f"Phone recorder:     http://{ip}:{PORT}/phone.html")
    print(f"Connection test:    http://{ip}:{PORT}/api/ping")
    print(f"Recordings folder:  {RECORDINGS_DIR}")
    print("Press Ctrl+C to stop.")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
