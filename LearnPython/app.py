"""
ShopEase Executive Dashboard Server
Lightweight, zero-dependency Python HTTP Server with REST APIs
Provides real-time interactive exploration for Part 1 (NumPy) and Part 2 (Pandas).
"""

import http.server
import socketserver
import json
import os
import sys
import time
import urllib.parse
import numpy as np

PORT = 5000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DATA_DIR = os.path.join(BASE_DIR, "data")

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        # API: Part 1 Report
        if path == "/api/part1":
            report_file = os.path.join(REPORTS_DIR, "part1_report.json")
            if os.path.exists(report_file):
                self._send_json(report_file)
            else:
                self._send_error_json("Part 1 report not found. Please run part1_numpy_audit.py.")
            return

        # API: Part 2 Report
        elif path == "/api/part2":
            report_file = os.path.join(REPORTS_DIR, "part2_report.json")
            if os.path.exists(report_file):
                self._send_json(report_file)
            else:
                self._send_error_json("Part 2 report not found. Please run part2_pandas_shopease.py.")
            return

        # API: Live Benchmark Runner
        elif path == "/api/benchmark":
            size_param = query.get("size", ["1000000"])[0]
            try:
                size = int(size_param)
                size = max(10_000, min(5_000_000, size))
            except ValueError:
                size = 1_000_000

            # Execute Live Benchmark
            np.random.seed(int(time.time()) % 10000)
            raw_np = np.random.uniform(45_000, 160_000, size=size).astype(np.float64)
            raw_list = raw_np.tolist()

            # Python list
            t0 = time.perf_counter()
            _ = [s * 1.10 for s in raw_list]
            t1 = time.perf_counter()
            list_ms = (t1 - t0) * 1000

            # NumPy
            t2 = time.perf_counter()
            _ = raw_np * 1.10
            t3 = time.perf_counter()
            np_ms = (t3 - t2) * 1000

            speedup = list_ms / np_ms if np_ms > 0 else 100.0
            time_saved = ((list_ms - np_ms) / list_ms) * 100

            res = {
                "size": size,
                "list_time_ms": round(list_ms, 2),
                "np_time_ms": round(np_ms, 2),
                "speedup_factor": round(speedup, 1),
                "time_saved_pct": round(time_saved, 2)
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
            return

        # Serve index.html by default on root
        elif path == "/" or path == "/index.html":
            self.path = "/index.html"
            return super().do_GET()

        return super().do_GET()

    def _send_json(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content.encode("utf-8"))

    def _send_error_json(self, message):
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode("utf-8"))

def start_server():
    # Allow port reuse
    socketserver.TCPServer.allow_reuse_address = True
    server_address = ("", PORT)
    try:
        httpd = socketserver.TCPServer(server_address, DashboardHandler)
        print("=" * 70)
        print(f"ShopEase Analytics Dashboard Server running at:")
        print(f"   -> http://localhost:{PORT}")
        print("Press Ctrl+C to terminate.")
        print("=" * 70)
        httpd.serve_forever()
    except OSError as e:
        print(f"Port {PORT} already in use or error: {e}")
        # Try fallback port
        alt_port = 5050
        httpd = socketserver.TCPServer(("", alt_port), DashboardHandler)
        print(f"Started on fallback port -> http://localhost:{alt_port}")
        httpd.serve_forever()

if __name__ == "__main__":
    start_server()
