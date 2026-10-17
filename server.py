"""
AlexGPT web server. Uses only the Python standard library.

Run from the project root:
    python server.py
then open http://localhost:8000
"""

import json
import os
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import numpy as np

from alexgpt.chat import load_models, generate_stream

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(PROJECT_ROOT, "web")
PORT = 8000

STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
}

print("Loading models...")
MODELS = load_models(PROJECT_ROOT)
rng = np.random.default_rng()


class Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/api/models":
            model_list = []
            for name in MODELS:
                model, description = MODELS[name]
                model_list.append({
                    "name": name,
                    "description": description,
                    "context_size": model.context_size,
                    "vocab_size": len(model.chars),
                    "params": model.num_params(),
                    "layers": model.layer_sizes(),
                })
            self.send_json(model_list)
            return

        if self.path in STATIC_FILES:
            file_name, content_type = STATIC_FILES[self.path]
            with open(os.path.join(WEB_DIR, file_name), "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)
            return

        self.send_error(404)

    def do_POST(self):
        if self.path != "/api/generate":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length))

        model_name = request.get("model")
        if model_name not in MODELS:
            self.send_json({"error": "unknown model"}, status=400)
            return
        model = MODELS[model_name][0]

        prompt = request.get("prompt", "")
        max_chars = int(request.get("max_chars", 0))
        if max_chars <= 0:
            max_chars = 2000    # "Auto": the model stops when it writes <|endoftext|>, this is just a safety limit
        temperature = max(0.05, float(request.get("temperature", 0.8)))
        delay_seconds = max(0, int(request.get("delay_ms", 25))) / 1000

        # stream one "data: {...}" line per character (server-sent events format)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        try:
            for step in generate_stream(model, prompt, max_chars, temperature, rng):
                self.write_event(step)
                time.sleep(delay_seconds)
            self.write_event({"type": "done"})
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass    # the user pressed stop and closed the connection

    def write_event(self, data):
        self.wfile.write(("data: " + json.dumps(data) + "\n\n").encode("utf-8"))
        self.wfile.flush()

    def send_json(self, data, status=200):
        content = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format, *args):
        pass    # keep the terminal quiet


if __name__ == "__main__":
    server = ThreadingHTTPServer(("localhost", PORT), Handler)
    print(f"\nAlexGPT is running at http://localhost:{PORT}  (Ctrl+C to stop)\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
