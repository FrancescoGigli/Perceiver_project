"""Local HTTP server con cache-busting automatico via ?v=<timestamp>.

Doppio click su start.bat (Windows) o `python serve.py` da terminale.
"""
import http.server, re, socket, sys, threading, time, webbrowser
from pathlib import Path

PORT_START, PORT_END = 4242, 4252
ROOT = Path(__file__).parent.resolve()


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def end_headers(self):
        # disable caching aggressively during dev
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            try:
                content = (ROOT / 'index.html').read_text(encoding='utf-8')
                v = str(int(time.time()))

                def add_v(m):
                    url = m.group(1)
                    if '//' in url:           # external url, skip
                        return m.group(0)
                    sep = '&' if '?' in url else '?'
                    return m.group(0).replace(url, f"{url}{sep}v={v}")

                content = re.sub(r'<script\s+src="([^"]+)"', add_v, content)
                content = re.sub(r'<link\s+rel="stylesheet"\s+href="([^"]+)"', add_v, content)
                data = content.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            except Exception as e:
                sys.stderr.write(f"injection failed: {e}\n")
        super().do_GET()


def find_port():
    for p in range(PORT_START, PORT_END + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", p))
                return p
        except OSError:
            continue
    return None


if __name__ == "__main__":
    port = find_port()
    if port is None:
        print("ERRORE: nessuna porta libera tra 4242 e 4252.")
        sys.exit(1)
    url = f"http://localhost:{port}/"
    print(f"\n  Perceiver Exam Trainer")
    print(f"  URL: {url}")
    print(f"  Premi Ctrl+C per fermare.\n")
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    http.server.ThreadingHTTPServer.allow_reuse_address = True
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Server fermato.")
