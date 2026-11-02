# Brauzer demosini lokal ishga tushirish (HF Space'dagi bilan bir xil sarlavhalar bilan).
# Repo ildizidan:  python web/serve.py   ->  http://127.0.0.1:8000/web/
#
# COOP/COEP sarlavhalari sahifani crossOriginIsolated qiladi, shunda ONNX Runtime WASM ko'p oqimda ishlaydi.
# Sarlavhalarsiz 12 soniyalik audio 188 s da, ular bilan 13 s da o'qildi (12 yadroli noutbuk, Edge).
import functools
import http.server
import sys


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        super().end_headers()


if __name__ == "__main__":
    ildiz = sys.argv[1] if len(sys.argv) > 1 else "."
    print("http://127.0.0.1:8000/web/")
    http.server.ThreadingHTTPServer(("127.0.0.1", 8000), functools.partial(Handler, directory=ildiz)).serve_forever()
