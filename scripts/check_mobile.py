"""Check mobile-only routing without exposing the host benchmark APIs."""
from __future__ import annotations

import json
import re
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from unittest.mock import patch

from fairbench.cli import build_parser
from fairbench.qr import svg
from fairbench.server import STATE, Handler, MobileHandler, mobile_urls


def check_qr() -> None:
    url = "http://192.168.0.214:65535/mobile"
    image = svg(url)
    assert '<svg ' in image and 'fill="white"' in image
    # Optional decoder validates the actual SVG modules, independently of the encoder.
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("QR SVG generated; independent decoder not installed.")
        return
    size = int(re.search(r'viewBox="0 0 (\d+)', image).group(1))
    pixels = np.full((size, size), 255, dtype=np.uint8)
    for x, y in re.findall(r'M(\d+),(\d+)h1v1h-1z', image):
        pixels[int(y), int(x)] = 0
    decoded, _, _ = cv2.QRCodeDetector().detectAndDecode(pixels.repeat(8, axis=0).repeat(8, axis=1))
    assert decoded == url, decoded
    print("QR SVG decoded to the exact phone URL (including port and path).")


def main() -> None:
    check_qr()
    assert build_parser().parse_args(["serve", "--mobile", "--port", "8768"]).mobile
    assert not build_parser().parse_args(["serve"]).mobile
    with patch("fairbench.server.system.IS_WINDOWS", True), patch("fairbench.server.system._powershell_json", return_value=[
        "192.168.1.10", "169.254.1.1", "127.0.0.1"]):
        assert mobile_urls(1234) == ["http://192.168.1.10:1234/mobile"]
    server = ThreadingHTTPServer(("127.0.0.1", 0), MobileHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    before = STATE.snapshot()
    try:
        for path in ("/", "/mobile", "/mobile/", "/mobile.html", "/mobile.js", "/mobile-bench.js", "/mobile.css"):
            connection = HTTPConnection("127.0.0.1", server.server_port)
            connection.request("GET", path)
            response = connection.getresponse()
            assert response.status == 200, path
            assert response.read(), path
            connection.close()
        for method, path in (("GET", "/api/state"), ("GET", "/api/catalog"), ("GET", "/api/results"),
                             ("GET", "/api/result/private.json"), ("GET", "/../README.md"),
                             ("POST", "/api/run"), ("POST", "/api/delete"), ("POST", "/api/stop")):
            connection = HTTPConnection("127.0.0.1", server.server_port)
            connection.request(method, path, body="{}", headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            assert response.status == (403 if method == "POST" else 404), (method, path)
            assert "error" in json.loads(response.read())
            connection.close()
        assert STATE.snapshot() == before
        connection = HTTPConnection("127.0.0.1", server.server_port)
        with patch("fairbench.server.mobile_urls", return_value=["http://192.168.0.214:8768/mobile"]):
            connection.request("GET", "/connect")
            response = connection.getresponse()
            page = response.read().decode("utf-8")
        assert response.status == 200 and '<svg ' in page and '<!-- QR_CODES -->' not in page
        assert 'http://192.168.0.214:8768/mobile' in page
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    dashboard = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    dashboard_thread = threading.Thread(target=dashboard.serve_forever, daemon=True)
    dashboard_thread.start()
    try:
        for _ in range(2):
            connection = HTTPConnection("127.0.0.1", dashboard.server_port)
            connection.request("GET", "/connect")
            response = connection.getresponse()
            assert response.status == 200 and b'<form ' in response.read()
            connection.close()
            assert not getattr(dashboard, "mobile_server", None), "GET unexpectedly opened a LAN listener"
        connection = HTTPConnection("127.0.0.1", dashboard.server_port)
        connection.request("POST", "/api/mobile/start", headers={"Origin": "http://untrusted.invalid"})
        response = connection.getresponse()
        assert response.status == 403
        response.read()
        connection.close()
        assert not getattr(dashboard, "mobile_server", None)
        connection = HTTPConnection("127.0.0.1", dashboard.server_port)
        with patch("fairbench.server.mobile_urls", return_value=["http://192.168.1.10:1234/mobile"]):
            connection.request("POST", "/api/mobile/start")
            response = connection.getresponse()
            assert response.status == 200 and b'<svg ' in response.read()
        connection.close()
        mobile = dashboard.mobile_server
        assert mobile.server_address[0] == "0.0.0.0"
        connection = HTTPConnection("127.0.0.1", mobile.server_port)
        connection.request("GET", "/api/results")
        response = connection.getresponse()
        assert response.status == 404
        response.read()
        connection.close()
        connection = HTTPConnection("127.0.0.1", dashboard.server_port)
        connection.request("POST", "/api/mobile/stop")
        response = connection.getresponse()
        assert response.status == 200
        response.read()
        assert dashboard.mobile_server is None
        connection.close()
    finally:
        dashboard.shutdown()
        dashboard.server_close()
        dashboard_thread.join(timeout=2)
        if getattr(dashboard, "mobile_server", None):
            dashboard.mobile_server.shutdown()
            dashboard.mobile_server.server_close()
    print("Mobile assets, CLI flag, host API isolation and path restrictions passed.")


if __name__ == "__main__":
    main()
