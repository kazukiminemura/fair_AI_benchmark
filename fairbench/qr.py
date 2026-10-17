"""Local QR encoding; render SVG without Pillow or network services."""

from __future__ import annotations

import qrcode


def svg(text: str) -> str:
    code = qrcode.QRCode(border=4, error_correction=qrcode.constants.ERROR_CORRECT_M)
    code.add_data(text)
    code.make(fit=True)
    matrix = code.get_matrix()
    size = len(matrix)
    path = "".join(f"M{x},{y}h1v1h-1z" for y, row in enumerate(matrix) for x, dark in enumerate(row) if dark)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" '
            'width="256" height="256" role="img" aria-label="スマホ測定画面へのQRコード" shape-rendering="crispEdges">'
            f'<rect width="{size}" height="{size}" fill="white"/><path d="{path}" fill="black"/></svg>')
