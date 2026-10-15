"""Runtime hook: make ONNX Runtime's ``transformers`` helpers importable.

``onnxruntime/quantization/shape_inference.py`` imports
``onnxruntime.transformers.onnx_utils``, and that module does

    from fusion_utils import NumpyHelper
    from onnx_model import OnnxModel

Two *bare top-level* imports, which only resolve because that directory is
normally on ``sys.path``. PyInstaller analyses absolute imports against the real
``sys.path``, so it cannot resolve them, the modules never make it into the
bundle, and ``import onnxruntime.quantization`` fails at run time with
``No module named 'fusion_utils'`` - which silently removes the INT8 workloads
from the suite.

The packaging spec therefore ships that directory as data files; this hook puts
it on ``sys.path`` so the bare imports resolve. Fixing it here rather than
patching the model library keeps the workaround visible and reversible.
"""

import os
import sys

try:
    base = getattr(sys, "_MEIPASS", None) or os.path.dirname(sys.executable)
    candidate = os.path.join(base, "onnxruntime", "transformers")
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.append(candidate)
except Exception:  # pragma: no cover - a broken hook must not stop the app
    pass
