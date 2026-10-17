"""Deterministic ONNX graph library.

Every workload in fairbench is an ONNX graph built here from a fixed seed, with
no downloads and no hidden randomness. That is what makes an Intel laptop and
an AMD laptop *actually* run the same work:

* weights are drawn from ``np.random.default_rng(<fixed seed>)`` (PCG64, stable
  across platforms), so two machines produce byte-identical files;
* input tensors are generated from the graph's own input declarations with a
  fixed seed;
* the SHA-256 of the serialised graph is written into every result file.

Sizes are chosen so that every workload is meaningfully long on both a low-end
laptop CPU and a datacentre-class GPU, without needing a model download.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# The one opset used by every graph. 17 gives us LayerNormalization while still
# being supported by ONNX Runtime, OpenVINO (CPU/GPU/NPU) and Vitis AI.
OPSET = 17
SEED = 1337

# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
FLOAT = TensorProto.FLOAT


def _vi(name: str, shape: list) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, FLOAT, shape)


def _w(name: str, shape: tuple[int, ...], scale: float, seed: int) -> onnx.TensorProto:
    """He-scaled deterministic weight initialiser."""
    rng = np.random.default_rng(seed)
    arr = (rng.standard_normal(shape) * scale).astype(np.float32)
    return numpy_helper.from_array(arr, name)


def _bias(name: str, size: int, seed: int) -> onnx.TensorProto:
    rng = np.random.default_rng(seed)
    arr = (rng.standard_normal(size) * 0.02).astype(np.float32)
    return numpy_helper.from_array(arr, name)


def _const_1d(name: str, values: list[float]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=np.float32), name)


def _seed_from(text: str) -> int:
    """Stable seed derived from a name.

    ``hash()`` is randomised per process, so it can never be used here: two
    machines must produce byte-identical weights.
    """
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big") % 100003


def _gelu_tanh(node_in: str, node_out: str, prefix: str) -> tuple[list[onnx.NodeProto], list[onnx.TensorProto]]:
    """GELU via the tanh approximation - supported identically everywhere."""
    inits = [
        _const_1d(prefix + "_gelu_c", [0.7978845608028654]),
        _const_1d(prefix + "_gelu_c2", [0.044715]),
        _const_1d(prefix + "_gelu_half", [0.5]),
        _const_1d(prefix + "_gelu_one", [1.0]),
    ]
    c = prefix + "_gelu_c"
    c2 = prefix + "_gelu_c2"
    half = prefix + "_gelu_half"
    one = prefix + "_gelu_one"
    nodes = [
        helper.make_node("Mul", [node_in, node_in], [prefix + "_g_x2"], name=prefix + "_g_mul1"),
        helper.make_node("Mul", [prefix + "_g_x2", node_in], [prefix + "_g_x3"], name=prefix + "_g_mul2"),
        helper.make_node("Mul", [prefix + "_g_x3", c2], [prefix + "_g_cx3"], name=prefix + "_g_mul3"),
        helper.make_node("Add", [prefix + "_g_cx3", node_in], [prefix + "_g_inner"], name=prefix + "_g_add1"),
        helper.make_node("Mul", [prefix + "_g_inner", c], [prefix + "_g_inner2"], name=prefix + "_g_mul4"),
        helper.make_node("Tanh", [prefix + "_g_inner2"], [prefix + "_g_tanh"], name=prefix + "_g_tanh"),
        helper.make_node("Add", [prefix + "_g_tanh", one], [prefix + "_g_plus1"], name=prefix + "_g_add2"),
        helper.make_node("Mul", [prefix + "_g_plus1", half], [prefix + "_g_half"], name=prefix + "_g_mul5"),
        helper.make_node("Mul", [node_in, prefix + "_g_half"], [node_out], name=prefix + "_g_mul6"),
    ]
    return nodes, inits


def _linear(
    x: str,
    w: str,
    b: str | None,
    out: str,
    prefix: str,
) -> list[onnx.NodeProto]:
    nodes = [helper.make_node("MatMul", [x, w], [prefix + "_mm"], name=prefix + "_mm")]
    src = prefix + "_mm"
    if b is not None:
        nodes.append(helper.make_node("Add", [src, b], [prefix + "_add"], name=prefix + "_add"))
        src = prefix + "_add"
    nodes.append(helper.make_node("Identity", [src], [out], name=prefix + "_id"))
    return nodes


@dataclass
class BuiltModel:
    """An ONNX graph plus the accounting needed to compute efficiency."""

    key: str
    description: str
    model: onnx.ModelProto
    flops: int
    unit: str  # what one graph execution produces: "sample" | "token" | "image"
    precision: str = "fp32"

    def serialise(self) -> bytes:
        return self.model.SerializeToString(deterministic=True)

    def materialise(self) -> Path:
        """Write the graph to the local cache, once, and return its path.

        Some runtimes (OpenVINO, Vitis AI, QNN) only accept a file path. The
        file name carries the graph hash, so a stale cache can never be served.
        """
        cache = Path(".cache") / "models"
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / f"{self.key}-{self.sha256[:12]}.onnx"
        if not path.exists():
            path.write_bytes(self.serialise())
        return path

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.serialise()).hexdigest()

    @property
    def size_bytes(self) -> int:
        return len(self.serialise())

    def input_names(self) -> list[str]:
        return [i.name for i in self.model.graph.input]

    def make_inputs(self, seed: int = SEED) -> dict[str, np.ndarray]:
        """Deterministic inputs derived from the graph's own input declarations."""
        rng = np.random.default_rng(seed)
        feed: dict[str, np.ndarray] = {}
        for value in self.model.graph.input:
            shape = [d.dim_value for d in value.type.tensor_type.shape.dim]
            feed[value.name] = rng.standard_normal(shape).astype(np.float32)
        return feed


# --------------------------------------------------------------------------
# graph builders
# --------------------------------------------------------------------------
def build_dense_gemm() -> BuiltModel:
    """Wide MLP / GEMM stack - pure compute throughput.

    Two large matrix multiplies with a GELU in between. This is the closest
    single-graph proxy for the fully-connected layers that dominate dense
    inference, and it stresses the same units an NPU is built around.
    """
    M, K, H, N = 256, 2048, 3072, 2048
    x = _vi("x", [M, K])
    y = _vi("y", [M, N])
    inits = [
        _w("w1", (K, H), 1.0 / math.sqrt(K), 1001),
        _bias("b1", H, 1002),
        _w("w2", (H, N), 1.0 / math.sqrt(H), 1003),
        _bias("b2", N, 1004),
    ]
    nodes: list[onnx.NodeProto] = []
    nodes += _linear("x", "w1", "b1", "h_pre", "fc1")
    g_nodes, g_inits = _gelu_tanh("h_pre", "h", "fc1")
    nodes += g_nodes
    inits += g_inits
    nodes += _linear("h", "w2", "b2", "y", "fc2")
    graph = helper.make_graph(nodes, "dense_gemm", [x], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 8
    flops = 2 * M * K * H + 2 * M * H * N
    return BuiltModel("dense_gemm", "2-layer MLP (GEMM bound)", model, flops, "sample")


def _conv_block(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    src: str,
    out: str,
    prefix: str,
    c_in: int,
    c_out: int,
    ksize: int,
    stride: int,
    groups: int = 1,
    activate: bool = True,
) -> str:
    pad = ksize // 2
    inits.append(_w(prefix + "_w", (c_out, c_in // groups, ksize, ksize), 1.0 / math.sqrt(c_in * ksize * ksize), _seed_from(prefix + "_w")))
    inits.append(_bias(prefix + "_b", c_out, _seed_from(prefix + "_b")))
    nodes.append(
        helper.make_node(
            "Conv",
            [src, prefix + "_w", prefix + "_b"],
            [prefix + "_conv"],
            name=prefix + "_conv",
            kernel_shape=[ksize, ksize],
            strides=[stride, stride],
            pads=[pad, pad, pad, pad],
            group=groups,
        )
    )
    last = prefix + "_conv"
    if activate:
        nodes.append(helper.make_node("Relu", [last], [prefix + "_relu"], name=prefix + "_relu"))
        last = prefix + "_relu"
    nodes.append(helper.make_node("Identity", [last], [out], name=prefix + "_id"))
    return out


def build_conv_vision() -> BuiltModel:
    """Convolutional feature extractor - the classic vision backbone."""
    N, C, SZ = 1, 3, 224
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    flops = 0
    # stem
    _conv_block(nodes, inits, "x", "s1", "stem", C, 32, 3, 2)
    flops += 2 * 32 * 112 * 112 * C * 9
    _conv_block(nodes, inits, "s1", "s2", "down1", 32, 64, 3, 2)
    flops += 2 * 64 * 56 * 56 * 32 * 9
    prev = "s2"
    for i in range(6):
        nxt = f"b{i}"
        _conv_block(nodes, inits, prev, nxt, f"blk{i}", 64, 64, 3, 1)
        flops += 2 * 64 * 56 * 56 * 64 * 9
        prev = nxt
    nodes.append(helper.make_node("GlobalAveragePool", [prev], ["gap"], name="gap"))
    nodes.append(helper.make_node("Flatten", ["gap"], ["flat"], name="flat", axis=1))
    inits.append(_w("fc_w", (64, 1000), 1.0 / math.sqrt(64), 2020))
    inits.append(_bias("fc_b", 1000, 2021))
    nodes.append(helper.make_node("MatMul", ["flat", "fc_w"], ["logits_pre"], name="fc_mm"))
    nodes.append(helper.make_node("Add", ["logits_pre", "fc_b"], ["y"], name="fc_add"))
    flops += 2 * 64 * 1000

    x = _vi("x", [N, C, SZ, SZ])
    y = _vi("y", [N, 1000])
    graph = helper.make_graph(nodes, "conv_vision", [x], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 8
    return BuiltModel("conv_vision", "CNN backbone (conv bound)", model, flops, "image")


def build_depthwise_mobile() -> BuiltModel:
    """MobileNet-style depthwise separable convolutions.

    This is the shape of network NPUs are actually specified against, so it is
    the fairest way to compare an Intel AI Boost and an AMD XDNA part.
    """
    N, C, SZ = 1, 3, 224
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    flops = 0

    _conv_block(nodes, inits, "x", "s0", "mstem", C, 32, 3, 2)
    flops += 2 * 32 * 112 * 112 * C * 9

    stages = [(32, 64, 1), (64, 128, 2), (128, 256, 2)]
    prev = "s0"
    for si, (cin, cout, stride) in enumerate(stages):
        sp = 112 if si == 0 else (56 if si == 1 else 28)
        out_sp = sp // stride
        dw = f"dw{si}"
        pw = f"pw{si}"
        # depthwise
        _conv_block(nodes, inits, prev, dw, f"dw{si}", cin, cin, 3, stride, groups=cin)
        flops += 2 * cin * out_sp * out_sp * 9
        # pointwise
        _conv_block(nodes, inits, dw, pw, f"pw{si}", cin, cout, 1, 1)
        flops += 2 * cout * out_sp * out_sp * cin
        prev = pw

    nodes.append(helper.make_node("GlobalAveragePool", [prev], ["gap"], name="gap"))
    nodes.append(helper.make_node("Flatten", ["gap"], ["flat"], name="flat", axis=1))
    inits.append(_w("head_w", (256, 1000), 1.0 / math.sqrt(256), 3030))
    inits.append(_bias("head_b", 1000, 3031))
    nodes.append(helper.make_node("MatMul", ["flat", "head_w"], ["head_mm"], name="head_mm"))
    nodes.append(helper.make_node("Add", ["head_mm", "head_b"], ["y"], name="head_add"))
    flops += 2 * 256 * 1000

    x = _vi("x", [N, C, SZ, SZ])
    y = _vi("y", [N, 1000])
    graph = helper.make_graph(nodes, "depthwise_mobile", [x], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 8
    return BuiltModel("depthwise_mobile", "MobileNet-style depthwise CNN", model, flops, "image")


def _encoder_layer(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    src: str,
    out: str,
    prefix: str,
    S: int,
    H: int,
    heads: int,
    ffn_mult: int,
    seed: int,
) -> None:
    hd = H // heads
    scale = 1.0 / math.sqrt(hd)
    inits += [
        _w(prefix + "_wq", (H, H), 1.0 / math.sqrt(H), seed + 1),
        _bias(prefix + "_bq", H, seed + 2),
        _w(prefix + "_wk", (H, H), 1.0 / math.sqrt(H), seed + 3),
        _bias(prefix + "_bk", H, seed + 4),
        _w(prefix + "_wv", (H, H), 1.0 / math.sqrt(H), seed + 5),
        _bias(prefix + "_bv", H, seed + 6),
        _w(prefix + "_wo", (H, H), 1.0 / math.sqrt(H), seed + 7),
        _bias(prefix + "_bo", H, seed + 8),
        _w(prefix + "_w1", (H, H * ffn_mult), 1.0 / math.sqrt(H), seed + 9),
        _bias(prefix + "_b1", H * ffn_mult, seed + 10),
        _w(prefix + "_w2", (H * ffn_mult, H), 1.0 / math.sqrt(H * ffn_mult), seed + 11),
        _bias(prefix + "_b2", H, seed + 12),
        numpy_helper.from_array(np.asarray([1, S, heads, hd], dtype=np.int64), prefix + "_to_heads"),
        numpy_helper.from_array(np.asarray([1, heads, S, hd], dtype=np.int64), prefix + "_transpose"),
        numpy_helper.from_array(np.asarray([1, S, H], dtype=np.int64), prefix + "_from_heads"),
        _const_1d(prefix + "_scale", [scale]),
    ]

    nodes += _linear(src, prefix + "_wq", prefix + "_bq", prefix + "_q0", prefix + "_q")
    nodes += _linear(src, prefix + "_wk", prefix + "_bk", prefix + "_k0", prefix + "_k")
    nodes += _linear(src, prefix + "_wv", prefix + "_bv", prefix + "_v0", prefix + "_v")

    for tag, name in (("q", prefix + "_q0"), ("k", prefix + "_k0"), ("v", prefix + "_v0")):
        nodes.append(helper.make_node("Reshape", [name, prefix + "_to_heads"], [prefix + f"_{tag}_rs"], name=prefix + f"_{tag}_reshape"))
        nodes.append(helper.make_node("Transpose", [prefix + f"_{tag}_rs"], [prefix + f"_{tag}"], name=prefix + f"_{tag}_transpose", perm=[0, 2, 1, 3]))

    nodes.append(helper.make_node("Transpose", [prefix + "_k"], [prefix + "_kT"], name=prefix + "_kT", perm=[0, 1, 3, 2]))
    nodes.append(helper.make_node("MatMul", [prefix + "_q", prefix + "_kT"], [prefix + "_scores"], name=prefix + "_scores"))
    nodes.append(helper.make_node("Mul", [prefix + "_scores", prefix + "_scale"], [prefix + "_scores_s"], name=prefix + "_scale_mul"))
    nodes.append(helper.make_node("Softmax", [prefix + "_scores_s"], [prefix + "_probs"], name=prefix + "_softmax", axis=-1))
    nodes.append(helper.make_node("MatMul", [prefix + "_probs", prefix + "_v"], [prefix + "_ctx"], name=prefix + "_ctx"))
    nodes.append(helper.make_node("Transpose", [prefix + "_ctx"], [prefix + "_ctx_t"], name=prefix + "_ctx_t", perm=[0, 2, 1, 3]))
    nodes.append(helper.make_node("Reshape", [prefix + "_ctx_t", prefix + "_from_heads"], [prefix + "_ctx_f"], name=prefix + "_ctx_reshape"))
    nodes += _linear(prefix + "_ctx_f", prefix + "_wo", prefix + "_bo", prefix + "_attn", prefix + "_o")

    # residual + layernorm
    inits.append(_const_1d(prefix + "_ln1_s", [1.0] * H))
    inits.append(_const_1d(prefix + "_ln1_o", [0.0] * H))
    nodes.append(helper.make_node("Add", [src, prefix + "_attn"], [prefix + "_res1"], name=prefix + "_res1"))
    nodes.append(
        helper.make_node(
            "LayerNormalization",
            [prefix + "_res1", prefix + "_ln1_s", prefix + "_ln1_o"],
            [prefix + "_n1"],
            name=prefix + "_ln1",
            axis=-1,
            epsilon=1e-5,
        )
    )
    nodes += _linear(prefix + "_n1", prefix + "_w1", prefix + "_b1", prefix + "_ff_pre", prefix + "_ff1")
    g_nodes, g_inits = _gelu_tanh(prefix + "_ff_pre", prefix + "_ff_gelu", prefix + "_ff")
    nodes += g_nodes
    inits += g_inits
    nodes += _linear(prefix + "_ff_gelu", prefix + "_w2", prefix + "_b2", prefix + "_ff", prefix + "_ff2")

    inits.append(_const_1d(prefix + "_ln2_s", [1.0] * H))
    inits.append(_const_1d(prefix + "_ln2_o", [0.0] * H))
    nodes.append(helper.make_node("Add", [prefix + "_n1", prefix + "_ff"], [prefix + "_res2"], name=prefix + "_res2"))
    nodes.append(
        helper.make_node(
            "LayerNormalization",
            [prefix + "_res2", prefix + "_ln2_s", prefix + "_ln2_o"],
            [out],
            name=prefix + "_ln2",
            axis=-1,
            epsilon=1e-5,
        )
    )


def build_transformer_encoder() -> BuiltModel:
    """Transformer encoder block stack - NLP representation quality."""
    S, H, heads, layers, ffn_mult = 192, 768, 12, 2, 4
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    inits.append(_w("embed_w", (H, H), 1.0 / math.sqrt(H), 4040))
    inits.append(_bias("embed_b", H, 4041))
    nodes += _linear("x", "embed_w", "embed_b", "e0", "embed")
    prev = "e0"
    for i in range(layers):
        nxt = f"enc{i}"
        _encoder_layer(nodes, inits, prev, nxt, f"L{i}", S, H, heads, ffn_mult, 5000 + i * 100)
        prev = nxt
    nodes.append(helper.make_node("Identity", [prev], ["y"], name="out_id"))

    x = _vi("x", [1, S, H])
    y = _vi("y", [1, S, H])
    graph = helper.make_graph(nodes, "transformer_encoder", [x], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 8

    per_layer = (
        3 * 2 * S * H * H  # qkv
        + 2 * 2 * S * S * H  # attention matmuls
        + 2 * S * H * H  # out proj
        + 2 * 2 * S * H * H * ffn_mult  # ffn
    )
    flops = per_layer * layers
    return BuiltModel("transformer", f"Transformer encoder x{layers} (S={S})", model, flops, "token")


def _decoder_weights(
    inits: list[onnx.TensorProto],
    H: int,
    heads: int,
    prefix: str,
    seed: int,
    seq: int,
    total: int,
    ffn_mult: int = 4,
) -> None:
    hd = H // heads
    inits += [
        _w(prefix + "_wq", (H, H), 1.0 / math.sqrt(H), seed + 1),
        _w(prefix + "_wk", (H, H), 1.0 / math.sqrt(H), seed + 2),
        _w(prefix + "_wv", (H, H), 1.0 / math.sqrt(H), seed + 3),
        _w(prefix + "_wo", (H, H), 1.0 / math.sqrt(H), seed + 4),
        _w(prefix + "_w1", (H, H * ffn_mult), 1.0 / math.sqrt(H), seed + 5),
        _w(prefix + "_w2", (H * ffn_mult, H), 1.0 / math.sqrt(H * ffn_mult), seed + 6),
        numpy_helper.from_array(np.asarray([1, seq, heads, hd], dtype=np.int64), prefix + "_q_heads"),
        numpy_helper.from_array(np.asarray([1, total, heads, hd], dtype=np.int64), prefix + "_kv_heads"),
        numpy_helper.from_array(np.asarray([1, seq, H], dtype=np.int64), prefix + "_from_heads"),
        _const_1d(prefix + "_scale", [1.0 / math.sqrt(hd)]),
        _const_1d(prefix + "_ln1_s", [1.0] * H),
        _const_1d(prefix + "_ln1_o", [0.0] * H),
        _const_1d(prefix + "_ln2_s", [1.0] * H),
        _const_1d(prefix + "_ln2_o", [0.0] * H),
    ]


def _decoder_layer(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    src: str,
    past_k: str,
    past_v: str,
    out_y: str,
    present_k: str,
    present_v: str,
    prefix: str,
    H: int,
    heads: int,
) -> None:
    nodes.append(helper.make_node("MatMul", [src, prefix + "_wq"], [prefix + "_q_lin"], name=prefix + "_q_lin"))
    nodes.append(helper.make_node("MatMul", [src, prefix + "_wk"], [prefix + "_k_lin"], name=prefix + "_k_lin"))
    nodes.append(helper.make_node("MatMul", [src, prefix + "_wv"], [prefix + "_v_lin"], name=prefix + "_v_lin"))
    # fold the single new token into the cache
    nodes.append(helper.make_node("Concat", [past_k, prefix + "_k_lin"], [present_k], name=prefix + "_cache_k", axis=1))
    nodes.append(helper.make_node("Concat", [past_v, prefix + "_v_lin"], [present_v], name=prefix + "_cache_v", axis=1))

    nodes.append(helper.make_node("Reshape", [prefix + "_q_lin", prefix + "_q_heads"], [prefix + "_q_rs"], name=prefix + "_q_rs"))
    nodes.append(helper.make_node("Transpose", [prefix + "_q_rs"], [prefix + "_q"], name=prefix + "_q_t", perm=[0, 2, 1, 3]))
    nodes.append(helper.make_node("Reshape", [present_k, prefix + "_kv_heads"], [prefix + "_k_rs"], name=prefix + "_k_rs"))
    nodes.append(helper.make_node("Transpose", [prefix + "_k_rs"], [prefix + "_k"], name=prefix + "_k_t", perm=[0, 2, 1, 3]))
    nodes.append(helper.make_node("Transpose", [prefix + "_k"], [prefix + "_kT"], name=prefix + "_kT", perm=[0, 1, 3, 2]))
    nodes.append(helper.make_node("Reshape", [present_v, prefix + "_kv_heads"], [prefix + "_v_rs"], name=prefix + "_v_rs"))
    nodes.append(helper.make_node("Transpose", [prefix + "_v_rs"], [prefix + "_v"], name=prefix + "_v_t", perm=[0, 2, 1, 3]))

    nodes.append(helper.make_node("MatMul", [prefix + "_q", prefix + "_kT"], [prefix + "_scores"], name=prefix + "_scores"))
    nodes.append(helper.make_node("Mul", [prefix + "_scores", prefix + "_scale"], [prefix + "_scores_s"], name=prefix + "_sc"))
    nodes.append(helper.make_node("Softmax", [prefix + "_scores_s"], [prefix + "_probs"], name=prefix + "_sm", axis=-1))
    nodes.append(helper.make_node("MatMul", [prefix + "_probs", prefix + "_v"], [prefix + "_ctx"], name=prefix + "_ctx"))
    nodes.append(helper.make_node("Transpose", [prefix + "_ctx"], [prefix + "_ctx_t"], name=prefix + "_ctx_t", perm=[0, 2, 1, 3]))
    nodes.append(helper.make_node("Reshape", [prefix + "_ctx_t", prefix + "_from_heads"], [prefix + "_ctx_f"], name=prefix + "_ctx_rs"))

    nodes.append(helper.make_node("MatMul", [prefix + "_ctx_f", prefix + "_wo"], [prefix + "_o_lin"], name=prefix + "_o_lin"))
    nodes.append(helper.make_node("Add", [src, prefix + "_o_lin"], [prefix + "_res1"], name=prefix + "_res1"))
    nodes.append(
        helper.make_node(
            "LayerNormalization",
            [prefix + "_res1", prefix + "_ln1_s", prefix + "_ln1_o"],
            [prefix + "_n1"],
            name=prefix + "_ln1",
            axis=-1,
            epsilon=1e-5,
        )
    )
    nodes.append(helper.make_node("MatMul", [prefix + "_n1", prefix + "_w1"], [prefix + "_ff1"], name=prefix + "_ff1"))
    g_nodes, g_inits = _gelu_tanh(prefix + "_ff1", prefix + "_ff_gelu", prefix + "_ffgelu")
    nodes += g_nodes
    inits += g_inits
    nodes.append(helper.make_node("MatMul", [prefix + "_ff_gelu", prefix + "_w2"], [prefix + "_ff2"], name=prefix + "_ff2"))
    nodes.append(helper.make_node("Add", [prefix + "_n1", prefix + "_ff2"], [prefix + "_res2"], name=prefix + "_res2"))
    nodes.append(
        helper.make_node(
            "LayerNormalization",
            [prefix + "_res2", prefix + "_ln2_s", prefix + "_ln2_o"],
            [out_y],
            name=prefix + "_ln2",
            axis=-1,
            epsilon=1e-5,
        )
    )


def _build_decoder(seq: int, past: int, H: int, heads: int, layers: int, key: str, desc: str) -> BuiltModel:
    """Decoder-only transformer.

    ``seq=1`` with a populated cache is the *decode* phase (memory-bandwidth
    bound, the phase users actually feel); ``seq>1`` is *prefill* (compute
    bound). NPUs and iGPUs look very different across these two phases, which
    is exactly why both are in the suite.
    """
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    inits.append(_w("embed_w", (H, H), 1.0 / math.sqrt(H), 6060))
    nodes.append(helper.make_node("MatMul", ["x", "embed_w"], ["e0"], name="embed"))

    prev_x = "e0"
    for i in range(layers):
        pk, pv = f"past_k_{i}", f"past_v_{i}"
        nk, nv = f"present_k_{i}", f"present_v_{i}"
        _decoder_weights(inits, H, heads, f"D{i}", 7000 + i * 100, seq, past + seq)
        _decoder_layer(nodes, inits, prev_x, pk, pv, f"d{i}", nk, nv, f"D{i}", H, heads)
        prev_x = f"d{i}"
    nodes.append(helper.make_node("Identity", [prev_x], ["y"], name="out_id"))

    inputs = [_vi("x", [1, seq, H])]
    outputs = [_vi("y", [1, seq, H])]
    for i in range(layers):
        # Each layer owns its own cache, so prefill and decode share one graph
        # and one set of weights.
        inputs.append(_vi(f"past_k_{i}", [1, past, H]))
        inputs.append(_vi(f"past_v_{i}", [1, past, H]))
        outputs.append(_vi(f"present_k_{i}", [1, past + seq, H]))
        outputs.append(_vi(f"present_v_{i}", [1, past + seq, H]))

    graph = helper.make_graph(nodes, key, inputs, outputs, inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 8

    total = past + seq
    per_layer = (
        3 * 2 * seq * H * H  # qkv projections
        + 2 * 2 * seq * total * H  # qk^T and attn@v
        + 2 * seq * H * H  # output projection
        + 2 * 2 * seq * H * 4 * H  # ffn
    )
    flops = per_layer * layers + 2 * seq * H * H
    return BuiltModel(key, desc, model, flops, "token")


def build_llm_prefill() -> BuiltModel:
    return _build_decoder(
        seq=192,
        past=0,
        H=1024,
        heads=16,
        layers=2,
        key="llm_prefill",
        desc="LLM prefill, 192 tokens (compute bound)",
    )


def build_llm_decode() -> BuiltModel:
    return _build_decoder(
        seq=1,
        past=768,
        H=1024,
        heads=16,
        layers=2,
        key="llm_decode",
        desc="LLM decode with 768-token KV cache (bandwidth bound)",
    )


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
def build_dense_static() -> BuiltModel:
    return replace(build_dense_gemm(), key="dense_static", description="MLP for static INT8 QDQ calibration")


BASE_BUILDERS: dict[str, Callable[[], BuiltModel]] = {
    "dense_static": build_dense_static,
    "dense_gemm": build_dense_gemm,
    "conv_vision": build_conv_vision,
    "depthwise_mobile": build_depthwise_mobile,
    "transformer": build_transformer_encoder,
    "llm_prefill": build_llm_prefill,
    "llm_decode": build_llm_decode,
}


def build_smoke() -> BuiltModel:
    """A ~1 ms graph used only to prove a device can execute anything at all.

    Keeping it tiny makes device verification cheap, so a target that hangs or
    aborts the process is discovered before the suite commits an hour to it.
    """
    M, K, N = 8, 128, 64
    x = _vi("x", [M, K])
    y = _vi("y", [M, N])
    inits = [
        _w("w1", (K, N), 1.0 / math.sqrt(K), 9001),
        _bias("b1", N, 9002),
    ]
    nodes = [
        helper.make_node("MatMul", ["x", "w1"], ["pre"], name="mm"),
        helper.make_node("Add", ["pre", "b1"], ["y"], name="add"),
    ]
    graph = helper.make_graph(nodes, "smoke", [x], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 8
    return BuiltModel("smoke", "device verification probe", model, 2 * M * K * N, "sample")

# Graphs that tolerate post-training integer quantisation cleanly (MatMul-only
# graphs; convolutions need a calibrated static pipeline, which is provided by
# the OpenVINO backend instead).
QUANTISABLE = ("dense_gemm", "llm_decode", "transformer", "dense_static")

# Why the last quantisation attempt failed, if it did. Surfaced to the user
# instead of a silent "unknown workload".
QUANTISATION_ERROR: str | None = None


def build_all(include_int8: bool = True) -> dict[str, BuiltModel]:
    models: dict[str, BuiltModel] = {}
    for key, fn in BASE_BUILDERS.items():
        models[key] = fn()
    if include_int8:
        for key in QUANTISABLE:
            q = quantise_static(models[key]) if key == "dense_static" else quantise_dynamic(models[key])
            if q is not None:
                models[q.key] = q
    return models


def build_one(key: str) -> BuiltModel:
    """Rebuild a single graph by key.

    Workers call this instead of receiving a pickled graph: regenerating from
    the seed and comparing the hash in the result file is a stronger guarantee
    than trusting a serialised blob.
    """
    if key == "smoke":
        return build_smoke()
    if key in BASE_BUILDERS:
        return BASE_BUILDERS[key]()
    if key.endswith("_int8"):
        base = key[: -len("_int8")]
        if base in BASE_BUILDERS:
            built = BASE_BUILDERS[base]()
            quantised = quantise_static(built) if base == "dense_static" else quantise_dynamic(built)
            if quantised is not None:
                return quantised
            raise KeyError(f"workload '{key}' is unavailable: {QUANTISATION_ERROR or 'quantisation failed'}")
    raise KeyError(f"unknown workload '{key}'")


def has_int8(key: str) -> bool:
    return key in QUANTISABLE


def quantise_static(built: BuiltModel) -> BuiltModel | None:
    """Calibrate activations and weights for QDQ INT8, using fixed synthetic inputs.

    These are synthetic workloads, so calibration represents their seeded input
    distribution rather than accuracy on a real-world dataset. Keep the dynamic
    variants unchanged for comparison with existing reports.
    """
    global QUANTISATION_ERROR
    import tempfile

    QUANTISATION_ERROR = None
    cache = Path(".cache") / "models"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{built.key}-qdq-v1-{built.sha256[:12]}.onnx"
    try:
        if not path.exists():
            from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

            class Calibration(CalibrationDataReader):
                def __init__(self):
                    self.samples = iter(range(SEED + 1, SEED + 9))

                def get_next(self):
                    seed = next(self.samples, None)
                    return built.make_inputs(seed) if seed is not None else None

            with tempfile.TemporaryDirectory() as td:
                source, output = Path(td) / "source.onnx", Path(td) / "int8.onnx"
                source.write_bytes(built.serialise())
                quantize_static(str(source), str(output), Calibration(),
                                quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8,
                                weight_type=QuantType.QInt8, per_channel=True,
                                op_types_to_quantize=["MatMul", "Gemm"],
                                extra_options={"ActivationSymmetric": True, "WeightSymmetric": True})
                quantised = onnx.load(str(output))
                onnx.checker.check_model(quantised)
                path.write_bytes(quantised.SerializeToString(deterministic=True))
        model = onnx.load(str(path))
        return replace(built, key=built.key + "_int8", model=model, precision="int8",
                       description=built.description + " [static INT8 activations + weights]")
    except Exception as exc:
        QUANTISATION_ERROR = f"static QDQ quantisation failed: {type(exc).__name__}: {exc}"
        return None


def quantise_dynamic(built: BuiltModel) -> BuiltModel | None:
    """Post-training dynamic INT8 quantisation (weights only, runtime scales).

    Chosen over static quantisation because it needs no calibration set, which
    makes it byte-reproducible on every machine - the property that keeps the
    CPU/GPU/NPU comparison honest.

    The result is cached next to the float graph and keyed by the *source* graph
    hash, so a worker never re-quantises and a stale cache is impossible. The
    cache is consulted *before* importing the quantiser, so a packaged build
    that cannot import it still gets to reuse a previously built graph.

    On failure the reason is left in ``QUANTISATION_ERROR`` rather than being
    swallowed: a silently missing INT8 column looks like a benchmark result.
    """
    global QUANTISATION_ERROR
    QUANTISATION_ERROR = None

    key = built.key + "_int8"
    cache = Path(".cache") / "models"
    cache.mkdir(parents=True, exist_ok=True)
    cached = cache / f"{key}-{built.sha256[:12]}.onnx"

    qmodel = None
    if cached.exists():
        try:
            qmodel = onnx.load(str(cached))
        except Exception as exc:
            QUANTISATION_ERROR = f"the cached INT8 graph could not be read ({exc})"
            qmodel = None

    if qmodel is None:
        import logging
        import tempfile

        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
        except Exception as exc:
            QUANTISATION_ERROR = f"onnxruntime.quantization is not available in this interpreter ({exc})"
            return None

        try:
            with tempfile.TemporaryDirectory() as td:
                src = Path(td) / "f32.onnx"
                dst = Path(td) / "i8.onnx"
                src.write_bytes(built.serialise())
                # ONNX Runtime's quantiser logs a "consider pre-processing"
                # warning to the root logger on every call; it is noise here.
                previous = logging.root.manager.disable
                logging.disable(logging.WARNING)
                try:
                    quantize_dynamic(
                        str(src),
                        str(dst),
                        weight_type=QuantType.QInt8,
                        extra_options={"MatMulConstBOnly": False},
                    )
                finally:
                    logging.disable(previous)
                qmodel = onnx.load(str(dst))
            cached.write_bytes(qmodel.SerializeToString(deterministic=True))
        except Exception as exc:
            QUANTISATION_ERROR = f"{type(exc).__name__}: {exc}"
            return None

    return BuiltModel(
        key=key,
        description=built.description + " [INT8 weights]",
        model=qmodel,
        flops=built.flops,
        unit=built.unit,
        precision="int8",
    )


def describe_build_error(key: str) -> str | None:
    """Why ``key`` could not be built, or None if it can."""
    try:
        build_one(key)
        return None
    except Exception as exc:
        return str(exc)
