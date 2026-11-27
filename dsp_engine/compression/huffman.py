"""
Compresión Sin Pérdida de Huffman con Árbol Serializable.
Responsable: Mariano Orrego (feat/compression-docs-mariano)

Fundamento Matemático (Teoría de la Información — Shannon / Huffman):
- Entropía de la fuente:
      H(X) = - sum_{i} p_i * log2(p_i)   [bits/símbolo]
- Longitud promedio del código óptimo prefijo:
      L_avg = sum_{i} p_i * l_i,   con H(X) <= L_avg < H(X) + 1
- Garantiza reconstrucción exacta sin pérdida: MSE = 0, BER = 0%.
"""

import heapq
import json
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from dsp_engine.contracts import CompressedPayload, ImageMatrix


class _HuffmanNode:
    """Nodo interno u hoja del árbol binario de Huffman."""

    __slots__ = ("freq", "symbol", "left", "right")

    def __init__(
        self,
        freq: int,
        symbol: Optional[int] = None,
        left: Optional["_HuffmanNode"] = None,
        right: Optional["_HuffmanNode"] = None,
    ) -> None:
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right


def compute_shannon_entropy(data: NDArray[np.uint8]) -> float:
    """Calcula la entropía empírica de Shannon H(X) en bits por byte."""
    flat = np.asarray(data, dtype=np.uint8).reshape(-1)
    if flat.size == 0:
        return 0.0
    counts = np.bincount(flat, minlength=256)
    probs = counts[counts > 0].astype(np.float64) / float(flat.size)
    return float(-np.sum(probs * np.log2(probs)))


def build_huffman_code_table(data: NDArray[np.uint8]) -> Tuple[Dict[int, str], float]:
    """
    Construye la tabla determinística de códigos prefijo de Huffman {símbolo: '01...'}.
    Retorna (code_table, average_code_length_bits).
    """
    flat = np.asarray(data, dtype=np.uint8).reshape(-1)
    if flat.size == 0:
        return {}, 0.0

    counts = np.bincount(flat, minlength=256)
    heap: List[Tuple[int, int, _HuffmanNode]] = []
    tie_id = 0

    for sym in range(256):
        freq = int(counts[sym])
        if freq > 0:
            heapq.heappush(heap, (freq, tie_id, _HuffmanNode(freq=freq, symbol=sym)))
            tie_id += 1

    if len(heap) == 1:
        _, _, only_node = heap[0]
        assert only_node.symbol is not None
        return {only_node.symbol: "0"}, 1.0

    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        merged = _HuffmanNode(freq=f1 + f2, symbol=None, left=n1, right=n2)
        heapq.heappush(heap, (merged.freq, tie_id, merged))
        tie_id += 1

    root = heap[0][2]
    code_table: Dict[int, str] = {}

    def _assign_codes(node: _HuffmanNode, prefix: str) -> None:
        if node.symbol is not None:
            code_table[node.symbol] = prefix
            return
        if node.left is not None:
            _assign_codes(node.left, prefix + "0")
        if node.right is not None:
            _assign_codes(node.right, prefix + "1")

    _assign_codes(root, "")

    total = float(flat.size)
    avg_len = sum((int(counts[sym]) / total) * len(code) for sym, code in code_table.items())
    return code_table, float(avg_len)


def serialize_huffman_tree(code_table: Dict[int, str]) -> str:
    """Serializa la tabla de códigos de Huffman a JSON compacto."""
    serializable = {str(sym): code for sym, code in sorted(code_table.items())}
    return json.dumps(serializable, separators=(",", ":"))


def deserialize_huffman_tree(tree_json: str) -> Dict[int, str]:
    """Deserializa la tabla de códigos de Huffman desde JSON."""
    if not tree_json or tree_json == "{}":
        return {}
    raw: Dict[str, str] = json.loads(tree_json)
    return {int(sym_str): code for sym_str, code in raw.items()}


def _build_decode_trie(code_table: Dict[int, str]) -> Dict[str, Any]:
    """Construye un trie binario ligero { '0': subnode, '1': subnode, 'sym': int }."""
    root: Dict[str, Any] = {}
    for sym, code in code_table.items():
        node = root
        for bit_char in code:
            if bit_char not in ("0", "1"):
                raise ValueError(f"Código Huffman inválido '{code}' para símbolo {sym}.")
            if bit_char not in node:
                node[bit_char] = {}
            node = node[bit_char]
        node["sym"] = sym
    return root


def compress_huffman(
    data: NDArray[np.uint8],
    source_type: Literal["audio", "image", "bitstream"] = "bitstream",
    metadata: Optional[Dict[str, Any]] = None,
) -> CompressedPayload:
    """Comprime un flujo uint8 mediante codificación de Huffman con árbol JSON serializable."""
    raw = np.asarray(data, dtype=np.uint8).reshape(-1)
    orig_len = int(raw.size)

    meta: Dict[str, Any] = dict(metadata) if metadata else {}
    meta["sourceType"] = source_type
    meta["originalByteLength"] = orig_len

    if orig_len == 0:
        meta["huffmanTreeJson"] = "{}"
        meta["totalEncodedBits"] = 0
        meta["entropyBitsPerByte"] = 0.0
        meta["avgCodeLengthBits"] = 0.0
        return CompressedPayload(
            codec="huffman",
            original_bytes=0,
            compressed_bytes=0,
            compression_ratio=1.0,
            savings_percent=0.0,
            payload=np.zeros(0, dtype=np.uint8),
            metadata=meta,
        )

    code_table, avg_len = build_huffman_code_table(raw)
    entropy = compute_shannon_entropy(raw)
    tree_json = serialize_huffman_tree(code_table)

    bit_str = "".join(code_table[int(b)] for b in raw)
    total_bits = len(bit_str)
    bit_arr = (np.frombuffer(bit_str.encode("ascii"), dtype=np.uint8) - 48).astype(np.uint8)
    packed = np.packbits(bit_arr, bitorder="big")

    comp_len = int(packed.size)
    ratio = float(orig_len / comp_len) if comp_len > 0 else float(orig_len)
    savings = float((1.0 - (comp_len / orig_len)) * 100.0)

    meta["huffmanTreeJson"] = tree_json
    meta["totalEncodedBits"] = total_bits
    meta["entropyBitsPerByte"] = entropy
    meta["avgCodeLengthBits"] = avg_len

    return CompressedPayload(
        codec="huffman",
        original_bytes=orig_len,
        compressed_bytes=comp_len,
        compression_ratio=ratio,
        savings_percent=savings,
        payload=packed,
        metadata=meta,
    )


def decompress_huffman(compressed: CompressedPayload) -> NDArray[np.uint8]:
    """Descomprime un CompressedPayload Huffman reconstruyendo bit a bit los bytes exactos."""
    if compressed.codec != "huffman":
        raise ValueError(f"Códec inválido para decompress_huffman: '{compressed.codec}'.")

    orig_len = int(compressed.metadata.get("originalByteLength", compressed.original_bytes))
    if orig_len == 0:
        return np.zeros(0, dtype=np.uint8)

    tree_json = str(compressed.metadata.get("huffmanTreeJson", ""))
    if not tree_json:
        raise ValueError("Metadatos Huffman incompletos: falta 'huffmanTreeJson'.")

    code_table = deserialize_huffman_tree(tree_json)
    if not code_table:
        raise ValueError("Tabla de códigos Huffman vacía para carga no vacía.")

    trie = _build_decode_trie(code_table)
    total_bits = int(compressed.metadata.get("totalEncodedBits", int(compressed.payload.size) * 8))
    bits = np.unpackbits(np.asarray(compressed.payload, dtype=np.uint8), bitorder="big")[:total_bits]

    out = np.zeros(orig_len, dtype=np.uint8)
    out_idx = 0
    node = trie
    for b in bits:
        key = "1" if b else "0"
        if key not in node:
            raise ValueError("Flujo de bits Huffman corrupto o incompatible con el árbol.")
        node = node[key]
        if "sym" in node:
            out[out_idx] = int(node["sym"])
            out_idx += 1
            node = trie
            if out_idx == orig_len:
                break

    if out_idx != orig_len:
        raise ValueError(
            f"Decodificación Huffman incompleta: se esperaban {orig_len} bytes, se recuperaron {out_idx}."
        )
    return out


def compress_huffman_bits(
    bits: NDArray[np.uint8],
    metadata: Optional[Dict[str, Any]] = None,
) -> CompressedPayload:
    """Empaqueta una traza de bits (0/1) MSB-first y la comprime sin pérdida con Huffman."""
    flat_bits = (np.asarray(bits, dtype=np.uint8) & 1).reshape(-1)
    total_bits = int(flat_bits.size)
    packed = np.packbits(flat_bits, bitorder="big") if total_bits > 0 else np.zeros(0, dtype=np.uint8)

    meta: Dict[str, Any] = dict(metadata) if metadata else {}
    meta["originalBitLength"] = total_bits
    return compress_huffman(packed, source_type="bitstream", metadata=meta)


def decompress_huffman_bits(compressed: CompressedPayload) -> NDArray[np.uint8]:
    """Recupera la traza binaria exacta (BER = 0%) desde un CompressedPayload Huffman."""
    recovered_bytes = decompress_huffman(compressed)
    if recovered_bytes.size == 0:
        return np.zeros(0, dtype=np.uint8)
    bits = np.unpackbits(recovered_bytes, bitorder="big").astype(np.uint8)
    total_bits = compressed.metadata.get("originalBitLength")
    if total_bits is not None:
        return bits[: int(total_bits)]
    return bits


def compress_huffman_image(image: ImageMatrix) -> CompressedPayload:
    """Comprime los píxeles de una ImageMatrix sin pérdida mediante Huffman."""
    total_px = image.width * image.height
    raw = np.asarray(image.pixels[: total_px * image.channels], dtype=np.uint8)
    return compress_huffman(
        raw,
        source_type="image",
        metadata={
            "width": int(image.width),
            "height": int(image.height),
            "channels": int(image.channels),
            "bitDepth": 8,
            "label": image.label,
        },
    )


def decompress_huffman_image(compressed: CompressedPayload) -> ImageMatrix:
    """Reconstruye exactamente una ImageMatrix desde un CompressedPayload Huffman (MSE = 0)."""
    width = int(compressed.metadata["width"])
    height = int(compressed.metadata["height"])
    channels = int(compressed.metadata["channels"])
    if channels not in (1, 3, 4):
        raise ValueError(f"Número de canales inválido en metadatos Huffman: {channels}")

    raw = decompress_huffman(compressed)
    expected_len = width * height * channels
    if raw.size != expected_len:
        raise ValueError(
            f"Tamaño reconstruido ({raw.size}) no coincide con {width}x{height}x{channels} ({expected_len})."
        )

    return ImageMatrix(
        width=width,
        height=height,
        channels=channels,  # type: ignore[arg-type]
        pixels=raw.astype(np.uint8),
        label=str(compressed.metadata.get("label", f"Huffman-{width}x{height}")),
    )
