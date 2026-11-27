"""
Módulo de Compresión (RLE, Huffman, Mu-Law, DCT 8x8) y Serialización (.dspbin, WAV, PNG, SVG, BIN, JSON).
Responsable asignado: Mariano Orrego (feat/compression-docs-mariano)

Contratos en dsp_engine/contracts.py:
- CompressedPayload, CompressionType, ImageMatrix, Signal1D, ModulationType, AuditMetrics
"""

from dsp_engine.compression.container import (
    DSPBIN_MAGIC,
    DSPBIN_VERSION,
    DspBinPacket,
    export_bin_bytes,
    export_png_bytes,
    export_session_report_json,
    export_svg_string,
    export_wav_bytes,
    import_wav_bytes,
    pack_dspbin,
    unpack_dspbin,
)
from dsp_engine.compression.dct_image import (
    JPEG_CHROMINANCE_Q50,
    JPEG_LUMINANCE_Q50,
    compress_dct_image,
    compute_image_mse_psnr,
    dct2d_blocks_forward,
    dct2d_blocks_inverse,
    decompress_dct_image,
    scale_quantization_matrix,
)
from dsp_engine.compression.huffman import (
    build_huffman_code_table,
    compress_huffman,
    compress_huffman_bits,
    compress_huffman_image,
    compute_shannon_entropy,
    decompress_huffman,
    decompress_huffman_bits,
    decompress_huffman_image,
    deserialize_huffman_tree,
    serialize_huffman_tree,
)
from dsp_engine.compression.mu_law import (
    compress_mu_law,
    decompress_mu_law,
    mu_law_compand,
    mu_law_expand,
)
from dsp_engine.compression.rle import (
    compress_rle,
    compress_rle_bits,
    compress_rle_image,
    decode_packbits_bytes,
    decompress_rle,
    decompress_rle_bits,
    decompress_rle_image,
    encode_packbits_bytes,
)

COMPRESSION_MODULE_OWNER = "Mariano Orrego"

__all__ = [
    "COMPRESSION_MODULE_OWNER",
    "DSPBIN_MAGIC",
    "DSPBIN_VERSION",
    "DspBinPacket",
    "JPEG_CHROMINANCE_Q50",
    "JPEG_LUMINANCE_Q50",
    "build_huffman_code_table",
    "compress_dct_image",
    "compress_huffman",
    "compress_huffman_bits",
    "compress_huffman_image",
    "compress_mu_law",
    "compress_rle",
    "compress_rle_bits",
    "compress_rle_image",
    "compute_image_mse_psnr",
    "compute_shannon_entropy",
    "dct2d_blocks_forward",
    "dct2d_blocks_inverse",
    "decode_packbits_bytes",
    "decompress_dct_image",
    "decompress_huffman",
    "decompress_huffman_bits",
    "decompress_huffman_image",
    "decompress_mu_law",
    "decompress_rle",
    "decompress_rle_bits",
    "decompress_rle_image",
    "deserialize_huffman_tree",
    "encode_packbits_bytes",
    "export_bin_bytes",
    "export_png_bytes",
    "export_session_report_json",
    "export_svg_string",
    "export_wav_bytes",
    "import_wav_bytes",
    "mu_law_compand",
    "mu_law_expand",
    "pack_dspbin",
    "scale_quantization_matrix",
    "serialize_huffman_tree",
    "unpack_dspbin",
]
