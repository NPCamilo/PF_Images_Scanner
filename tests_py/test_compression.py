import json
import struct
import unittest
import zlib
import numpy as np

from dsp_engine.compression import (
    DSPBIN_MAGIC,
    JPEG_LUMINANCE_Q50,
    compress_dct_image,
    compress_huffman,
    compress_huffman_bits,
    compress_huffman_image,
    compress_mu_law,
    compress_rle,
    compress_rle_bits,
    compress_rle_image,
    compute_image_mse_psnr,
    compute_shannon_entropy,
    dct2d_blocks_forward,
    dct2d_blocks_inverse,
    decompress_dct_image,
    decompress_huffman,
    decompress_huffman_bits,
    decompress_huffman_image,
    decompress_mu_law,
    decompress_rle,
    decompress_rle_bits,
    decompress_rle_image,
    deserialize_huffman_tree,
    export_bin_bytes,
    export_png_bytes,
    export_session_report_json,
    export_svg_string,
    export_wav_bytes,
    import_wav_bytes,
    mu_law_compand,
    mu_law_expand,
    pack_dspbin,
    unpack_dspbin,
)
from dsp_engine.contracts import AuditMetrics, ImageMatrix, Signal1D
from dsp_engine.core import generate_synthetic_signal, generate_test_image, quantize_pcm


class TestCompressionAndSerialization(unittest.TestCase):
    """Pruebas DDD de compresión (RLE, Huffman, μ-law, DCT 8x8) y serialización (.dspbin / exportadores)."""

    def test_rle_lossless_roundtrip_images_bits_and_edge_cases(self):
        # 1. Imagen sintética de barras con planar RLE (MSE = 0)
        img = generate_test_image(32, 32, pattern="bars")
        comp_img = compress_rle_image(img, planar=True)
        self.assertEqual(comp_img.codec, "rle")
        self.assertGreater(comp_img.compression_ratio, 2.0)
        self.assertGreater(comp_img.savings_percent, 50.0)

        rec_img = decompress_rle_image(comp_img)
        np.testing.assert_array_equal(rec_img.pixels, img.pixels)
        mse, _ = compute_image_mse_psnr(img, rec_img)
        self.assertEqual(mse, 0.0)

        # 2. Traza binaria de longitud impar (137 bits, BER = 0%)
        rng = np.random.default_rng(7)
        bits = rng.integers(0, 2, size=137, dtype=np.uint8)
        comp_bits = compress_rle_bits(bits)
        rec_bits = decompress_rle_bits(comp_bits)
        np.testing.assert_array_equal(rec_bits, bits)

        # 3. Datos aleatorios de alta entropía: expansión acotada (< 2%)
        rand_bytes = rng.integers(0, 256, size=1024, dtype=np.uint8)
        comp_rand = compress_rle(rand_bytes)
        self.assertLessEqual(comp_rand.compressed_bytes, int(1024 * 1.02) + 2)
        np.testing.assert_array_equal(decompress_rle(comp_rand), rand_bytes)

        # 4. Casos borde: vacío, 1 byte, saturación 0x00 y 0xFF, corrida > 129
        for edge in [
            np.zeros(0, dtype=np.uint8),
            np.array([42], dtype=np.uint8),
            np.zeros(500, dtype=np.uint8),
            np.full(500, 255, dtype=np.uint8),
        ]:
            c = compress_rle(edge)
            np.testing.assert_array_equal(decompress_rle(c), edge)

    def test_huffman_lossless_entropy_and_tree_serialization(self):
        # Fuente sesgada: H(X) ≈ 0.88 bits/símbolo -> fuerte compresión
        rng = np.random.default_rng(123)
        symbols = rng.choice(
            np.array([65, 66, 67, 68], dtype=np.uint8),
            size=2000,
            p=[0.80, 0.12, 0.05, 0.03],
        )
        entropy = compute_shannon_entropy(symbols)
        comp = compress_huffman(symbols)

        self.assertEqual(comp.codec, "huffman")
        self.assertGreater(comp.compression_ratio, 3.5)
        self.assertIn("huffmanTreeJson", comp.metadata)

        # Verificar teorema de codificación de fuente de Shannon: H(X) <= L_avg < H(X) + 1
        avg_len = float(comp.metadata["avgCodeLengthBits"])
        self.assertGreaterEqual(avg_len, entropy - 1e-9)
        self.assertLess(avg_len, entropy + 1.0)

        # Verificar árbol serializable y reconstrucción exacta (MSE = 0)
        tree_map = deserialize_huffman_tree(comp.metadata["huffmanTreeJson"])
        self.assertEqual(set(tree_map.keys()), {65, 66, 67, 68})
        rec = decompress_huffman(comp)
        np.testing.assert_array_equal(rec, symbols)

        # Imagen y bitstream
        img = generate_test_image(24, 24, pattern="target")
        rec_img = decompress_huffman_image(compress_huffman_image(img))
        np.testing.assert_array_equal(rec_img.pixels, img.pixels)

        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=np.uint8)
        np.testing.assert_array_equal(decompress_huffman_bits(compress_huffman_bits(bits)), bits)

        # Casos borde: vacío, símbolo único repetido, 256 símbolos uniformes
        for edge in [
            np.zeros(0, dtype=np.uint8),
            np.full(300, 0xAA, dtype=np.uint8),
            np.arange(256, dtype=np.uint8),
        ]:
            c = compress_huffman(edge)
            np.testing.assert_array_equal(decompress_huffman(c), edge)

    def test_mu_law_companding_identity_and_low_amplitude_sqnr(self):
        # 1. Identidad analítica F^{-1}(F(x)) == x para μ = 255
        grid = np.linspace(-1.0, 1.0, 1001, dtype=np.float64)
        companded = mu_law_compand(grid, mu=255.0)
        expanded = mu_law_expand(companded, mu=255.0)
        np.testing.assert_allclose(expanded, grid, atol=1e-12)
        self.assertEqual(float(mu_law_compand(np.array([0.0]), mu=255.0)[0]), 0.0)
        self.assertEqual(float(mu_law_expand(np.array([0.0]), mu=255.0)[0]), 0.0)

        # 2. Ventaja de μ-law sobre PCM lineal uniforme de 8 bits en señal de baja amplitud
        sig = generate_synthetic_signal(
            waveform="sine",
            amplitude=0.04,
            frequency_hz=211.0,
            duration_sec=0.1,
            fs=8000.0,
        )
        comp_ml = compress_mu_law(sig, mu=255.0, compressed_bit_depth=8, original_bit_depth=16)
        self.assertEqual(comp_ml.codec, "mu-law")
        self.assertAlmostEqual(comp_ml.compression_ratio, 2.0, places=3)
        self.assertAlmostEqual(comp_ml.savings_percent, 50.0, places=2)

        rec_sig = decompress_mu_law(comp_ml)
        ml_err = np.mean((sig.samples - rec_sig.samples) ** 2)
        ml_snr = 10.0 * np.log10(np.mean(sig.samples ** 2) / ml_err)

        pcm_uniform = quantize_pcm(sig, bit_depth=8, v_min=-1.0, v_max=1.0)
        self.assertGreater(ml_snr, pcm_uniform.sqnr_measured_db + 10.0)

        # 3. Casos borde: señal en cero, saturación > 1.0, señal vacía y μ inválido
        zero_sig = Signal1D("z", np.zeros(128, dtype=np.float64), 8000.0, 1 / 8000.0, 128 / 8000.0, True, "Zero")
        rec_zero = decompress_mu_law(compress_mu_law(zero_sig))
        np.testing.assert_array_equal(rec_zero.samples, np.zeros(128, dtype=np.float64))

        sat_sig = Signal1D("s", np.array([-2.5, -1.0, 0.0, 1.0, 2.5]), 8000.0, 1 / 8000.0, 5 / 8000.0, True, "Sat")
        rec_sat = decompress_mu_law(compress_mu_law(sat_sig))
        np.testing.assert_allclose(rec_sat.samples, sat_sig.samples, atol=0.05)

        empty_sig = Signal1D("e", np.zeros(0, dtype=np.float64), 8000.0, 1 / 8000.0, 0.0, True, "Empty")
        self.assertEqual(decompress_mu_law(compress_mu_law(empty_sig)).samples.size, 0)

        with self.assertRaises(ValueError):
            mu_law_compand(grid, mu=0.0)

    def test_dct_2d_orthonormal_and_image_compression_quality(self):
        # 1. Identidad ortonormal DCT-2D <-> IDCT-2D
        rng = np.random.default_rng(99)
        blocks = rng.uniform(-128.0, 127.0, size=(12, 8, 8))
        rec_blocks = dct2d_blocks_inverse(dct2d_blocks_forward(blocks))
        np.testing.assert_allclose(rec_blocks, blocks, atol=1e-11)

        # 2. Imagen suave RGB 64x64: calidad alta vs baja y ahorro real
        yy, xx = np.meshgrid(np.linspace(0, 255, 64), np.linspace(0, 255, 64), indexing="ij")
        rgba = np.zeros((64, 64, 4), dtype=np.uint8)
        rgba[:, :, 0] = np.round(127.5 + 100.0 * np.sin(xx / 10.0)).astype(np.uint8)
        rgba[:, :, 1] = np.round(127.5 + 100.0 * np.cos(yy / 12.0)).astype(np.uint8)
        rgba[:, :, 2] = np.round((xx + yy) * 0.5).astype(np.uint8)
        rgba[:, :, 3] = 255
        smooth_img = ImageMatrix(width=64, height=64, channels=4, pixels=rgba.reshape(-1), label="Smooth")

        c_low = compress_dct_image(smooth_img, quality=20)
        c_high = compress_dct_image(smooth_img, quality=90)
        r_low = decompress_dct_image(c_low)
        r_high = decompress_dct_image(c_high)

        mse_low, psnr_low = compute_image_mse_psnr(smooth_img, r_low)
        mse_high, psnr_high = compute_image_mse_psnr(smooth_img, r_high)

        self.assertGreater(psnr_high, psnr_low)
        self.assertLess(mse_high, mse_low)
        self.assertGreater(psnr_high, 32.0)
        self.assertGreater(c_low.compression_ratio, c_high.compression_ratio)
        self.assertGreater(c_low.compression_ratio, 3.0)

        # 3. Dimensiones no múltiplos de 8 (29x19 RGB y 17x13 escala de grises)
        odd_rgb = ImageMatrix(
            width=29,
            height=19,
            channels=3,
            pixels=rng.integers(0, 256, size=29 * 19 * 3, dtype=np.uint8),
            label="OddRGB",
        )
        rec_odd_rgb = decompress_dct_image(compress_dct_image(odd_rgb, quality=80))
        self.assertEqual((rec_odd_rgb.width, rec_odd_rgb.height, rec_odd_rgb.channels), (29, 19, 3))

        gray = ImageMatrix(
            width=17,
            height=13,
            channels=1,
            pixels=np.tile(np.arange(17, dtype=np.uint8) * 14, 13),
            label="Gray17x13",
        )
        rec_gray = decompress_dct_image(compress_dct_image(gray, quant_matrix=JPEG_LUMINANCE_Q50 * 0.5))
        self.assertEqual((rec_gray.width, rec_gray.height, rec_gray.channels), (17, 13, 1))
        _, psnr_gray = compute_image_mse_psnr(gray, rec_gray)
        self.assertGreater(psnr_gray, 28.0)

        with self.assertRaises(ValueError):
            compress_dct_image(gray, quant_matrix=np.ones((4, 4)))

    def test_dspbin_container_pack_unpack_and_integrity(self):
        img = generate_test_image(24, 24, pattern="target")
        sig = generate_synthetic_signal("sine", amplitude=0.6, frequency_hz=440.0, duration_sec=0.05, fs=8000.0)

        cases = [
            (compress_rle_image(img), "FSK", 8000.0),
            (compress_huffman_image(img), "QPSK", 16000.0),
            (compress_mu_law(sig, mu=255.0), "AM-LC", 8000.0),
            (compress_dct_image(img, quality=75), "BPSK", 24000.0),
        ]

        for comp, mod, fs in cases:
            blob = pack_dspbin(comp, modulation=mod, fs=fs)  # type: ignore[arg-type]
            magic_read = struct.unpack_from(">I", blob, 0)[0]
            self.assertEqual(magic_read, DSPBIN_MAGIC)
            self.assertEqual(blob[:4], b"DSP1")

            pkt = unpack_dspbin(blob)
            self.assertEqual(pkt.magic, DSPBIN_MAGIC)
            self.assertEqual(pkt.modulation, mod)
            self.assertEqual(pkt.fs, fs)
            self.assertEqual(pkt.compressed.codec, comp.codec)
            self.assertEqual(pkt.compressed.original_bytes, comp.original_bytes)
            np.testing.assert_array_equal(pkt.compressed.payload, comp.payload)

            if comp.codec == "rle":
                np.testing.assert_array_equal(decompress_rle_image(pkt.compressed).pixels, img.pixels)
            elif comp.codec == "huffman":
                np.testing.assert_array_equal(decompress_huffman_image(pkt.compressed).pixels, img.pixels)
            elif comp.codec == "mu-law":
                self.assertEqual(decompress_mu_law(pkt.compressed).samples.size, sig.samples.size)

        # Corrupción de CRC32 y cabecera inválida deben fallar explícitamente
        valid_blob = bytearray(pack_dspbin(cases[0][0], modulation="FSK", fs=8000.0))
        corrupt_payload = bytearray(valid_blob)
        corrupt_payload[-1] ^= 0xFF
        with self.assertRaises(ValueError):
            unpack_dspbin(bytes(corrupt_payload))

        corrupt_magic = bytearray(valid_blob)
        corrupt_magic[0] = 0x00
        with self.assertRaises(ValueError):
            unpack_dspbin(bytes(corrupt_magic))

        with self.assertRaises(ValueError):
            unpack_dspbin(bytes(valid_blob[:16]))

    def test_multiformat_exporters_wav_png_svg_bin_json(self):
        # 1. WAV RIFF 16-bit y 8-bit
        sig = generate_synthetic_signal("sine", amplitude=0.8, frequency_hz=500.0, duration_sec=0.05, fs=16000.0)
        wav16 = export_wav_bytes(sig.samples, fs=16000, bit_depth=16)
        self.assertEqual(wav16[:4], b"RIFF")
        self.assertEqual(wav16[8:12], b"WAVE")
        self.assertEqual(wav16[12:16], b"fmt ")
        self.assertEqual(wav16[36:40], b"data")
        rec16, fs16, bd16 = import_wav_bytes(wav16)
        self.assertEqual((fs16, bd16), (16000, 16))
        np.testing.assert_allclose(rec16, sig.samples, atol=1e-4)

        wav8 = export_wav_bytes(sig.samples, fs=8000, bit_depth=8)
        rec8, fs8, bd8 = import_wav_bytes(wav8)
        self.assertEqual((fs8, bd8), (8000, 8))
        np.testing.assert_allclose(rec8, sig.samples, atol=0.02)

        # 2. PNG puro (RGBA, RGB y Escala de grises) sin dependencias externas
        img_rgba = generate_test_image(16, 12, pattern="bars")
        png_bytes = export_png_bytes(img_rgba)
        self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertIn(b"IHDR", png_bytes)
        self.assertIn(b"IDAT", png_bytes)
        self.assertTrue(png_bytes.endswith(b"\x00\x00\x00\x00IEND\xaeB`\x82"))

        # Verificar que IDAT descomprime exactamente las 12 scanlines RGBA + byte de filtro
        idat_idx = png_bytes.index(b"IDAT")
        idat_len = struct.unpack_from(">I", png_bytes, idat_idx - 4)[0]
        idat_raw = zlib.decompress(png_bytes[idat_idx + 4 : idat_idx + 4 + idat_len])
        self.assertEqual(len(idat_raw), 12 * (1 + 16 * 4))

        # 3. SVG vectorial
        svg_str = export_svg_string(img_rgba, pixel_scale=2)
        self.assertIn("<svg", svg_str)
        self.assertIn('width="32"', svg_str)
        self.assertIn('height="24"', svg_str)
        self.assertIn("<rect", svg_str)

        # 4. BIN crudo y empaquetado de bits
        bits = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=np.uint8)
        self.assertEqual(export_bin_bytes(bits, is_bitstream=True), b"\xA5\xF0")
        self.assertEqual(export_bin_bytes(np.array([0x12, 0x34], dtype=np.uint8)), b"\x12\x34")

        # 5. JSON reporte de sesión
        comp = compress_rle_image(img_rgba)
        audit = AuditMetrics(
            mse=0.0,
            rmse=0.0,
            snr_db=99.9,
            psnr_db=99.9,
            ber=0.0,
            bit_errors=0,
            total_bits=1024,
            pearson_correlation=1.0,
        )
        report_str = export_session_report_json(comp, modulation="BPSK", fs=8000.0, metrics=audit)
        parsed = json.loads(report_str)
        self.assertEqual(parsed["compression"]["codec"], "rle")
        self.assertEqual(parsed["transmission"]["modulation"], "BPSK")
        self.assertEqual(parsed["auditMetrics"]["ber"], 0.0)

    def test_mu_law_multibit_depths_and_invalid_inputs(self):
        sig = generate_synthetic_signal("sine", amplitude=0.7, frequency_hz=330.0, duration_sec=0.05, fs=8000.0)
        snrs = []
        for depth in (4, 8, 12, 16):
            comp = compress_mu_law(sig, mu=255.0, compressed_bit_depth=depth, original_bit_depth=16)
            rec = decompress_mu_law(comp)
            self.assertEqual(rec.samples.size, sig.samples.size)
            err = float(np.mean((sig.samples - rec.samples) ** 2))
            snr = 10.0 * np.log10(float(np.mean(sig.samples ** 2)) / max(err, 1e-16))
            snrs.append(snr)

        # Mayor profundidad de bits comprimida debe incrementar estrictamente el SNR
        self.assertGreater(snrs[1], snrs[0] + 15.0)
        self.assertGreater(snrs[2], snrs[1] + 15.0)
        self.assertGreater(snrs[3], snrs[2] + 10.0)

        with self.assertRaises(ValueError):
            compress_mu_law(sig, compressed_bit_depth=1)
        with self.assertRaises(ValueError):
            compress_mu_law(sig, compressed_bit_depth=17)

    def test_png_and_svg_all_channel_modes_plus_stereo_wav(self):
        # PNG 1 canal (Grayscale), 3 canales (RGB) y 4 canales (RGBA con transparencia)
        gray = ImageMatrix(width=8, height=6, channels=1, pixels=np.arange(48, dtype=np.uint8) * 5, label="G")
        rgb = ImageMatrix(width=8, height=6, channels=3, pixels=np.full(8 * 6 * 3, 120, dtype=np.uint8), label="RGB")
        rgba_px = np.zeros((6, 8, 4), dtype=np.uint8)
        rgba_px[:, :, 0] = 200
        rgba_px[:, :, 3] = 128
        rgba = ImageMatrix(width=8, height=6, channels=4, pixels=rgba_px.reshape(-1), label="RGBA")

        for img, expected_ctype in ((gray, 0), (rgb, 2), (rgba, 6)):
            png = export_png_bytes(img)
            ihdr_ctype = png[8 + 8 + 9]  # signature(8) + len/type(8) + width(4)+height(4)+bit_depth(1)
            self.assertEqual(ihdr_ctype, expected_ctype)

            svg = export_svg_string(img, pixel_scale=1)
            self.assertIn(f'width="{img.width}"', svg)
            self.assertIn(f'height="{img.height}"', svg)

        self.assertIn('fill-opacity="0.502"', export_svg_string(rgba))

        # Decodificación de WAV estéreo simulado (promediado a mono)
        fs = 8000
        left = np.full(100, 0.5, dtype=np.float64)
        right = np.full(100, -0.5, dtype=np.float64)
        stereo = np.column_stack([left, right]).reshape(-1)
        pcm = np.round(stereo * 32767.0).astype("<i2").tobytes()
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", 36 + len(pcm), b"WAVE", b"fmt ", 16, 1, 2, fs, fs * 4, 4, 16, b"data", len(pcm)
        )
        mono, rec_fs, rec_bd = import_wav_bytes(header + pcm)
        self.assertEqual((rec_fs, rec_bd, mono.size), (fs, 16, 100))
        np.testing.assert_allclose(mono, np.zeros(100), atol=1e-4)

    def test_dct_long_zero_runs_zrl_and_non_planar_rle(self):
        # Imagen uniforme grande (32x32) fuerza corridas > 16 ceros (marcador ZRL 0xF0 en Zig-Zag)
        uniform = ImageMatrix(width=32, height=32, channels=3, pixels=np.full(32 * 32 * 3, 180, dtype=np.uint8))
        comp_dct = compress_dct_image(uniform, quality=50)
        rec_dct = decompress_dct_image(comp_dct)
        mse, psnr = compute_image_mse_psnr(uniform, rec_dct)
        self.assertLess(mse, 2.0)
        self.assertGreater(psnr, 45.0)
        self.assertGreater(comp_dct.compression_ratio, 10.0)

        # RLE no planar también preserva MSE = 0 exacto
        img = generate_test_image(20, 16, pattern="target")
        rec_np = decompress_rle_image(compress_rle_image(img, planar=False))
        np.testing.assert_array_equal(rec_np.pixels, img.pixels)

        # Validación estricta de códecs cruzados al decodificar
        with self.assertRaises(ValueError):
            decompress_huffman(comp_dct)
        with self.assertRaises(ValueError):
            decompress_rle(comp_dct)
        with self.assertRaises(ValueError):
            decompress_mu_law(comp_dct)
        with self.assertRaises(ValueError):
            decompress_dct_image(compress_rle_image(img))


if __name__ == "__main__":
    unittest.main()
