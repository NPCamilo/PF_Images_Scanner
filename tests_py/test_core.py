import unittest
import numpy as np

from dsp_engine.bridge import run_core_pipeline_demo
from dsp_engine.core import (
    bitstream_to_image,
    compute_alias_frequency,
    compute_spectrum,
    dequantize_bitstream,
    extract_red_dominant_mask,
    generate_prbs,
    generate_synthetic_signal,
    generate_test_image,
    image_to_bitstream,
    pack_bits_to_bytes,
    quantize_pcm,
    simulate_sampling_nqs,
    unpack_bytes_to_bits,
)


class TestCoreDspPhase1(unittest.TestCase):
    """Pruebas matemáticas rigurosas del núcleo DSP en Python (Doubt-Driven Development)."""

    def test_fft_dominant_frequency_and_parseval(self):
        fs = 1024.0
        freq = 100.0
        sig = generate_synthetic_signal(
            waveform="sine",
            amplitude=1.0,
            frequency_hz=freq,
            duration_sec=1.0,
            fs=fs,
        )
        spec = compute_spectrum(sig, window="rectangular")
        self.assertLess(abs(spec.dominant_frequency_hz - freq), 1.5)

        # Identidad de Parseval (potencia media temporal ≈ potencia espectral unilateral)
        time_power = float(np.mean(sig.samples ** 2))
        freq_power = float(spec.magnitudes[0] ** 2 + np.sum(0.5 * (spec.magnitudes[1:-1] ** 2)))
        self.assertLess(abs(freq_power - time_power), 0.02)

    def test_nqs_sampling_and_exact_alias_frequency(self):
        sig = generate_synthetic_signal(
            waveform="sine",
            amplitude=1.0,
            frequency_hz=400.0,
            duration_sec=0.25,
            fs=4000.0,
        )
        # Submuestreo por 8 -> Fs_eff = 500 Hz (< 800 Hz Nyquist)
        aliasing = simulate_sampling_nqs(sig, downsample_factor=8, known_max_freq_hz=400.0)
        self.assertEqual(aliasing.effective_fs, 500.0)
        self.assertEqual(aliasing.nyquist_minimum_hz, 800.0)
        self.assertFalse(aliasing.nyquist_satisfied)
        self.assertEqual(aliasing.alias_frequency_hz, 100.0)
        self.assertEqual(compute_alias_frequency(400.0, 500.0), 100.0)

        # Submuestreo por 4 -> Fs_eff = 1000 Hz (>= 800 Hz)
        valid = simulate_sampling_nqs(sig, downsample_factor=4, known_max_freq_hz=400.0)
        self.assertTrue(valid.nyquist_satisfied)
        self.assertIsNone(valid.alias_frequency_hz)

    def test_pcm_quantization_sqnr_and_byte_bit_packing(self):
        # Frecuencia no coherente (53.7 Hz @ 4000 Hz) para cumplir hipótesis uniforme de Bennett
        sig = generate_synthetic_signal(
            waveform="sine",
            amplitude=1.0,
            frequency_hz=53.7,
            phase_rad=0.13,
            duration_sec=0.5,
            fs=4000.0,
        )
        pcm8 = quantize_pcm(sig, bit_depth=8, v_min=-1.0, v_max=1.0)
        self.assertAlmostEqual(pcm8.sqnr_theoretical_db, 49.92, places=2)
        self.assertLess(abs(pcm8.sqnr_measured_db - pcm8.sqnr_theoretical_db), 0.5)
        self.assertEqual(len(pcm8.bitstream), len(sig.samples) * 8)

        unpacked = unpack_bytes_to_bits(pack_bits_to_bytes(pcm8.bitstream), len(pcm8.bitstream))
        np.testing.assert_array_equal(unpacked, pcm8.bitstream)

        decoded = dequantize_bitstream(pcm8.bitstream, bit_depth=8, v_min=-1.0, v_max=1.0)
        np.testing.assert_allclose(decoded, pcm8.quantized, atol=1e-10)

        pcm1 = quantize_pcm(sig, bit_depth=1)
        self.assertEqual(pcm1.num_levels, 2)
        self.assertEqual(len(pcm1.bitstream), len(sig.samples))

    def test_prbs_and_image_bitstream_roundtrip(self):
        prbs_a = generate_prbs(128, seed=0x1234)
        prbs_b = generate_prbs(128, seed=0x1234)
        np.testing.assert_array_equal(prbs_a, prbs_b)

        img = generate_test_image(16, 16, pattern="bars")
        bits = image_to_bitstream(img, bit_depth=8)
        self.assertEqual(len(bits), 16 * 16 * 3 * 8)

        recon = bitstream_to_image(bits, 16, 16, source_channels=3, bit_depth=8)
        np.testing.assert_array_equal(recon.pixels, img.pixels)

        target = generate_test_image(32, 32, pattern="target")
        _, red_ratio = extract_red_dominant_mask(target)
        self.assertGreater(red_ratio, 0.05)

    def test_pyodide_bridge_smoke(self):
        out = run_core_pipeline_demo(
            {"waveform": "sine", "frequencyHz": 220, "fs": 8000, "durationSec": 0.05, "bitDepth": 8}
        )
        self.assertEqual(out["quantization"]["bitDepth"], 8)
        self.assertTrue(out["nqs"]["nyquistSatisfied"])
        self.assertGreater(len(out["spectrum"]["frequencies"]), 10)


if __name__ == "__main__":
    unittest.main()
