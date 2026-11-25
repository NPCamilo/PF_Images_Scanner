import unittest
import numpy as np

from dsp_engine.bridge import (
    export_wav_base64,
    run_audio_signal_scan,
    run_bitstream_scan,
    run_core_pipeline_demo,
    run_cross_pipeline,
    run_image_scan,
)
from dsp_engine.core import (
    bitstream_to_image,
    compute_alias_frequency,
    compute_spectrogram,
    compute_spectrum,
    decode_wav_bytes,
    dequantize_bitstream,
    encode_wav_bytes,
    extract_red_dominant_mask,
    generate_bit_trace_waveform,
    generate_prbs,
    generate_synthetic_signal,
    generate_test_image,
    hex_to_bits,
    image_to_bitstream,
    pack_bits_to_bytes,
    quantize_pcm,
    simulate_sampling_nqs,
    text_to_bits,
    unpack_bytes_to_bits,
)


class TestCoreDspPhase1And2(unittest.TestCase):
    """Pruebas matemáticas rigurosas del núcleo DSP en Python (Fases 1 y 2)."""

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
        aliasing = simulate_sampling_nqs(sig, downsample_factor=8, known_max_freq_hz=400.0)
        self.assertEqual(aliasing.effective_fs, 500.0)
        self.assertEqual(aliasing.nyquist_minimum_hz, 800.0)
        self.assertFalse(aliasing.nyquist_satisfied)
        self.assertEqual(aliasing.alias_frequency_hz, 100.0)
        self.assertEqual(compute_alias_frequency(400.0, 500.0), 100.0)

        valid = simulate_sampling_nqs(sig, downsample_factor=4, known_max_freq_hz=400.0)
        self.assertTrue(valid.nyquist_satisfied)
        self.assertIsNone(valid.alias_frequency_hz)

    def test_pcm_quantization_sqnr_and_byte_bit_packing(self):
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

    def test_stft_spectrogram_tracks_chirp_frequency(self):
        chirp = generate_synthetic_signal(
            waveform="chirp",
            frequency_hz=150.0,
            secondary_frequency_hz=1200.0,
            duration_sec=0.5,
            fs=4000.0,
        )
        stft = compute_spectrogram(chirp, window_size=256, hop_size=64, window="hann")
        self.assertGreater(stft["num_frames"], 10)
        first_peak_hz = float(stft["frequencies"][int(np.argmax(stft["power_db"][0]))])
        last_peak_hz = float(stft["frequencies"][int(np.argmax(stft["power_db"][-1]))])
        self.assertLess(first_peak_hz, last_peak_hz)

    def test_wav_riff_encode_decode_roundtrip(self):
        sig = generate_synthetic_signal(waveform="sine", frequency_hz=440.0, duration_sec=0.1, fs=8000.0)
        wav_bytes = encode_wav_bytes(sig.samples, fs=8000, bit_depth=16)
        decoded_samples, decoded_fs = decode_wav_bytes(wav_bytes)
        self.assertEqual(decoded_fs, 8000)
        self.assertEqual(len(decoded_samples), len(sig.samples))
        mse = float(np.mean((sig.samples - decoded_samples) ** 2))
        self.assertLess(mse, 1e-8)

    def test_baseband_line_coding_and_text_hex_bit_trace(self):
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        manchester = generate_bit_trace_waveform(bits, samples_per_bit=16, coding="manchester")
        self.assertEqual(len(manchester), 128)
        self.assertAlmostEqual(float(np.mean(manchester)), 0.0, places=10)

        self.assertEqual(len(text_to_bits("UTP")), 24)
        self.assertEqual(len(hex_to_bits("A5 5A")), 16)

    def test_bridge_endpoints_end_to_end(self):
        audio_out = run_audio_signal_scan({"waveform": "sine", "frequencyHz": 300, "fs": 8000, "durationSec": 0.05})
        self.assertIn("spectrogram", audio_out)
        self.assertIn("bitTraceWaveform", audio_out["quantization"])

        img_out = run_image_scan({"width": 32, "height": 32, "pattern": "target", "bitDepth": 8})
        self.assertGreater(img_out["metrics"]["psnrDb"], 90.0)

        bit_out = run_bitstream_scan({"source": "text", "text": "UTP", "lineCoding": "manchester"})
        self.assertEqual(bit_out["totalBits"], 24)
        self.assertAlmostEqual(bit_out["dcMean"], 0.0, places=6)

        b64 = export_wav_base64({"samples": [0.0, 0.5, -0.5, 0.0], "fs": 8000, "bitDepth": 16})
        self.assertTrue(len(b64) > 20)
        self.assertEqual(run_core_pipeline_demo({"frequencyHz": 220})["quantization"]["bitDepth"], 8)

    def test_cross_pipeline_zero_ber_and_awgn_degradation(self):
        for scheme in ("BPSK", "FSK", "ASK"):
            clean = run_cross_pipeline(
                {
                    "width": 16,
                    "height": 16,
                    "pattern": "bars",
                    "bitDepth": 8,
                    "useCompression": True,
                    "modulation": scheme,
                    "snrDb": 60.0,
                }
            )
            self.assertEqual(clean["metrics"]["ber"], 0.0)
            self.assertEqual(clean["metrics"]["bitErrors"], 0)
            self.assertAlmostEqual(clean["metrics"]["pearsonCorrelation"], 1.0, places=5)
            self.assertGreaterEqual(clean["compression"]["compressionRatio"], 1.5)

        noisy = run_cross_pipeline(
            {
                "width": 16,
                "height": 16,
                "pattern": "target",
                "bitDepth": 4,
                "useCompression": False,
                "modulation": "BPSK",
                "snrDb": -8.0,
            }
        )
        self.assertGreater(noisy["metrics"]["ber"], 0.0)
        self.assertGreater(len(noisy["demodulation"]["eyeTraces"]), 5)


if __name__ == "__main__":
    unittest.main()
