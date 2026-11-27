import math
import unittest

import numpy as np
from numpy.typing import NDArray

from dsp_engine.contracts import DemodulationOutput, ModulatedSignal
from dsp_engine.core import generate_prbs
from dsp_engine.demodulation import (
    apply_zero_phase_filter,
    bandpass_filter,
    compute_audit_metrics,
    compute_ber,
    compute_mse,
    compute_pearson_correlation,
    compute_psnr_db,
    compute_rmse,
    compute_snr_db,
    demodulate_am_coherent,
    demodulate_am_envelope,
    demodulate_analog,
    demodulate_ask,
    demodulate_baseband,
    demodulate_bpsk,
    demodulate_digital,
    demodulate_fm,
    demodulate_fsk,
    demodulate_qpsk,
    demodulate_signal,
    design_butterworth_bpf,
    design_butterworth_lpf,
    extract_eye_diagram_traces,
    lowpass_filter,
    reconstruct_dac,
    reconstruct_sinc,
    reconstruct_zoh,
)


def _rms(values: NDArray[np.float64]) -> float:
    """Calcula la potencia RMS de un tramo estacionario."""
    return float(np.sqrt(np.mean(values ** 2)))


def _am_lc_signal() -> tuple[ModulatedSignal, NDArray[np.float64]]:
    """Genera AM-LC con índice de modulación 0.8 y mensaje cosmético."""
    fs = 4000.0
    carrier = 200.0
    message_freq = 20.0
    index = 0.8
    t = np.arange(2000, dtype=np.float64) / fs
    message = np.cos(2.0 * np.pi * message_freq * t)
    waveform = (1.0 + index * message) * np.cos(2.0 * np.pi * carrier * t)
    modulated = ModulatedSignal(
        scheme="AM-LC",
        waveform=waveform.astype(np.float64),
        fs=fs,
        carrier_hz=carrier,
        symbol_rate_baud=message_freq,
        samples_per_symbol=200,
        estimated_bandwidth_hz=2.0 * message_freq,
        modulation_index=index,
    )
    return modulated, message


def _am_sc_signal() -> tuple[ModulatedSignal, NDArray[np.float64]]:
    """Genera AM-SC sin portadora residual y con el mismo mensaje cosmético."""
    fs = 4000.0
    carrier = 200.0
    message_freq = 20.0
    index = 0.8
    t = np.arange(2000, dtype=np.float64) / fs
    message = np.cos(2.0 * np.pi * message_freq * t)
    waveform = index * message * np.cos(2.0 * np.pi * carrier * t)
    modulated = ModulatedSignal(
        scheme="AM-SC",
        waveform=waveform.astype(np.float64),
        fs=fs,
        carrier_hz=carrier,
        symbol_rate_baud=message_freq,
        samples_per_symbol=200,
        estimated_bandwidth_hz=2.0 * message_freq,
        modulation_index=index,
    )
    return modulated, message


def _fm_signal() -> tuple[ModulatedSignal, NDArray[np.float64]]:
    """Genera FM con desviación conocida y fase analítica continua."""
    fs = 6000.0
    carrier = 300.0
    message_freq = 20.0
    deviation = 60.0
    t = np.arange(3000, dtype=np.float64) / fs
    message = np.cos(2.0 * np.pi * message_freq * t)
    phase = 2.0 * np.pi * carrier * t + 2.0 * np.pi * (deviation / message_freq) * np.sin(
        2.0 * np.pi * message_freq * t
    )
    modulated = ModulatedSignal(
        scheme="FM",
        waveform=np.cos(phase).astype(np.float64),
        fs=fs,
        carrier_hz=carrier,
        symbol_rate_baud=message_freq,
        samples_per_symbol=300,
        estimated_bandwidth_hz=2.0 * (deviation + message_freq),
        modulation_index=deviation / message_freq,
    )
    return modulated, message


def _baseband_signal(bits: NDArray[np.uint8], coding: str) -> ModulatedSignal:
    """Construye banda base NRZ o Manchester con la convención de transición positiva → negativa para 1."""
    sps = 16
    fs = 1000.0
    rate = fs / sps
    waveform = np.zeros(bits.size * sps, dtype=np.float64)
    for index, bit in enumerate(bits):
        start = index * sps
        stop = start + sps
        if coding == "unipolar-nrz":
            waveform[start:stop] = 1.0 if bit else 0.0
        elif coding == "polar-nrz":
            waveform[start:stop] = 1.0 if bit else -1.0
        else:
            half = sps // 2
            waveform[start : start + half] = 1.0 if bit else -1.0
            waveform[start + half : stop] = -1.0 if bit else 1.0
    return ModulatedSignal(
        scheme="BB",
        waveform=waveform,
        fs=fs,
        carrier_hz=0.0,
        symbol_rate_baud=rate,
        samples_per_symbol=sps,
        estimated_bandwidth_hz=rate,
        line_coding=coding,
    )


def _ask_signal(bits: NDArray[np.uint8]) -> ModulatedSignal:
    """Genera ASK OOK con cuatro ciclos de portadora por símbolo."""
    sps = 40
    fs = 8000.0
    carrier = 800.0
    t = np.arange(bits.size * sps, dtype=np.float64) / fs
    envelope = np.repeat(bits.astype(np.float64), sps)
    return ModulatedSignal(
        scheme="ASK",
        waveform=(envelope * np.cos(2.0 * np.pi * carrier * t)).astype(np.float64),
        fs=fs,
        carrier_hz=carrier,
        symbol_rate_baud=fs / sps,
        samples_per_symbol=sps,
        estimated_bandwidth_hz=fs / sps,
    )


def _fsk_signal(bits: NDArray[np.uint8], low: float, high: float) -> ModulatedSignal:
    """Genera FSK con fase continua entre símbolos."""
    sps = 40
    fs = 8000.0
    t = np.arange(sps, dtype=np.float64) / fs
    chunks: list[NDArray[np.float64]] = []
    phase = 0.0
    for bit in bits:
        frequency = high if bit else low
        chunks.append(np.cos(2.0 * np.pi * frequency * t + phase))
        phase = float(phase + 2.0 * np.pi * frequency * sps / fs)
    return ModulatedSignal(
        scheme="FSK",
        waveform=np.concatenate(chunks).astype(np.float64),
        fs=fs,
        carrier_hz=(low + high) / 2.0,
        symbol_rate_baud=fs / sps,
        samples_per_symbol=sps,
        estimated_bandwidth_hz=high - low,
    )


def _bpsk_signal(bits: NDArray[np.uint8], phase_rad: float = 0.0) -> ModulatedSignal:
    """Genera BPSK con símbolos +/-1 sobre portadora coseno."""
    sps = 40
    fs = 8000.0
    carrier = 800.0
    t = np.arange(bits.size * sps, dtype=np.float64) / fs
    symbols = np.repeat(2.0 * bits.astype(np.float64) - 1.0, sps)
    return ModulatedSignal(
        scheme="BPSK",
        waveform=(symbols * np.cos(2.0 * np.pi * carrier * t + phase_rad)).astype(np.float64),
        fs=fs,
        carrier_hz=carrier,
        symbol_rate_baud=fs / sps,
        samples_per_symbol=sps,
        estimated_bandwidth_hz=fs / sps,
    )


def _qpsk_signal(bits: NDArray[np.uint8]) -> ModulatedSignal:
    """Genera QPSK Gray con i=cos y q=-sin para que el correlador local recupere (I,Q)."""
    sps = 40
    fs = 8000.0
    carrier = 800.0
    pairs = bits.reshape(-1, 2)
    low_bits = pairs[:, 1]
    high_bits = pairs[:, 0]
    indices = np.where(
        low_bits == 0,
        np.where(high_bits == 0, 0, 3),
        np.where(high_bits == 0, 1, 2),
    )
    i_levels = np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    q_levels = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
    amplitude = 1.0 / math.sqrt(2.0)
    i_wave = np.repeat(amplitude * i_levels[indices], sps)
    q_wave = np.repeat(amplitude * q_levels[indices], sps)
    t = np.arange(bits.size // 2 * sps, dtype=np.float64) / fs
    waveform = i_wave * np.cos(2.0 * np.pi * carrier * t) - q_wave * np.sin(2.0 * np.pi * carrier * t)
    return ModulatedSignal(
        scheme="QPSK",
        waveform=waveform.astype(np.float64),
        fs=fs,
        carrier_hz=carrier,
        symbol_rate_baud=fs / sps,
        samples_per_symbol=sps,
        estimated_bandwidth_hz=fs / sps,
    )


def _assert_eye_shape(test_case: unittest.TestCase, output: DemodulationOutput, sps: int) -> None:
    """Verifica que todas las trazas sean listas serializables de 2*Ts muestras."""
    traces = output.eye_diagram_traces
    test_case.assertIsNotNone(traces)
    assert traces is not None
    test_case.assertGreater(len(traces), 0)
    test_case.assertLessEqual(len(traces), 64)
    for trace in traces:
        test_case.assertIsInstance(trace, list)
        test_case.assertEqual(len(trace), 2 * sps)
        test_case.assertTrue(all(isinstance(value, float) for value in trace))


class TestDemodulationFilters(unittest.TestCase):
    """Pruebas de respuesta en frecuencia, fase cero y validación de filtros SOS."""

    def test_lpf_passband_and_stopband_attenuation(self) -> None:
        fs = 1000.0
        t = np.arange(4096, dtype=np.float64) / fs
        pass_input = np.sin(2.0 * np.pi * 40.0 * t)
        stop_input = np.sin(2.0 * np.pi * 350.0 * t)
        pass_band = lowpass_filter(pass_input, 100.0, fs)
        stop_band = lowpass_filter(stop_input, 100.0, fs)
        pass_ratio = _rms(pass_band[500:-500]) / _rms(pass_input[500:-500])
        stop_ratio = _rms(stop_band[500:-500]) / _rms(stop_input[500:-500])
        self.assertGreater(pass_ratio, 0.95)
        self.assertLess(stop_ratio, 0.02)

    def test_bpf_passband_and_stopband_attenuation(self) -> None:
        fs = 2000.0
        t = np.arange(4096, dtype=np.float64) / fs
        center_input = np.sin(2.0 * np.pi * 200.0 * t)
        low_input = np.sin(2.0 * np.pi * 60.0 * t)
        high_input = np.sin(2.0 * np.pi * 700.0 * t)
        center = bandpass_filter(center_input, 180.0, 220.0, fs)
        low_stop = bandpass_filter(low_input, 180.0, 220.0, fs)
        high_stop = bandpass_filter(high_input, 180.0, 220.0, fs)
        center_ratio = _rms(center[500:-500]) / _rms(center_input[500:-500])
        self.assertGreater(center_ratio, 0.95)
        self.assertLess(_rms(low_stop[500:-500]) / _rms(low_input[500:-500]), 0.05)
        self.assertLess(_rms(high_stop[500:-500]) / _rms(high_input[500:-500]), 0.05)

    def test_zero_phase_filter_preserves_time_symmetry(self) -> None:
        n = 512
        t = np.arange(n, dtype=np.float64)
        pulse = np.exp(-0.5 * ((t - (n - 1) / 2.0) / 20.0) ** 2)
        filtered = lowpass_filter(pulse, 200.0, 2000.0)
        np.testing.assert_allclose(filtered, filtered[::-1], atol=1e-9, rtol=1e-6)

        impulse = np.zeros(n, dtype=np.float64)
        impulse[n // 2] = 1.0
        response = lowpass_filter(impulse, 200.0, 2000.0)
        self.assertEqual(int(np.argmax(response)), n // 2)

    def test_filter_validation_rejects_invalid_cutoffs_and_order(self) -> None:
        with self.assertRaises(ValueError):
            design_butterworth_lpf(0.0, 1000.0)
        with self.assertRaises(ValueError):
            design_butterworth_lpf(500.0, 1000.0)
        with self.assertRaises(ValueError):
            design_butterworth_bpf(300.0, 200.0, 1000.0)
        with self.assertRaises(ValueError):
            lowpass_filter(np.ones(32), 100.0, 1000.0, order=0)
        with self.assertRaises(ValueError):
            apply_zero_phase_filter(np.ones(8), np.ones((1, 4)))

    def test_short_and_empty_signals_are_safe(self) -> None:
        sos = design_butterworth_lpf(100.0, 1000.0)
        empty = lowpass_filter(np.empty(0, dtype=np.float64), 100.0, 1000.0)
        self.assertEqual(empty.size, 0)
        self.assertEqual(empty.dtype, np.float64)

        short = np.array([1.0, -1.0, 0.5], dtype=np.float64)
        filtered = apply_zero_phase_filter(short, sos)
        self.assertEqual(filtered.shape, short.shape)
        self.assertTrue(np.all(np.isfinite(filtered)))


class TestDemodulationAnalog(unittest.TestCase):
    """Pruebas de roundtrip sin ruido para AM y FM."""

    def test_am_lc_hilbert_roundtrip(self) -> None:
        modulated, message = _am_lc_signal()
        output = demodulate_am_envelope(modulated, method="hilbert")
        recovered = output.recovered_signal
        assert recovered is not None
        cut = 300
        correlation = compute_pearson_correlation(message[cut:-cut], recovered[cut:-cut])
        self.assertGreater(correlation, 0.95)
        self.assertEqual(output.recovered_bits.dtype, np.uint8)
        self.assertEqual(output.recovered_bits.size, 0)

    def test_am_lc_rectifier_roundtrip(self) -> None:
        modulated, message = _am_lc_signal()
        output = demodulate_am_envelope(modulated, method="rectifier", cutoff_hz=40.0)
        recovered = output.recovered_signal
        assert recovered is not None
        cut = 300
        self.assertGreater(compute_pearson_correlation(message[cut:-cut], recovered[cut:-cut]), 0.95)

    def test_am_sc_coherent_roundtrip(self) -> None:
        modulated, message = _am_sc_signal()
        output = demodulate_am_coherent(modulated)
        recovered = output.recovered_signal
        assert recovered is not None
        cut = 300
        self.assertGreater(compute_pearson_correlation(message[cut:-cut], recovered[cut:-cut]), 0.95)

    def test_am_lc_coherent_removes_dc_component(self) -> None:
        modulated, message = _am_lc_signal()
        output = demodulate_am_coherent(modulated, cutoff_hz=40.0)
        recovered = output.recovered_signal
        assert recovered is not None
        cut = 300
        self.assertLess(abs(float(np.mean(recovered[cut:-cut]))), 0.05)
        self.assertGreater(compute_pearson_correlation(message[cut:-cut], recovered[cut:-cut]), 0.95)

    def test_fm_discriminator_roundtrip(self) -> None:
        modulated, message = _fm_signal()
        output = demodulate_fm(modulated, freq_deviation_hz=60.0)
        recovered = output.recovered_signal
        assert recovered is not None
        cut = 300
        self.assertGreater(compute_pearson_correlation(message[cut:-cut], recovered[cut:-cut]), 0.95)

    def test_analog_dispatchers_return_empty_bits(self) -> None:
        am, _ = _am_lc_signal()
        fm, _ = _fm_signal()
        am_output = demodulate_analog(am, method="hilbert")
        fm_output = demodulate_signal(fm, freq_deviation_hz=60.0)
        for output in (am_output, fm_output):
            self.assertEqual(output.recovered_bits.size, 0)
            self.assertEqual(output.recovered_bits.dtype, np.uint8)
            assert output.recovered_signal is not None
            self.assertEqual(output.recovered_signal.dtype, np.float64)


class TestDemodulationDigital(unittest.TestCase):
    """Pruebas de BER exacto, constelaciones y diagramas de ojo."""

    def test_baseband_unipolar_nrz_roundtrip(self) -> None:
        bits = generate_prbs(24, seed=0x1234)
        output = demodulate_baseband(_baseband_signal(bits, "unipolar-nrz"))
        np.testing.assert_array_equal(output.recovered_bits, bits)
        self.assertEqual(compute_ber(bits, output.recovered_bits), 0.0)
        _assert_eye_shape(self, output, 16)

    def test_baseband_polar_nrz_roundtrip(self) -> None:
        bits = generate_prbs(32, seed=0x4321)
        output = demodulate_digital(_baseband_signal(bits, "polar-nrz"))
        np.testing.assert_array_equal(output.recovered_bits, bits)
        self.assertEqual(compute_ber(bits, output.recovered_bits), 0.0)

    def test_baseband_manchester_roundtrip(self) -> None:
        bits = generate_prbs(20, seed=0xACE1)
        output = demodulate_signal(_baseband_signal(bits, "manchester"))
        np.testing.assert_array_equal(output.recovered_bits, bits)
        self.assertEqual(compute_ber(bits, output.recovered_bits), 0.0)
        _assert_eye_shape(self, output, 16)

    def test_eye_diagram_trace_shape_and_limit(self) -> None:
        signal = np.sin(np.arange(1000, dtype=np.float64) / 10.0)
        traces = extract_eye_diagram_traces(signal, 16, max_traces=5)
        self.assertEqual(len(traces), 5)
        self.assertTrue(all(len(trace) == 32 for trace in traces))
        self.assertTrue(all(isinstance(value, float) for trace in traces for value in trace))
        self.assertEqual(extract_eye_diagram_traces(signal[:10], 16), [])

    def test_ask_roundtrip_with_adaptive_threshold(self) -> None:
        bits = generate_prbs(32, seed=0x005A)
        output = demodulate_ask(_ask_signal(bits))
        np.testing.assert_array_equal(output.recovered_bits, bits)
        self.assertEqual(compute_ber(bits, output.recovered_bits), 0.0)
        assert output.rx_constellation is not None
        self.assertEqual(len(output.rx_constellation), bits.size)
        self.assertTrue(all(q == 0.0 for _, q in output.rx_constellation))
        _assert_eye_shape(self, output, 40)

    def test_ask_explicit_threshold_and_constellation(self) -> None:
        bits = generate_prbs(16, seed=0x00FF)
        output = demodulate_digital(_ask_signal(bits), threshold=0.5)
        np.testing.assert_array_equal(output.recovered_bits, bits)
        assert output.recovered_signal is not None
        self.assertEqual(output.recovered_signal.dtype, np.float64)
        assert output.rx_constellation is not None
        self.assertEqual(output.rx_constellation[0][1], 0.0)

    def test_fsk_roundtrip_with_explicit_frequencies(self) -> None:
        bits = generate_prbs(32, seed=0x0F0F)
        modulated = _fsk_signal(bits, 800.0, 1200.0)
        output = demodulate_fsk(modulated, f0_hz=800.0, f1_hz=1200.0)
        np.testing.assert_array_equal(output.recovered_bits, bits)
        self.assertEqual(compute_ber(bits, output.recovered_bits), 0.0)
        assert output.rx_constellation is not None
        self.assertEqual(len(output.rx_constellation), bits.size)
        _assert_eye_shape(self, output, 40)

    def test_fsk_roundtrip_with_inferred_frequencies(self) -> None:
        bits = generate_prbs(24, seed=0x3333)
        modulated = _fsk_signal(bits, 900.0, 1100.0)
        output = demodulate_digital(modulated)
        np.testing.assert_array_equal(output.recovered_bits, bits)
        self.assertEqual(compute_ber(bits, output.recovered_bits), 0.0)

    def test_bpsk_roundtrip_and_constellation(self) -> None:
        bits = generate_prbs(40, seed=0x7777)
        output = demodulate_bpsk(_bpsk_signal(bits))
        np.testing.assert_array_equal(output.recovered_bits, bits)
        self.assertEqual(compute_ber(bits, output.recovered_bits), 0.0)
        assert output.rx_constellation is not None
        self.assertEqual(len(output.rx_constellation), bits.size)
        self.assertTrue(all(i >= 0.0 for i, _ in output.rx_constellation) == all(bits == 1))
        _assert_eye_shape(self, output, 40)

    def test_bpsk_phase_offset_with_local_oscillator(self) -> None:
        bits = generate_prbs(24, seed=0x2222)
        modulated = _bpsk_signal(bits, phase_rad=0.4)
        output = demodulate_digital(modulated, phase_rad=0.4)
        np.testing.assert_array_equal(output.recovered_bits, bits)

    def test_qpsk_gray_roundtrip(self) -> None:
        bits = generate_prbs(32, seed=0x9999)
        output = demodulate_qpsk(_qpsk_signal(bits))
        np.testing.assert_array_equal(output.recovered_bits, bits)
        self.assertEqual(compute_ber(bits, output.recovered_bits), 0.0)
        assert output.rx_constellation is not None
        self.assertEqual(len(output.rx_constellation), bits.size // 2)
        _assert_eye_shape(self, output, 40)

    def test_qpsk_ideal_constellation_roundtrip(self) -> None:
        bits = generate_prbs(24, seed=0xABCD)
        amplitude = 1.0 / math.sqrt(2.0)
        modulated = _qpsk_signal(bits)
        ideal = [
            (amplitude, amplitude),
            (-amplitude, amplitude),
            (-amplitude, -amplitude),
            (amplitude, -amplitude),
        ]
        modulated = ModulatedSignal(
            scheme=modulated.scheme,
            waveform=modulated.waveform,
            fs=modulated.fs,
            carrier_hz=modulated.carrier_hz,
            symbol_rate_baud=modulated.symbol_rate_baud,
            samples_per_symbol=modulated.samples_per_symbol,
            estimated_bandwidth_hz=modulated.estimated_bandwidth_hz,
            ideal_constellation=ideal,
        )
        output = demodulate_signal(modulated)
        np.testing.assert_array_equal(output.recovered_bits, bits)

    def test_unsupported_scheme_is_rejected(self) -> None:
        modulated = ModulatedSignal(
            scheme="OFDM",
            waveform=np.ones(32, dtype=np.float64),
            fs=1000.0,
            carrier_hz=100.0,
            symbol_rate_baud=100.0,
            samples_per_symbol=8,
            estimated_bandwidth_hz=100.0,
        )
        with self.assertRaises(ValueError):
            demodulate_signal(modulated)


class TestDemodulationReconstruction(unittest.TestCase):
    """Pruebas de interpolación sinc, retención ZOH y contrato Signal1D."""

    def test_sinc_reconstruction_of_bandlimited_sine(self) -> None:
        fs_in = 800.0
        fs_out = 8000.0
        t_in = np.arange(400, dtype=np.float64) / fs_in
        samples = np.sin(2.0 * np.pi * 100.0 * t_in)
        reconstructed = reconstruct_sinc(samples, fs_in, fs_out)
        t_out = np.arange(reconstructed.size, dtype=np.float64) / fs_out
        reference = np.sin(2.0 * np.pi * 100.0 * t_out)
        cut = 200
        self.assertGreater(compute_pearson_correlation(reference[cut:-cut], reconstructed[cut:-cut]), 0.999)
        self.assertGreater(compute_snr_db(reference[cut:-cut], reconstructed[cut:-cut]), 30.0)

    def test_zoh_reconstruction_of_bandlimited_sine(self) -> None:
        fs_in = 800.0
        fs_out = 8000.0
        t_in = np.arange(400, dtype=np.float64) / fs_in
        samples = np.sin(2.0 * np.pi * 100.0 * t_in)
        reconstructed = reconstruct_zoh(samples, fs_in, fs_out)
        t_out = np.arange(reconstructed.size, dtype=np.float64) / fs_out
        reference = np.sin(2.0 * np.pi * 100.0 * t_out)
        cut = 200
        self.assertGreater(compute_pearson_correlation(reference[cut:-cut], reconstructed[cut:-cut]), 0.99)
        self.assertGreater(compute_snr_db(reference[cut:-cut], reconstructed[cut:-cut]), 15.0)

    def test_reconstruct_dac_contract_and_target_length(self) -> None:
        samples = np.sin(np.arange(64, dtype=np.float64) / 4.0)
        dac = reconstruct_dac(samples, 1000.0, 8000.0, method="sinc", target_length=321)
        self.assertEqual(dac.samples.size, 321)
        self.assertEqual(dac.samples.dtype, np.float64)
        self.assertFalse(dac.is_discrete)
        self.assertAlmostEqual(dac.fs, 8000.0)
        self.assertAlmostEqual(dac.ts, 1.0 / 8000.0)
        self.assertAlmostEqual(dac.duration, 321 / 8000.0)
        self.assertEqual(dac.metadata["method"], "sinc")

    def test_reconstruction_empty_and_explicit_cutoff(self) -> None:
        empty = reconstruct_sinc(np.empty(0, dtype=np.float64), 1000.0, 8000.0)
        self.assertEqual(empty.size, 0)
        samples = np.sin(np.arange(64, dtype=np.float64) / 4.0)
        zoh = reconstruct_dac(samples, 1000.0, 8000.0, method="zoh", cutoff_hz=200.0)
        self.assertEqual(zoh.samples.dtype, np.float64)
        self.assertTrue(np.all(np.isfinite(zoh.samples)))


class TestDemodulationMetrics(unittest.TestCase):
    """Pruebas de MSE, SNR, PSNR, BER, Pearson y agregación AuditMetrics."""

    def test_mse_and_rmse_exact_values(self) -> None:
        tx = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        rx = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        self.assertAlmostEqual(compute_mse(tx, rx), 5.0 / 3.0, places=12)
        self.assertAlmostEqual(compute_rmse(tx, rx), math.sqrt(5.0 / 3.0), places=12)

    def test_snr_db_and_psnr_db_identical_and_zero_signals(self) -> None:
        tx = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        zeros = np.zeros(3, dtype=np.float64)
        self.assertEqual(compute_snr_db(tx, tx.copy()), 120.0)
        self.assertEqual(compute_psnr_db(tx, tx.copy()), 120.0)
        self.assertEqual(compute_snr_db(zeros, zeros.copy()), 120.0)
        self.assertEqual(compute_snr_db(zeros, np.ones(3)), -120.0)
        self.assertEqual(compute_psnr_db(zeros, np.ones(3)), 0.0)

    def test_psnr_db_uses_peak_value(self) -> None:
        tx = np.ones(4, dtype=np.float64)
        rx = np.zeros(4, dtype=np.float64)
        self.assertAlmostEqual(compute_psnr_db(tx, rx, peak_value=2.0), 10.0 * math.log10(4.0), places=12)

    def test_ber_known_errors_and_length_mismatch(self) -> None:
        tx = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        rx = np.array([1, 0, 0, 0, 0, 0, 1, 0], dtype=np.uint8)
        self.assertEqual(compute_ber(tx, rx), 0.25)
        self.assertEqual(compute_ber(tx, rx[:6]), 1.0 / 3.0)
        self.assertEqual(compute_ber(np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.uint8)), 0.0)

    def test_pearson_constant_and_inverted_signals(self) -> None:
        constant = np.full(8, 2.0, dtype=np.float64)
        other = np.full(8, -3.0, dtype=np.float64)
        signal = np.arange(8, dtype=np.float64)
        self.assertEqual(compute_pearson_correlation(constant, constant.copy()), 1.0)
        self.assertEqual(compute_pearson_correlation(constant, other), 0.0)
        self.assertAlmostEqual(compute_pearson_correlation(signal, -signal), -1.0, places=12)

    def test_compute_audit_metrics_integrates_all_fields(self) -> None:
        tx = np.sin(np.arange(64, dtype=np.float64) / 5.0)
        rx = 0.9 * tx + 0.01
        tx_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        rx_bits = np.array([1, 0, 1, 0, 0, 0, 1, 0], dtype=np.uint8)
        metrics = compute_audit_metrics(tx, rx, tx_bits, rx_bits)
        self.assertAlmostEqual(metrics.mse, compute_mse(tx, rx), places=12)
        self.assertAlmostEqual(metrics.rmse, compute_rmse(tx, rx), places=12)
        self.assertAlmostEqual(metrics.ber, 0.125, places=12)
        self.assertEqual(metrics.bit_errors, 1)
        self.assertEqual(metrics.total_bits, 8)
        self.assertGreater(metrics.pearson_correlation, 0.999)
        self.assertTrue(math.isfinite(metrics.snr_db))
        self.assertTrue(math.isfinite(metrics.psnr_db))

    def test_metrics_never_return_nan_with_missing_inputs(self) -> None:
        metrics = compute_audit_metrics()
        self.assertEqual(metrics.mse, 0.0)
        self.assertEqual(metrics.rmse, 0.0)
        self.assertEqual(metrics.snr_db, 0.0)
        self.assertEqual(metrics.psnr_db, 0.0)
        self.assertEqual(metrics.ber, 0.0)
        self.assertEqual(metrics.bit_errors, 0)
        self.assertEqual(metrics.total_bits, 0)
        self.assertTrue(math.isfinite(metrics.pearson_correlation))


if __name__ == "__main__":
    unittest.main()
