"""
Suite de Pruebas Matemáticas Unitarias — Modulación y Codificación de Línea
Responsable: Ángel Gabriel Burbano (tests_py/test_modulation.py)
"""

import unittest
import numpy as np

from dsp_engine.core import compute_spectrum, generate_prbs, generate_synthetic_signal
from dsp_engine.contracts import Signal1D
from dsp_engine.modulation import (
    compute_carson_bandwidth,
    encode_baseband,
    generate_symbol_clock,
    modulate_am_lc,
    modulate_am_sc,
    modulate_ask,
    modulate_bpsk,
    modulate_fm,
    modulate_fsk,
    modulate_qpsk,
)


class TestBasebandLineCoding(unittest.TestCase):
    """Pruebas unitarias matemáticas para RF-08 (Codificación en Banda Base)."""

    def test_unipolar_nrz_levels_and_dc_component(self):
        bits = [1, 0, 1, 0, 1, 1, 0, 0]
        sps = 16
        rb = 1000.0
        amp = 2.0

        sig = encode_baseband(
            bits=bits,
            line_coding="unipolar-nrz",
            symbol_rate_baud=rb,
            samples_per_symbol=sps,
            amplitude=amp,
        )

        self.assertEqual(sig.scheme, "BB")
        self.assertEqual(sig.carrier_hz, 0.0)
        self.assertEqual(sig.line_coding, "unipolar-nrz")
        self.assertEqual(sig.symbol_rate_baud, rb)
        self.assertEqual(sig.samples_per_symbol, sps)
        self.assertEqual(sig.fs, rb * sps)
        self.assertEqual(sig.estimated_bandwidth_hz, rb)
        self.assertEqual(len(sig.waveform), len(bits) * sps)

        # Niveles válidos: solo 0.0 y +amp
        unique_levels = np.unique(sig.waveform)
        np.testing.assert_array_almost_equal(unique_levels, [0.0, amp])

        # Verificación de valor medio (DC) para secuencia equilibrada (50% unos, 50% ceros): DC = amp / 2
        self.assertAlmostEqual(float(np.mean(sig.waveform)), amp / 2.0, places=6)

        # Constelación ideal de decisión
        self.assertEqual(sig.ideal_constellation, [(0.0, 0.0), (amp, 0.0)])

    def test_polar_nrz_levels_and_bipolarity(self):
        bits = [1, 0, 1, 0]
        sps = 20
        rb = 500.0
        amp = 1.5

        sig = encode_baseband(
            bits=bits,
            line_coding="polar-nrz",
            symbol_rate_baud=rb,
            samples_per_symbol=sps,
            amplitude=amp,
        )

        self.assertEqual(sig.scheme, "BB")
        self.assertEqual(sig.line_coding, "polar-nrz")
        self.assertEqual(len(sig.waveform), len(bits) * sps)

        # Niveles polares simétricos: -amp y +amp
        unique_levels = np.unique(sig.waveform)
        np.testing.assert_array_almost_equal(unique_levels, [-amp, amp])

        # Media nula para secuencia equilibrada
        self.assertAlmostEqual(float(np.mean(sig.waveform)), 0.0, places=6)

        # Constelación ideal simétrica
        self.assertEqual(sig.ideal_constellation, [(-amp, 0.0), (amp, 0.0)])

    def test_manchester_midbit_transition_and_dc_rejection(self):
        bits = [1, 0]
        sps = 16
        amp = 1.0

        sig = encode_baseband(
            bits=bits,
            line_coding="manchester",
            symbol_rate_baud=1000.0,
            samples_per_symbol=sps,
            amplitude=amp,
        )

        self.assertEqual(sig.scheme, "BB")
        self.assertEqual(sig.line_coding, "manchester")
        self.assertEqual(sig.estimated_bandwidth_hz, 2000.0)

        # Bit 1 (primer símbolo, muestras 0..15):
        # Primer semi-intervalo (0..7) = +amp, Segundo semi-intervalo (8..15) = -amp
        first_half_bit1 = sig.waveform[0:8]
        second_half_bit1 = sig.waveform[8:16]
        np.testing.assert_array_almost_equal(first_half_bit1, np.full(8, amp))
        np.testing.assert_array_almost_equal(second_half_bit1, np.full(8, -amp))

        # Bit 0 (segundo símbolo, muestras 16..31):
        # Primer semi-intervalo (16..23) = -amp, Segundo semi-intervalo (24..31) = +amp
        first_half_bit0 = sig.waveform[16:24]
        second_half_bit0 = sig.waveform[24:32]
        np.testing.assert_array_almost_equal(first_half_bit0, np.full(8, -amp))
        np.testing.assert_array_almost_equal(second_half_bit0, np.full(8, amp))

        # Invariante clave de Manchester: componente DC = 0 incluso en secuencias puras de unos
        all_ones = encode_baseband([1, 1, 1, 1], line_coding="manchester", samples_per_symbol=16)
        self.assertAlmostEqual(float(np.mean(all_ones.waveform)), 0.0, places=10)

    def test_manchester_requires_even_samples_per_symbol(self):
        with self.assertRaises(ValueError):
            encode_baseband([1, 0, 1], line_coding="manchester", samples_per_symbol=15)

    def test_symbol_clock_synchronism_and_duty_cycle(self):
        num_symbols = 8
        sps = 24
        clk = generate_symbol_clock(num_symbols=num_symbols, samples_per_symbol=sps)

        self.assertEqual(len(clk), num_symbols * sps)
        # Cada ciclo de reloj debe tener 12 muestras en 1.0 y 12 muestras en 0.0
        first_cycle = clk[0:sps]
        self.assertEqual(int(np.sum(first_cycle == 1.0)), 12)
        self.assertEqual(int(np.sum(first_cycle == 0.0)), 12)

    def test_manchester_spectral_null_at_dc(self):
        # Usar PRBS para evaluar el espectro de potencia
        prbs_bits = generate_prbs(256, seed=0xCAFE)
        manchester_sig = encode_baseband(
            prbs_bits,
            line_coding="manchester",
            symbol_rate_baud=1000.0,
            samples_per_symbol=16,
        )

        from dsp_engine.contracts import Signal1D
        sig1d = Signal1D(
            id="manchester-test",
            samples=manchester_sig.waveform,
            fs=manchester_sig.fs,
            ts=1.0 / manchester_sig.fs,
            duration=len(manchester_sig.waveform) / manchester_sig.fs,
            is_discrete=False,
            label="Manchester PRBS",
        )
        spec = compute_spectrum(sig1d, window="rectangular")

        # La magnitud en frecuencia f=0 Hz (DC) debe ser prácticamente nula
        dc_magnitude = float(spec.magnitudes[0])
        max_magnitude = float(np.max(spec.magnitudes))
        self.assertLess(dc_magnitude / max_magnitude, 1e-4)

    def test_input_validations_and_edge_cases(self):
        # Bitstream vacío
        with self.assertRaises(ValueError):
            encode_baseband([], line_coding="polar-nrz")

        # Bits no binarios
        with self.assertRaises(ValueError):
            encode_baseband([0, 1, 2, 0], line_coding="polar-nrz")

        # Tasa de símbolos no positiva
        with self.assertRaises(ValueError):
            encode_baseband([1, 0], symbol_rate_baud=-10.0)

        # Muestras por símbolo menores a 2
        with self.assertRaises(ValueError):
            encode_baseband([1, 0], samples_per_symbol=1)

        # Amplitud no positiva
        with self.assertRaises(ValueError):
            encode_baseband([1, 0], amplitude=0.0)

        # Esquema desconocido
        with self.assertRaises(ValueError):
            encode_baseband([1, 0], line_coding="invalid-coding")  # type: ignore



class TestAnalogModulation(unittest.TestCase):
    """Pruebas unitarias matemáticas para RF-09 (Modulación CW Analógica: AM-LC, AM-SC, FM)."""

    def test_carson_bandwidth_calculation(self):
        # FM Comercial: Δf = 75 kHz, fm = 15 kHz -> BW = 2 * (75 + 15) = 180 kHz
        bw = compute_carson_bandwidth(frequency_deviation_hz=75000.0, message_f_max_hz=15000.0)
        self.assertEqual(bw, 180000.0)

        with self.assertRaises(ValueError):
            compute_carson_bandwidth(0.0, 1000.0)
        with self.assertRaises(ValueError):
            compute_carson_bandwidth(500.0, -10.0)

    def test_am_lc_modulation_and_spectral_translation(self):
        # Usar Fs potencia de 2 y frecuencias armónicas para evitar leakage por zero-padding en FFT
        fs = 8192.0
        fm = 128.0
        fc = 1024.0
        m = 0.5
        ac = 2.0

        msg = generate_synthetic_signal(
            waveform="cosine",
            amplitude=1.0,
            frequency_hz=fm,
            duration_sec=1.0,
            fs=fs,
        )

        mod = modulate_am_lc(
            signal=msg,
            carrier_hz=fc,
            modulation_index=m,
            carrier_amplitude=ac,
        )

        self.assertEqual(mod.scheme, "AM-LC")
        self.assertEqual(mod.carrier_hz, fc)
        self.assertEqual(mod.modulation_index, m)
        self.assertEqual(mod.estimated_bandwidth_hz, 2.0 * fm)
        self.assertEqual(len(mod.waveform), len(msg.samples))

        # Envolvente: siempre positiva cuando m <= 1.0 (sin inversión de fase)
        # Ac * (1 - m) <= s(t) envelope <= Ac * (1 + m) -> min = 1.0, max = 3.0
        peak_envelope = np.max(np.abs(mod.waveform))
        self.assertAlmostEqual(peak_envelope, ac * (1.0 + m), delta=0.05)

        # Verificación espectral mediante FFT
        sig_mod = Signal1D(
            id="am-lc-fft",
            samples=mod.waveform,
            fs=fs,
            ts=1.0 / fs,
            duration=1.0,
            is_discrete=False,
            label="AM-LC Test",
        )
        spec = compute_spectrum(sig_mod, window="rectangular")

        # Frecuencia dominante en fc (portadora fuerte)
        self.assertLess(abs(spec.dominant_frequency_hz - fc), 2.0)

        # Localizar magnitudes de portadora (fc) y bandas laterales (fc - fm, fc + fm)
        idx_carrier = int(round(fc * spec.fft_size / fs))
        idx_lsb = int(round((fc - fm) * spec.fft_size / fs))
        idx_usb = int(round((fc + fm) * spec.fft_size / fs))

        mag_carrier = spec.magnitudes[idx_carrier]
        mag_lsb = spec.magnitudes[idx_lsb]
        mag_usb = spec.magnitudes[idx_usb]

        # Relación teórica: mag_lsb / mag_carrier ≈ m / 2 = 0.25
        self.assertAlmostEqual(mag_lsb / mag_carrier, m / 2.0, delta=0.02)
        self.assertAlmostEqual(mag_usb / mag_carrier, m / 2.0, delta=0.02)

        # En banda base (fm = 128 Hz), la energía debe estar suprimida (traslación exitosa a fc)
        idx_baseband = int(round(fm * spec.fft_size / fs))
        self.assertLess(spec.magnitudes[idx_baseband] / mag_carrier, 0.01)

    def test_am_sc_carrier_suppression(self):
        fs = 8000.0
        fm = 100.0
        fc = 1000.0
        ac = 1.0

        msg = generate_synthetic_signal(
            waveform="cosine",
            amplitude=1.0,
            frequency_hz=fm,
            duration_sec=1.0,
            fs=fs,
        )

        mod = modulate_am_sc(
            signal=msg,
            carrier_hz=fc,
            carrier_amplitude=ac,
        )

        self.assertEqual(mod.scheme, "AM-SC")
        self.assertEqual(mod.carrier_hz, fc)
        self.assertEqual(mod.estimated_bandwidth_hz, 2.0 * fm)

        # Análisis espectral: portadora en fc debe estar suprimida
        sig_mod = Signal1D(
            id="am-sc-fft",
            samples=mod.waveform,
            fs=fs,
            ts=1.0 / fs,
            duration=1.0,
            is_discrete=False,
            label="AM-SC Test",
        )
        spec = compute_spectrum(sig_mod, window="rectangular")

        idx_carrier = int(round(fc * spec.fft_size / fs))
        idx_usb = int(round((fc + fm) * spec.fft_size / fs))

        mag_carrier = spec.magnitudes[idx_carrier]
        mag_usb = spec.magnitudes[idx_usb]

        # Portadora suprimida frente a bandas laterales (atenuación > 40 dB o cociente muy pequeño)
        self.assertLess(mag_carrier / mag_usb, 0.01)

    def test_fm_modulation_constant_envelope_and_carson_bw(self):
        fs = 16000.0
        fm = 50.0
        fc = 2000.0
        delta_f = 200.0
        ac = 2.0

        msg = generate_synthetic_signal(
            waveform="cosine",
            amplitude=1.0,
            frequency_hz=fm,
            duration_sec=0.5,
            fs=fs,
        )

        mod = modulate_fm(
            signal=msg,
            carrier_hz=fc,
            frequency_deviation_hz=delta_f,
            carrier_amplitude=ac,
        )

        self.assertEqual(mod.scheme, "FM")
        self.assertEqual(mod.carrier_hz, fc)
        # Carson BW = 2 * (delta_f + fm) = 2 * (200 + 50) = 500 Hz
        self.assertEqual(mod.estimated_bandwidth_hz, 500.0)
        # Beta = delta_f / fm = 200 / 50 = 4.0
        self.assertEqual(mod.modulation_index, 4.0)

        # Envolvente constante en FM: |s(t)| <= Ac, y potencia media P ≈ Ac^2 / 2 = 2.0
        measured_power = float(np.mean(mod.waveform ** 2))
        expected_power = (ac ** 2) / 2.0
        self.assertAlmostEqual(measured_power, expected_power, delta=0.08)

    def test_analog_sampling_rate_and_validations(self):
        # Caso de insuficiencia de Fs frente a Carson / Nyquist
        with self.assertRaises(ValueError):
            # fc = 2000, fm = 100 -> BW = 200 -> max freq = 2100 -> requiere fs >= 4200
            modulate_am_lc(
                signal=[1.0, 0.5, -0.5, -1.0],
                carrier_hz=2000.0,
                fs=3000.0,  # Insuficiente!
                message_f_max_hz=100.0,
            )

        with self.assertRaises(ValueError):
            modulate_am_lc([1.0, 0.0], carrier_hz=-100.0, fs=1000.0)

        with self.assertRaises(ValueError):
            modulate_am_lc([1.0, 0.0], carrier_hz=100.0, modulation_index=-0.1, fs=1000.0)

        with self.assertRaises(ValueError):
            modulate_fm([1.0, 0.0], carrier_hz=500.0, frequency_deviation_hz=-10.0, fs=2000.0)



class TestDigitalModulation(unittest.TestCase):
    """Pruebas unitarias matemáticas para RF-09 (Modulación Digital Pasobanda: ASK, BFSK, BPSK, QPSK)."""

    def test_ask_ook_properties_and_constellation(self):
        bits = [1, 0, 1, 0]
        fc = 1000.0
        rb = 500.0
        sps = 16
        ac = 2.0

        mod = modulate_ask(
            bits=bits,
            carrier_hz=fc,
            symbol_rate_baud=rb,
            samples_per_symbol=sps,
            carrier_amplitude=ac,
        )

        self.assertEqual(mod.scheme, "ASK")
        self.assertEqual(mod.carrier_hz, fc)
        self.assertEqual(mod.symbol_rate_baud, rb)
        self.assertEqual(mod.samples_per_symbol, sps)
        self.assertEqual(mod.estimated_bandwidth_hz, 2.0 * rb)
        self.assertEqual(len(mod.waveform), len(bits) * sps)

        # Para bit 0 (muestras 16..31 y 48..63), la onda debe ser estrictamente 0.0
        bit0_samples = mod.waveform[16:32]
        np.testing.assert_array_almost_equal(bit0_samples, np.zeros(sps))

        # Para bit 1, el pico debe alcanzar ac
        bit1_samples = mod.waveform[0:16]
        self.assertAlmostEqual(float(np.max(bit1_samples)), ac, delta=0.05)

        # Constelación ideal OOK
        self.assertEqual(mod.ideal_constellation, [(0.0, 0.0), (ac, 0.0)])

    def test_bpsk_phase_inversion_and_constellation(self):
        bits = [1, 0]
        fc = 1000.0
        rb = 500.0
        sps = 16
        ac = 1.5

        mod = modulate_bpsk(
            bits=bits,
            carrier_hz=fc,
            symbol_rate_baud=rb,
            samples_per_symbol=sps,
            carrier_amplitude=ac,
        )

        self.assertEqual(mod.scheme, "BPSK")
        self.assertEqual(mod.carrier_hz, fc)
        self.assertEqual(mod.symbol_rate_baud, rb)
        self.assertEqual(mod.estimated_bandwidth_hz, 2.0 * rb)

        # Inversión de fase exacta: bit 0 es exactamente -bit 1
        sym1 = mod.waveform[0:sps]
        sym0 = mod.waveform[sps : 2 * sps]
        np.testing.assert_array_almost_equal(sym0, -sym1)

        # Potencia media de BPSK: P = Ac^2 / 2
        measured_power = float(np.mean(mod.waveform ** 2))
        expected_power = (ac ** 2) / 2.0
        self.assertAlmostEqual(measured_power, expected_power, delta=0.05)

        # Constelación simétrica antipodal
        self.assertEqual(mod.ideal_constellation, [(-ac, 0.0), (ac, 0.0)])

    def test_fsk_audible_and_custom_carrier(self):
        bits = [1, 0, 1, 1, 0]
        sps = 32
        rb = 300.0
        ac = 1.0

        # Modo Audible estándar Bell 202 (f0=1200 Hz, f1=2200 Hz)
        mod_audible = modulate_fsk(
            bits=bits,
            symbol_rate_baud=rb,
            samples_per_symbol=sps,
            carrier_amplitude=ac,
            continuous_phase=True,
        )

        self.assertEqual(mod_audible.scheme, "FSK")
        self.assertEqual(mod_audible.carrier_hz, 1700.0)
        self.assertEqual(mod_audible.estimated_bandwidth_hz, abs(2200.0 - 1200.0) + 2.0 * rb)
        self.assertEqual(len(mod_audible.waveform), len(bits) * sps)

        # Envolvente constante en CPFSK
        measured_power = float(np.mean(mod_audible.waveform ** 2))
        self.assertAlmostEqual(measured_power, 0.5 * (ac ** 2), delta=0.05)

        # Modo Pasabanda personalizado (fc = 4000 Hz, delta_f = 800 Hz)
        mod_custom = modulate_fsk(
            bits=bits,
            carrier_hz=4000.0,
            freq_shift_hz=800.0,
            symbol_rate_baud=400.0,
            samples_per_symbol=32,
        )
        self.assertEqual(mod_custom.carrier_hz, 4000.0)
        self.assertEqual(mod_custom.estimated_bandwidth_hz, 1600.0 + 2.0 * 400.0)

    def test_qpsk_constellation_and_bandwidth_efficiency(self):
        # 8 bits -> 4 dibits
        bits = [0, 0, 0, 1, 1, 0, 1, 1]
        fc = 1000.0
        rb = 500.0
        sps = 16
        ac = 2.0

        mod = modulate_qpsk(
            bits=bits,
            carrier_hz=fc,
            symbol_rate_baud=rb,
            samples_per_symbol=sps,
            carrier_amplitude=ac,
        )

        self.assertEqual(mod.scheme, "QPSK")
        self.assertEqual(mod.carrier_hz, fc)
        # Tasa de símbolos Rs = Rb / 2 = 250 Baud
        self.assertEqual(mod.symbol_rate_baud, 250.0)
        self.assertEqual(mod.samples_per_symbol, sps)
        # Ancho de banda QPSK = 2 * Rs = Rb = 500 Hz (la mitad de BPSK)
        self.assertEqual(mod.estimated_bandwidth_hz, rb)
        self.assertEqual(len(mod.waveform), 4 * sps)

        # Constelación 4-QAM / QPSK normalizada a vector de longitud Ac
        d = ac / np.sqrt(2.0)
        expected_points = [(-d, -d), (-d, d), (d, -d), (d, d)]
        for pt, exp in zip(mod.ideal_constellation, expected_points):
            self.assertAlmostEqual(pt[0], exp[0], places=6)
            self.assertAlmostEqual(pt[1], exp[1], places=6)
            # Norma de cada punto de constelación debe ser Ac
            self.assertAlmostEqual(np.sqrt(pt[0] ** 2 + pt[1] ** 2), ac, places=6)

        # Relleno de bit impar (ej. 5 bits -> pad a 6 bits = 3 símbolos)
        mod_odd = modulate_qpsk(
            bits=[1, 0, 1, 1, 0],
            carrier_hz=fc,
            symbol_rate_baud=rb,
            samples_per_symbol=sps,
        )
        self.assertEqual(len(mod_odd.waveform), 3 * sps)

    def test_digital_modulation_validations(self):
        # Bitstream vacío
        with self.assertRaises(ValueError):
            modulate_ask([], carrier_hz=1000.0)

        # Bits no binarios
        with self.assertRaises(ValueError):
            modulate_bpsk([0, 1, 3], carrier_hz=1000.0)

        # Portadora negativa o cero
        with self.assertRaises(ValueError):
            modulate_qpsk([1, 0], carrier_hz=0.0)

        # Frecuencia de muestreo insuficiente (Nyquist)
        with self.assertRaises(ValueError):
            modulate_ask([1, 0], carrier_hz=2000.0, symbol_rate_baud=100.0, samples_per_symbol=4)


if __name__ == "__main__":
    unittest.main()
