"""
Suite de Pruebas Matemáticas Unitarias — Modulación y Codificación de Línea
Responsable: Ángel Gabriel Burbano (tests_py/test_modulation.py)
"""

import unittest
import numpy as np

from dsp_engine.core import compute_spectrum, generate_prbs
from dsp_engine.modulation import encode_baseband, generate_symbol_clock


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


if __name__ == "__main__":
    unittest.main()
