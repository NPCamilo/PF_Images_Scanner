"""
Puente de invocación JS <-> Pyodide para ejecutar el motor Python 100% en navegador.
Convierte arreglos NumPy a listas/buffers serializables de forma determinística.
"""

from typing import Any, Dict
import numpy as np

from dsp_engine.core import (
    compute_spectrum,
    generate_synthetic_signal,
    quantize_pcm,
    simulate_sampling_nqs,
)


def run_core_pipeline_demo(params: Dict[str, Any]) -> Dict[str, Any]:
    """Ejecuta Adquisición/Síntesis -> Análisis FFT -> Muestreo NQS -> Cuantización PCM."""
    sig = generate_synthetic_signal(
        waveform=params.get("waveform", "sine"),
        amplitude=float(params.get("amplitude", 1.0)),
        frequency_hz=float(params.get("frequencyHz", 440.0)),
        phase_rad=float(params.get("phaseRad", 0.0)),
        dc_offset=float(params.get("dcOffset", 0.0)),
        duration_sec=float(params.get("durationSec", 0.05)),
        fs=float(params.get("fs", 8000.0)),
    )
    downsample = int(params.get("downsampleFactor", 1))
    bit_depth = int(params.get("bitDepth", 8))
    window = params.get("window", "hann")

    nqs = simulate_sampling_nqs(
        sig,
        downsample_factor=downsample,
        known_max_freq_hz=float(params.get("frequencyHz", 440.0)),
    )
    spec = compute_spectrum(sig, window=window)
    pcm = quantize_pcm(sig, bit_depth=bit_depth)

    return {
        "samples": sig.samples.tolist(),
        "fs": sig.fs,
        "ts": sig.ts,
        "duration": sig.duration,
        "nqs": {
            "effectiveFs": nqs.effective_fs,
            "effectiveTs": nqs.effective_ts,
            "nyquistMinimumHz": nqs.nyquist_minimum_hz,
            "nyquistSatisfied": nqs.nyquist_satisfied,
            "aliasFrequencyHz": nqs.alias_frequency_hz,
        },
        "spectrum": {
            "frequencies": spec.frequencies.tolist(),
            "magnitudesDb": spec.magnitudes_db.tolist(),
            "dominantFrequencyHz": spec.dominant_frequency_hz,
            "bandwidthHz": spec.bandwidth_hz,
        },
        "quantization": {
            "bitDepth": pcm.bit_depth,
            "stepSizeDelta": pcm.step_size_delta,
            "sqnrTheoreticalDb": pcm.sqnr_theoretical_db,
            "sqnrMeasuredDb": pcm.sqnr_measured_db,
            "totalBits": int(pcm.bitstream.size),
            "packedBytesLength": int(pcm.packed_bytes.size),
        },
    }
