"""
Puente de invocación JS <-> Pyodide para ejecutar el motor Python 100% en navegador.
Provee los endpoints de escaneo Tri-Modal (Audio, Imagen, Bitstream) y exportación WAV.
"""

import base64
from typing import Any, Dict
import numpy as np

from dsp_engine.contracts import ImageMatrix, Signal1D
from dsp_engine.core import (
    bits_to_hex,
    bitstream_to_image,
    compute_spectrogram,
    compute_spectrum,
    encode_wav_bytes,
    extract_red_dominant_mask,
    generate_bit_trace_waveform,
    generate_prbs,
    generate_synthetic_signal,
    generate_test_image,
    hex_to_bits,
    image_to_bitstream,
    image_to_signal_vector,
    quantize_pcm,
    simulate_sampling_nqs,
    text_to_bits,
)


def run_audio_signal_scan(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Escaneo completo para señales de audio/sintéticas x(t) y x[n]:
    Admisión/Síntesis -> NQS -> Cuantización PCM -> FFT -> STFT Espectrograma -> Traza de Bits.
    """
    raw_samples = params.get("rawSamples")
    fs = float(params.get("fs", 8000.0))

    if raw_samples is not None and len(raw_samples) > 0:
        samples = np.asarray(raw_samples, dtype=np.float64)
        # Limitar longitud máxima de visualización en vivo si es muy larga
        max_len = int(params.get("maxSamples", 32768))
        if len(samples) > max_len:
            samples = samples[:max_len]
        sig = Signal1D(
            id="sig-imported",
            samples=samples,
            fs=fs,
            ts=1.0 / fs,
            duration=len(samples) / fs,
            is_discrete=False,
            label=str(params.get("label", "Señal Externa")),
        )
        known_fmax = None
    else:
        freq_hz = float(params.get("frequencyHz", 440.0))
        sig = generate_synthetic_signal(
            waveform=params.get("waveform", "sine"),
            amplitude=float(params.get("amplitude", 1.0)),
            frequency_hz=freq_hz,
            secondary_frequency_hz=params.get("secondaryFrequencyHz"),
            phase_rad=float(params.get("phaseRad", 0.0)),
            dc_offset=float(params.get("dcOffset", 0.0)),
            duration_sec=float(params.get("durationSec", 0.05)),
            fs=fs,
        )
        known_fmax = freq_hz if params.get("waveform", "sine") in ("sine", "cosine") else None

    downsample = int(params.get("downsampleFactor", 1))
    bit_depth = int(params.get("bitDepth", 8))
    window = params.get("window", "hann")
    line_coding = params.get("lineCoding", "polar-nrz")

    nqs = simulate_sampling_nqs(sig, downsample_factor=downsample, known_max_freq_hz=known_fmax)
    spec = compute_spectrum(sig, window=window, reference_max_freq_hz=known_fmax)
    stft = compute_spectrogram(sig, window_size=min(256, max(32, len(sig.samples) // 8)), hop_size=64, window=window)

    # Cuantización sobre la señal muestreada x[n]
    sampled_sig = Signal1D(
        id="sig-sampled",
        samples=nqs.sampled_values,
        fs=nqs.effective_fs,
        ts=nqs.effective_ts,
        duration=len(nqs.sampled_values) * nqs.effective_ts,
        is_discrete=True,
        label="Muestreada NQS",
    )
    pcm = quantize_pcm(sampled_sig, bit_depth=bit_depth)

    preview_bits = pcm.bitstream[:96]
    bit_trace_wave = generate_bit_trace_waveform(preview_bits, samples_per_bit=12, coding=line_coding)

    return {
        "samples": sig.samples.tolist(),
        "quantized": pcm.quantized.tolist(),
        "errorSignal": pcm.error_signal.tolist(),
        "sampledTimes": nqs.sampled_times.tolist(),
        "sampledValues": nqs.sampled_values.tolist(),
        "fs": sig.fs,
        "ts": sig.ts,
        "duration": sig.duration,
        "nqs": {
            "effectiveFs": nqs.effective_fs,
            "effectiveTs": nqs.effective_ts,
            "nyquistMinimumHz": nqs.nyquist_minimum_hz,
            "nyquistSatisfied": nqs.nyquist_satisfied,
            "nyquistMarginHz": nqs.nyquist_margin_hz,
            "aliasFrequencyHz": nqs.alias_frequency_hz,
        },
        "spectrum": {
            "frequencies": spec.frequencies.tolist(),
            "magnitudes": spec.magnitudes.tolist(),
            "magnitudesDb": spec.magnitudes_db.tolist(),
            "dominantFrequencyHz": spec.dominant_frequency_hz,
            "fMaxHz": spec.f_max_hz,
            "bandwidthHz": spec.bandwidth_hz,
        },
        "spectrogram": {
            "times": stft["times"].tolist(),
            "frequencies": stft["frequencies"].tolist(),
            "powerDb": stft["power_db"].tolist(),
        },
        "quantization": {
            "bitDepth": pcm.bit_depth,
            "numLevels": pcm.num_levels,
            "stepSizeDelta": pcm.step_size_delta,
            "sqnrTheoreticalDb": pcm.sqnr_theoretical_db,
            "sqnrMeasuredDb": pcm.sqnr_measured_db,
            "totalBits": int(pcm.bitstream.size),
            "packedBytesLength": int(pcm.packed_bytes.size),
            "previewBits": preview_bits.tolist(),
            "previewHex": bits_to_hex(pcm.bitstream, max_bytes=24),
            "bitTraceWaveform": bit_trace_wave.tolist(),
        },
    }


def run_image_scan(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Escaneo espacial y espectral de matriz de imagen 2D:
    Imagen -> Vector x[n] -> FFT espacial -> Máscara Roja -> Bitstream N bits -> Reconstrucción.
    """
    width = int(params.get("width", 64))
    height = int(params.get("height", 64))
    bit_depth = int(params.get("bitDepth", 8))
    raw_pixels = params.get("pixels")

    if raw_pixels is not None and len(raw_pixels) > 0:
        img = ImageMatrix(
            width=width,
            height=height,
            channels=4,
            pixels=np.asarray(raw_pixels, dtype=np.uint8),
            label=str(params.get("label", "Imagen Cargada")),
        )
    else:
        img = generate_test_image(width=width, height=height, pattern=params.get("pattern", "target"))

    vec_1d = image_to_signal_vector(img)
    pseudo_fs = float(width)  # Frecuencia espacial en ciclos/ancho de línea
    sig_spatial = Signal1D(
        id="img-raster",
        samples=vec_1d,
        fs=pseudo_fs,
        ts=1.0 / pseudo_fs,
        duration=len(vec_1d) / pseudo_fs,
        is_discrete=True,
        label=img.label,
    )
    spec = compute_spectrum(sig_spatial, window="hann")
    bits = image_to_bitstream(img, bit_depth=bit_depth)
    recon = bitstream_to_image(bits, width=width, height=height, source_channels=3, bit_depth=bit_depth)
    red_masked, red_ratio = extract_red_dominant_mask(img)

    # Error cuadrático medio espacial entre imagen original (RGB) y cuantizada reconstruida
    orig_rgb = img.pixels.reshape(-1, 4)[:, :3].astype(np.float64)
    recon_rgb = recon.pixels.reshape(-1, 4)[:, :3].astype(np.float64)
    mse = float(np.mean((orig_rgb - recon_rgb) ** 2))
    psnr = float(10.0 * np.log10((255.0 ** 2) / mse)) if mse > 1e-12 else 99.9

    return {
        "width": width,
        "height": height,
        "originalPixels": img.pixels.tolist(),
        "reconstructedPixels": recon.pixels.tolist(),
        "redMaskPixels": red_masked.pixels.tolist(),
        "redPixelRatio": red_ratio,
        "rasterSamples": vec_1d[: min(2048, len(vec_1d))].tolist(),
        "spectrum": {
            "frequencies": spec.frequencies[: min(512, len(spec.frequencies))].tolist(),
            "magnitudesDb": spec.magnitudes_db[: min(512, len(spec.magnitudes_db))].tolist(),
            "bandwidthHz": spec.bandwidth_hz,
        },
        "metrics": {
            "bitDepthPerChannel": bit_depth,
            "totalBits": int(bits.size),
            "rawBytes": int(np.ceil(bits.size / 8)),
            "mse": mse,
            "psnrDb": psnr,
            "previewBits": bits[:96].tolist(),
            "previewHex": bits_to_hex(bits, max_bytes=24),
        },
    }


def run_bitstream_scan(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Escaneo y generación de traza de bits en Banda Base (BB):
    PRBS / Texto UTF-8 / Hexadecimal -> Forma de onda NRZ/Manchester -> FFT y Ancho de Banda.
    """
    source = params.get("source", "prbs")
    coding = params.get("lineCoding", "polar-nrz")
    bit_rate_bps = float(params.get("bitRateBps", 1000.0))
    spb = int(params.get("samplesPerBit", 16))

    if source == "text":
        bits = text_to_bits(str(params.get("text", "UTP-DSP")))
    elif source == "hex":
        bits = hex_to_bits(str(params.get("hex", "A5 5A F0 0F")))
    else:
        bits = generate_prbs(int(params.get("length", 64)), seed=int(params.get("seed", 0xACE1)))

    if len(bits) == 0:
        bits = np.array([1, 0, 1, 0], dtype=np.uint8)

    waveform = generate_bit_trace_waveform(bits, samples_per_bit=spb, coding=coding)
    fs = bit_rate_bps * spb
    sig = Signal1D(
        id="bitstream-bb",
        samples=waveform,
        fs=fs,
        ts=1.0 / fs,
        duration=len(waveform) / fs,
        is_discrete=False,
        label=f"Baseband {coding}",
    )
    spec = compute_spectrum(sig, window="hann")

    return {
        "bits": bits.tolist(),
        "totalBits": int(len(bits)),
        "hexPreview": bits_to_hex(bits, max_bytes=48),
        "waveform": waveform.tolist(),
        "fs": fs,
        "bitRateBps": bit_rate_bps,
        "coding": coding,
        "dcMean": float(np.mean(waveform)),
        "spectrum": {
            "frequencies": spec.frequencies.tolist(),
            "magnitudesDb": spec.magnitudes_db.tolist(),
            "bandwidthHz": spec.bandwidth_hz,
            "dominantFrequencyHz": spec.dominant_frequency_hz,
        },
    }


def export_wav_base64(params: Dict[str, Any]) -> str:
    """Genera un archivo RIFF WAV PCM estándar en Base64 listo para descarga en navegador."""
    samples = np.asarray(params.get("samples", []), dtype=np.float64)
    fs = int(params.get("fs", 8000))
    bit_depth = int(params.get("bitDepth", 16))
    wav_data = encode_wav_bytes(samples, fs=fs, bit_depth=bit_depth)
    return base64.b64encode(wav_data).decode("ascii")


def run_core_pipeline_demo(params: Dict[str, Any]) -> Dict[str, Any]:
    """Alias retrocompatible con pruebas de Fase 1."""
    return run_audio_signal_scan(params)
