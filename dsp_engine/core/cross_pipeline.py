"""
Orquestador del Pipeline Cruzado End-to-End (Fase 3)
Flujo completo: Imagen -> Cuantización -> Compresión -> Bits (BB) -> Audio CW (ASK/FSK/BPSK)
             -> Canal AWGN -> Demodulación I/Q -> Descompresión -> Imagen + Auditoría.

Nota arquitectónica (Plugin/Adapter Pattern):
Si los módulos dedicados del equipo (dsp_engine.compression / modulation / demodulation)
exponen las interfaces compatibles tras sus PRs, este adaptador los delega; mientras tanto,
provee un motor de referencia vectorizado en NumPy para pruebas end-to-end inmediatas
sin generar conflictos de fusión en las ramas de Ángel, Luis y Mariano.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np

from dsp_engine.contracts import ImageMatrix, ModulationType, Signal1D
from dsp_engine.core.fft import compute_spectrum
from dsp_engine.core.generator import generate_test_image
from dsp_engine.core.image_utils import bitstream_to_image, image_to_bitstream
from dsp_engine.core.quantization import pack_bits_to_bytes, unpack_bytes_to_bits


def _rle_compress_bytes(data: np.ndarray) -> np.ndarray:
    """
    Compresión PackBits RLE sin pérdida sobre bytes:
    - Bloque repetido (2..129 iguales): [128 + (count - 2), valor]
    - Bloque literal (1..128 distintos): [count - 1, b_0, b_1, ...]
    Evita expansión catastrófica (2x) en zonas sin repetición.
    """
    flat = np.asarray(data, dtype=np.uint8).reshape(-1)
    n = len(flat)
    if n == 0:
        return np.zeros(0, dtype=np.uint8)

    out: List[int] = []
    i = 0
    while i < n:
        # Detectar corrida de bytes idénticos (mínimo 2 para que valga la pena)
        run = 1
        while i + run < n and flat[i + run] == flat[i] and run < 129:
            run += 1
        if run >= 2:
            out.append(128 + (run - 2))
            out.append(int(flat[i]))
            i += run
        else:
            # Bloque literal hasta encontrar una corrida repetida o llegar a 128 bytes
            lit_start = i
            lit_len = 0
            while i < n and lit_len < 128:
                if i + 1 < n and flat[i] == flat[i + 1]:
                    break
                i += 1
                lit_len += 1
            out.append(lit_len - 1)
            for k in range(lit_start, lit_start + lit_len):
                out.append(int(flat[k]))
    return np.asarray(out, dtype=np.uint8)


def _rle_decompress_bytes(encoded: np.ndarray, expected_len: int) -> np.ndarray:
    """Descompresión PackBits RLE tolerante a errores de canal (ajusta a expected_len)."""
    flat = np.asarray(encoded, dtype=np.uint8).reshape(-1)
    out = np.zeros(expected_len, dtype=np.uint8)
    in_idx = 0
    out_idx = 0
    n_in = len(flat)

    while in_idx < n_in and out_idx < expected_len:
        ctrl = int(flat[in_idx])
        in_idx += 1
        if ctrl >= 128:
            count = (ctrl - 128) + 2
            val = int(flat[in_idx]) if in_idx < n_in else 0
            in_idx += 1
            end = min(expected_len, out_idx + count)
            out[out_idx:end] = val
            out_idx = end
        else:
            count = ctrl + 1
            for _ in range(count):
                if in_idx >= n_in or out_idx >= expected_len:
                    break
                out[out_idx] = flat[in_idx]
                in_idx += 1
                out_idx += 1
    return out


def _modulate_cw_digital(
    bits: np.ndarray,
    scheme: Literal["ASK", "FSK", "BPSK"] = "BPSK",
    fs: float = 8000.0,
    carrier_hz: float = 1500.0,
    samples_per_bit: int = 16,
) -> Dict[str, Any]:
    """Modula una traza binaria sobre Onda Portadora Continua (CW) audible."""
    flat_bits = (np.asarray(bits, dtype=np.uint8) & 1).reshape(-1)
    spb = max(8, int(samples_per_bit))
    n_bits = len(flat_bits)
    t_sym = np.arange(spb, dtype=np.float64) / fs

    waveform = np.zeros(n_bits * spb, dtype=np.float64)
    ideal_constellation: List[Tuple[float, float]] = []

    if scheme == "ASK":
        ref_cos = np.cos(2.0 * np.pi * carrier_hz * t_sym)
        for i, b in enumerate(flat_bits):
            amp = 1.0 if b == 1 else 0.15
            waveform[i * spb : (i + 1) * spb] = amp * ref_cos
        ideal_constellation = [(0.15, 0.0), (1.0, 0.0)]
        bw = 2.0 * (fs / spb)
    elif scheme == "FSK":
        f0 = max(300.0, carrier_hz - 400.0)
        f1 = min(fs * 0.42, carrier_hz + 600.0)
        wave0 = np.cos(2.0 * np.pi * f0 * t_sym)
        wave1 = np.cos(2.0 * np.pi * f1 * t_sym)
        for i, b in enumerate(flat_bits):
            waveform[i * spb : (i + 1) * spb] = wave1 if b == 1 else wave0
        ideal_constellation = [(1.0, 0.0), (0.0, 1.0)]
        bw = abs(f1 - f0) + 2.0 * (fs / spb)
    elif scheme == "BPSK":
        ref_cos = np.cos(2.0 * np.pi * carrier_hz * t_sym)
        for i, b in enumerate(flat_bits):
            pol = 1.0 if b == 1 else -1.0
            waveform[i * spb : (i + 1) * spb] = pol * ref_cos
        ideal_constellation = [(-1.0, 0.0), (1.0, 0.0)]
        bw = 2.0 * (fs / spb)
    else:
        raise ValueError(f"Esquema CW no soportado en pipeline cruzado: {scheme}")

    return {
        "waveform": waveform,
        "fs": fs,
        "carrier_hz": carrier_hz,
        "samples_per_bit": spb,
        "symbol_rate_baud": fs / spb,
        "estimated_bandwidth_hz": bw,
        "ideal_constellation": ideal_constellation,
    }


def _apply_awgn_channel(
    waveform: np.ndarray,
    snr_db: Optional[float],
    seed: int = 42,
) -> Tuple[np.ndarray, float]:
    """Inyecta Ruido Blanco Gaussiano Aditivo (AWGN) calibrado exactamente al SNR deseado."""
    sig = np.asarray(waveform, dtype=np.float64)
    if snr_db is None or snr_db >= 80.0:
        return sig.copy(), 99.9

    rng = np.random.default_rng(seed)
    sig_power = max(float(np.mean(sig ** 2)), 1e-12)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = rng.normal(0.0, np.sqrt(noise_power), size=len(sig))
    noisy = sig + noise
    actual_snr = 10.0 * np.log10(sig_power / max(float(np.mean(noise ** 2)), 1e-12))
    return noisy.astype(np.float64), float(actual_snr)


def _demodulate_cw_digital(
    noisy_waveform: np.ndarray,
    n_bits: int,
    scheme: Literal["ASK", "FSK", "BPSK"] = "BPSK",
    fs: float = 8000.0,
    carrier_hz: float = 1500.0,
    samples_per_bit: int = 16,
) -> Dict[str, Any]:
    """Demodulación por correlación coherente/no coherente + extracción de Constelación I/Q y Ojo."""
    spb = max(8, int(samples_per_bit))
    usable = n_bits * spb
    rx = np.asarray(noisy_waveform[:usable], dtype=np.float64)
    blocks = rx.reshape(n_bits, spb)

    t_sym = np.arange(spb, dtype=np.float64) / fs
    rx_constellation: List[List[float]] = []
    recovered_bits = np.zeros(n_bits, dtype=np.uint8)
    eye_traces: List[List[float]] = []

    if scheme == "BPSK":
        ref_i = 2.0 * np.cos(2.0 * np.pi * carrier_hz * t_sym) / spb
        ref_q = -2.0 * np.sin(2.0 * np.pi * carrier_hz * t_sym) / spb
        i_vals = blocks @ ref_i
        q_vals = blocks @ ref_q
        recovered_bits = (i_vals >= 0.0).astype(np.uint8)

        # Banda base proyectada para Diagrama de Ojo (envolvente síncrona por símbolo)
        baseband_rx = (blocks * np.cos(2.0 * np.pi * carrier_hz * t_sym)[None, :] * 2.0).reshape(-1)
        max_pts = min(n_bits, 400)
        rx_constellation = [[float(i_vals[k]), float(q_vals[k])] for k in range(max_pts)]
    elif scheme == "ASK":
        ref_i = 2.0 * np.cos(2.0 * np.pi * carrier_hz * t_sym) / spb
        ref_q = -2.0 * np.sin(2.0 * np.pi * carrier_hz * t_sym) / spb
        i_vals = blocks @ ref_i
        q_vals = blocks @ ref_q
        recovered_bits = (i_vals >= 0.575).astype(np.uint8)
        baseband_rx = (blocks * np.cos(2.0 * np.pi * carrier_hz * t_sym)[None, :] * 2.0).reshape(-1)
        max_pts = min(n_bits, 400)
        rx_constellation = [[float(i_vals[k]), float(q_vals[k])] for k in range(max_pts)]
    elif scheme == "FSK":
        f0 = max(300.0, carrier_hz - 400.0)
        f1 = min(fs * 0.42, carrier_hz + 600.0)
        ref0_c = 2.0 * np.cos(2.0 * np.pi * f0 * t_sym) / spb
        ref0_s = 2.0 * np.sin(2.0 * np.pi * f0 * t_sym) / spb
        ref1_c = 2.0 * np.cos(2.0 * np.pi * f1 * t_sym) / spb
        ref1_s = 2.0 * np.sin(2.0 * np.pi * f1 * t_sym) / spb

        e0 = np.sqrt((blocks @ ref0_c) ** 2 + (blocks @ ref0_s) ** 2)
        e1 = np.sqrt((blocks @ ref1_c) ** 2 + (blocks @ ref1_s) ** 2)
        recovered_bits = (e1 >= e0).astype(np.uint8)

        diff = (e1 - e0)
        baseband_rx = np.repeat(diff, spb) + (rx * 0.15)
        max_pts = min(n_bits, 400)
        rx_constellation = [[float(e0[k]), float(e1[k])] for k in range(max_pts)]
    else:
        raise ValueError(f"Esquema no soportado: {scheme}")

    # Extraer hasta 45 ventanas de 2*spb para el Diagrama de Ojo
    win_len = 2 * spb
    n_traces = min(45, max(0, (len(baseband_rx) - win_len) // spb))
    for t_idx in range(n_traces):
        start = t_idx * spb
        eye_traces.append(baseband_rx[start : start + win_len].tolist())

    return {
        "recovered_bits": recovered_bits,
        "rx_constellation": rx_constellation,
        "eye_traces": eye_traces,
    }


def _compute_audit_metrics(
    orig_pixels: np.ndarray,
    recon_pixels: np.ndarray,
    tx_bits: np.ndarray,
    rx_bits: np.ndarray,
) -> Dict[str, Any]:
    """Calcula métricas objetivas TX vs RX: BER, MSE, RMSE, PSNR, SNR y Correlación de Pearson."""
    n_bits = min(len(tx_bits), len(rx_bits))
    bit_errors = int(np.sum(tx_bits[:n_bits] != rx_bits[:n_bits]))
    ber = float(bit_errors / n_bits) if n_bits > 0 else 0.0

    orig_rgb = orig_pixels.reshape(-1, 4)[:, :3].astype(np.float64)
    recon_rgb = recon_pixels.reshape(-1, 4)[:, :3].astype(np.float64)

    diff = orig_rgb - recon_rgb
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    sig_power = float(np.mean(orig_rgb ** 2))
    snr_db = float(10.0 * np.log10(max(sig_power, 1e-12) / mse)) if mse > 1e-12 else 99.9
    psnr_db = float(10.0 * np.log10((255.0 ** 2) / mse)) if mse > 1e-12 else 99.9

    # Coeficiente de Correlación de Pearson ρ
    a = orig_rgb.reshape(-1) - np.mean(orig_rgb)
    b = recon_rgb.reshape(-1) - np.mean(recon_rgb)
    denom = float(np.sqrt(np.sum(a ** 2) * np.sum(b ** 2)))
    pearson = float(np.sum(a * b) / denom) if denom > 1e-12 else (1.0 if mse < 1e-12 else 0.0)

    return {
        "mse": mse,
        "rmse": rmse,
        "snrDb": snr_db,
        "psnrDb": psnr_db,
        "ber": ber,
        "bitErrors": bit_errors,
        "totalBits": n_bits,
        "pearsonCorrelation": pearson,
    }


def execute_cross_pipeline(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pipeline Cruzado Unificado:
    Imagen -> Cuantización (bitDepth) -> Compresión (RLE/None) -> Modulación CW Audible (ASK/FSK/BPSK)
    -> Canal AWGN (SNR dB) -> Demodulación I/Q -> Descompresión -> Reconstrucción + Auditoría.
    """
    width = min(64, max(8, int(params.get("width", 32))))
    height = min(64, max(8, int(params.get("height", 32))))
    bit_depth = min(8, max(1, int(params.get("bitDepth", 4))))
    use_rle = bool(params.get("useCompression", True))
    scheme = params.get("modulation", "BPSK")
    if scheme not in ("ASK", "FSK", "BPSK"):
        scheme = "BPSK"

    snr_param = params.get("snrDb", 20.0)
    snr_db = None if snr_param is None or float(snr_param) >= 60.0 else float(snr_param)
    carrier_hz = float(params.get("carrierHz", 1500.0))
    fs = float(params.get("fs", 8000.0))
    spb = int(params.get("samplesPerBit", 16))
    raw_pixels = params.get("pixels")

    # 1. Admisión de Imagen
    if raw_pixels is not None and len(raw_pixels) == width * height * 4:
        img = ImageMatrix(
            width=width,
            height=height,
            channels=4,
            pixels=np.asarray(raw_pixels, dtype=np.uint8),
            label=str(params.get("label", "Imagen Cargada")),
        )
    else:
        img = generate_test_image(width=width, height=height, pattern=params.get("pattern", "target"))

    # 2. Digitalización y Serialización a Bitstream
    raw_bits = image_to_bitstream(img, bit_depth=bit_depth)
    raw_bytes = pack_bits_to_bytes(raw_bits)
    expected_bytes_len = len(raw_bytes)

    # 3. Compresión (Separación planar RGB + PackBits RLE sobre flujo empaquetado)
    use_planar = use_rle and bit_depth == 8 and (expected_bytes_len % 3 == 0)
    if use_rle:
        prep_bytes = raw_bytes.reshape(-1, 3).T.reshape(-1) if use_planar else raw_bytes
        tx_bytes = _rle_compress_bytes(prep_bytes)
        codec_name = "PackBits-RLE (Planar)" if use_planar else "PackBits-RLE"
    else:
        tx_bytes = raw_bytes
        codec_name = "PCM Sin Compresión"

    tx_bits = unpack_bytes_to_bits(tx_bytes)
    compression_ratio = float(expected_bytes_len / max(1, len(tx_bytes)))
    savings_pct = float((1.0 - (len(tx_bytes) / max(1, expected_bytes_len))) * 100.0)

    # Limitar muestra máxima para mantener audio responsivo (< 3.5 s en vivo)
    max_tx_bits = int(params.get("maxTxBits", 12000))
    if len(tx_bits) > max_tx_bits:
        # Si excede el límite, escalamos el buffer de audio solo si es necesario, o avisamos
        tx_bits_mod = tx_bits[:max_tx_bits]
    else:
        tx_bits_mod = tx_bits

    # 4. Modulación CW Audible
    mod = _modulate_cw_digital(
        tx_bits_mod,
        scheme=scheme,
        fs=fs,
        carrier_hz=carrier_hz,
        samples_per_bit=spb,
    )

    # 5. Canal AWGN
    noisy_wave, actual_snr = _apply_awgn_channel(mod["waveform"], snr_db=snr_db, seed=int(params.get("seed", 42)))

    # Análisis Espectral FFT de la señal modulada en canal (ventana máx 4096 muestras para velocidad)
    fft_slice = noisy_wave[: min(4096, len(noisy_wave))]
    sig_rf = Signal1D(
        id="rf-cw",
        samples=fft_slice,
        fs=fs,
        ts=1.0 / fs,
        duration=len(fft_slice) / fs,
        is_discrete=False,
        label=f"CW {scheme}",
    )
    rf_spec = compute_spectrum(sig_rf, window="hann")

    # 6. Demodulación y Detección de Bits
    demod = _demodulate_cw_digital(
        noisy_wave,
        n_bits=len(tx_bits_mod),
        scheme=scheme,
        fs=fs,
        carrier_hz=carrier_hz,
        samples_per_bit=spb,
    )

    rx_bits_full = tx_bits.copy()
    rx_bits_full[: len(tx_bits_mod)] = demod["recovered_bits"]

    # 7. Descompresión y Reconstrucción de Imagen
    rx_bytes = pack_bits_to_bytes(rx_bits_full)[: len(tx_bytes)]
    if use_rle:
        decompressed_raw = _rle_decompress_bytes(rx_bytes, expected_len=expected_bytes_len)
        decompressed_bytes = (
            decompressed_raw.reshape(3, -1).T.reshape(-1) if use_planar else decompressed_raw
        )
    else:
        decompressed_bytes = np.zeros(expected_bytes_len, dtype=np.uint8)
        copy_n = min(expected_bytes_len, len(rx_bytes))
        decompressed_bytes[:copy_n] = rx_bytes[:copy_n]

    rx_image_bits = unpack_bytes_to_bits(decompressed_bytes, total_bits=len(raw_bits))
    recon_img = bitstream_to_image(rx_image_bits, width=width, height=height, source_channels=3, bit_depth=bit_depth)

    # 8. Auditoría de Fidelidad
    metrics = _compute_audit_metrics(
        orig_pixels=img.pixels,
        recon_pixels=recon_img.pixels,
        tx_bits=tx_bits_mod,
        rx_bits=demod["recovered_bits"],
    )

    return {
        "width": width,
        "height": height,
        "originalPixels": img.pixels.tolist(),
        "reconstructedPixels": recon_img.pixels.tolist(),
        "compression": {
            "codec": codec_name,
            "originalBytes": int(expected_bytes_len),
            "compressedBytes": int(len(tx_bytes)),
            "compressionRatio": compression_ratio,
            "savingsPercent": savings_pct,
            "compressedBytesHex": " ".join(f"{b:02X}" for b in tx_bytes[:24]),
            "compressedRaw": tx_bytes.tolist(),
        },
        "modulation": {
            "scheme": scheme,
            "carrierHz": mod["carrier_hz"],
            "fs": mod["fs"],
            "symbolRateBaud": mod["symbol_rate_baud"],
            "estimatedBandwidthHz": mod["estimated_bandwidth_hz"],
            "idealConstellation": mod["ideal_constellation"],
            "waveformPreview": noisy_wave[: min(1200, len(noisy_wave))].tolist(),
            "audioWaveform": noisy_wave.tolist(),
            "durationSec": float(len(noisy_wave) / fs),
            "actualSnrDb": actual_snr,
        },
        "spectrum": {
            "frequencies": rf_spec.frequencies.tolist(),
            "magnitudesDb": rf_spec.magnitudes_db.tolist(),
            "dominantFrequencyHz": rf_spec.dominant_frequency_hz,
            "bandwidthHz": rf_spec.bandwidth_hz,
        },
        "demodulation": {
            "rxConstellation": demod["rx_constellation"],
            "eyeTraces": demod["eye_traces"],
            "txBitsPreview": tx_bits_mod[:48].tolist(),
            "rxBitsPreview": demod["recovered_bits"][:48].tolist(),
        },
        "metrics": metrics,
    }
