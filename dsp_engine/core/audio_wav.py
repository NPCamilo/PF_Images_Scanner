"""Codificador y decodificador RIFF WAV estándar (100% compatible con Pyodide sin librerías C)."""

import io
import wave
from typing import Tuple
import numpy as np


def encode_wav_bytes(
    samples: np.ndarray,
    fs: int,
    bit_depth: int = 16,
) -> bytes:
    """Codifica un arreglo normalizado [-1, 1] a bytes RIFF WAV PCM (8 o 16 bits)."""
    if fs <= 0:
        raise ValueError("La frecuencia de muestreo WAV debe ser positiva.")
    clipped = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setframerate(int(fs))
        if bit_depth == 8:
            wf.setsampwidth(1)
            pcm = np.round((clipped + 1.0) * 127.5).astype(np.uint8)
            wf.writeframes(pcm.tobytes())
        else:
            wf.setsampwidth(2)
            pcm = np.round(clipped * 32767.0).astype("<i2")
            wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def decode_wav_bytes(wav_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Decodifica un buffer RIFF WAV PCM (mono o estéreo -> mono normalizado [-1, 1]) y su Fs."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        fs = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 1:
        arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif sampwidth == 2:
        arr = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sampwidth == 4:
        arr = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"Ancho de muestra WAV no soportado: {sampwidth} bytes.")

    if n_channels > 1:
        arr = arr.reshape(-1, n_channels).mean(axis=1)

    return arr.astype(np.float64), int(fs)
