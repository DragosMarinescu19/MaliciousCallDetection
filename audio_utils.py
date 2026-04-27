import os
from typing import Type, TypedDict

import librosa
import numpy as np
import streamlit as st
import torch

from models import CNN, EnhancedCNN



class ModelEntry(TypedDict):
    file: str
    cls: Type[CNN]
    desc: str


class PredResult(TypedDict):
    malicious_prob: float
    is_malicious: bool



MODEL_INFO: dict[str, ModelEntry] = {
    "V17 — EnhancedCNN + Simulate Telephony": {
        "file": "malicious_call_detector_v17_best.pt",
        "cls": EnhancedCNN,
        "desc": "EnhancedCNN architecture with simulated telephony data.",
    },
    "V18 — Balanced EnhancedCNN + Simulate Telephony": {
        "file": "malicious_call_detector_v18_best.pt",
        "cls": EnhancedCNN,
        "desc": "EnhancedCNN architecture with simulated telephony data.",
    },
    "V19 — Optuna EnhancedCNN": {
        "file": "malicious_call_detector_v19_best.pt",
        "cls": EnhancedCNN,
        "desc": "Optuna-tuned EnhancedCNN architecture.",
    },
    "V20 — Best for malicious detection": {
        "file": "malicious_call_detector_v20_best.pt",
        "cls": EnhancedCNN,
        "desc": "Best for malicious detection.",
    },
    "V22 — V20 + Audio_With_Noise": {
        "file": "malicious_call_detector_v22_best.pt",
        "cls": EnhancedCNN,
        "desc": "V20 + Audio_With_Noise.",
    }
}



def is_silent(
    audio_segment: np.ndarray,
    sr: int,
    top_db: int = 30,
    min_active_time: float = 2.0,
) -> bool:
    """Return True if the segment is essentially silent (< min_active_time of speech)."""
    if np.max(np.abs(audio_segment)) < 1e-5:
        return True

    intervals = librosa.effects.split(audio_segment, top_db=top_db)
    if len(intervals) == 0:
        return True

    # Total duration of non-silent parts
    active_samples = sum(end - start for start, end in intervals)
    active_time = active_samples / sr
    return active_time < min_active_time


def create_spectrograms(
    audio: np.ndarray,
    sr: int = 16000,
    window_size: float = 5.0,
    min_overlap: float = 1.5,
) -> tuple[list[torch.Tensor], list[tuple[float, float]]]:
    """Split audio into overlapping windows and create normalised mel spectrograms.

    Returns:
        spectrograms: list of tensors shaped (1, 1, n_mels, time_frames).
        time_ranges:  list of (start_sec, end_sec) for each window.
    """
    window_samples = int(window_size * sr)
    min_overlap_samples = int(min_overlap * sr)
    total_samples = len(audio)

    if total_samples < int(2.5 * sr):
        return [], []

    # Pick step size based on audio length
    if total_samples < int(8.5 * sr) and total_samples >= window_samples:
        step = max(1, total_samples - window_samples)
    else:
        step = window_samples - min_overlap_samples

    spectrograms: list[torch.Tensor] = []
    time_ranges: list[tuple[float, float]] = []

    for start in range(0, total_samples, step):
        segment = audio[start : min(total_samples, start + window_samples)]

        # Pad short segments to full window size
        if len(segment) < window_samples:
            segment = np.pad(segment, (0, window_samples - len(segment)), mode="constant")

        if is_silent(segment, sr):
            continue

        # Create mel spectrogram and normalise
        mel_spec = librosa.feature.melspectrogram(
            y=segment, sr=sr, n_fft=2048, hop_length=512, n_mels=128,
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel = np.clip(log_mel, -80, 0)
        log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-9)

        tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        spectrograms.append(tensor)
        time_ranges.append((start / sr, min(start + window_samples, total_samples) / sr))

    return spectrograms, time_ranges



@st.cache_resource
def load_model(model_name: str) -> tuple:
    """Load a model from disk and cache it so it is only loaded once.

    Returns:
        (model, threshold, error_message)
        If error_message is not None, model and threshold will be None.
    """
    info = MODEL_INFO[model_name]
    model_path = os.path.join("models", info["file"])

    if not os.path.isdir("models"):
        return None, None, "The 'models/' directory was not found. Please place your .pt files there."

    if not os.path.exists(model_path):
        return None, None, f"Model file not found: {model_path}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = info["cls"]()

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        threshold: float = checkpoint.get("threshold", 0.5)
    else:
        model.load_state_dict(checkpoint)
        threshold = 0.5

    model.to(device).eval()
    return model, threshold, None


def predict(
    model: torch.nn.Module,
    spectrograms: list[torch.Tensor],
    threshold: float,
    device: torch.device,
) -> list[PredResult]:
    """Run inference on a list of spectrogram tensors and return per-window results."""
    results: list[PredResult] = []
    with torch.no_grad():
        for spec in spectrograms:
            spec = spec.to(device)
            pred = model(spec)
            prob = torch.softmax(pred, dim=1)
            mal_prob: float = prob[0][1].item()
            results.append({
                "malicious_prob": mal_prob,
                "is_malicious": mal_prob >= threshold,
            })
    return results
