import os
import tempfile
from typing import cast

import librosa
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from audio_utils import (
    MODEL_INFO,
    PredResult,
    create_spectrograms,
    load_model,
    predict,
)


def display_results(
    audio: np.ndarray,
    sr: int,
    model: torch.nn.Module,
    threshold: float,
    device: torch.device,
) -> None:
    duration = len(audio) / sr

    # Create spectrograms
    with st.spinner("Creating spectrograms..."):
        spectrograms, time_ranges = create_spectrograms(audio, sr)

    if len(spectrograms) == 0:
        st.warning("Audio is too short or too silent for analysis. Please provide at least 2.5 seconds of speech.")
        return

    # Run prediction
    with st.spinner("Analyzing audio..."):
        results = predict(model, spectrograms, threshold, device)

    flagged = sum(1 for r in results if r["is_malicious"])
    total = len(results)
    mal_ratio = flagged / total
    avg_confidence = float(np.mean([r["malicious_prob"] if r["is_malicious"] else 1-r["malicious_prob"] for r in results])) * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if mal_ratio >= 0.5:
            st.error("**Verdict: MALICIOUS**")
        else:
            st.success("**Verdict: NORMAL**")
    with col2:
        if mal_ratio >= 0.5:
            st.metric("Flagged Windows", f"{flagged}/{total}", delta=f"{mal_ratio * 100:.1f}%", delta_color="red")
        else:
            st.metric("Flagged Windows", f"{flagged}/{total}", delta=f"{mal_ratio * 100:.1f}%")
        
    with col3:
        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
    with col4:
        st.metric("Duration", f"{duration:.1f}s")

    # Risk score bar
    risk = avg_confidence if mal_ratio >= 0.5 else 100 - avg_confidence
    st.markdown(f"**Fraud Risk Score:** {risk:.0f}/100")
    st.progress(min(risk / 100, 1.0))

    st.subheader("Per-Window Timeline")
    cols_per_row = 10
    for row_start in range(0, total, cols_per_row):
        cols = st.columns(min(cols_per_row, total - row_start))
        for i, col in enumerate(cols):
            idx = row_start + i
            r = results[idx]
            t = time_ranges[idx]
            if r["is_malicious"]:
                col.markdown(
                    f"<div style='background:#ff1744;color:white;padding:4px;"
                    f"border-radius:4px;text-align:center;font-size:11px;'>"
                    f"{t[0]:.1f}s<br>{r['malicious_prob']:.0%}</div>",
                    unsafe_allow_html=True,
                )
            else:
                col.markdown(
                    f"<div style='background:#00c853;color:white;padding:4px;"
                    f"border-radius:4px;text-align:center;font-size:11px;'>"
                    f"{t[0]:.1f}s<br>{1 - r['malicious_prob']:.0%}</div>",
                    unsafe_allow_html=True,
                )

    st.subheader("Spectrogram Analysis")
    n_specs: int = min(len(spectrograms), 20)
    n_cols: int = 5
    n_rows: int = (n_specs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    fig.patch.set_facecolor("#0e1117")
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    grid_size: int = n_rows * n_cols
    for i in range(grid_size):
        ax = axes[i // n_cols][i % n_cols]
        if i < n_specs:
            mel = spectrograms[i].squeeze().numpy()
            ax.imshow(mel, origin="lower", aspect="auto", cmap="magma")
            r = results[i]
            t = time_ranges[i]
            if r["is_malicious"]:
                ax.set_title(
                    f"⚠ MAL ({r['malicious_prob']:.2f})\n{t[0]:.1f}s–{t[1]:.1f}s",
                    color="red", fontsize=9,
                )
            else:
                ax.set_title(
                    f"✓ NOR ({1 - r['malicious_prob']:.2f})\n{t[0]:.1f}s–{t[1]:.1f}s",
                    color="lime", fontsize=9,
                )
            ax.set_facecolor("#0e1117")
        else:
            ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if flagged > 0:
        st.subheader("⚠️ Flagged Segments")
        for i, r in enumerate(results):
            if r["is_malicious"]:
                t = time_ranges[i]
                st.markdown(
                    f"- Window {i + 1}: **{t[0]:.1f}s → {t[1]:.1f}s** "
                    f"— Malicious confidence: `{r['malicious_prob']:.1%}`"
                )



def _load_audio_from_file(
    file_data,
    suffix: str,
    max_duration: int,
) -> tuple[np.ndarray, int] | None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_data.read())
        tmp_path = tmp.name

    audio, sr = librosa.load(tmp_path, sr=16000)
    os.unlink(tmp_path) #delete the temporary file after reading it
    duration = len(audio) / sr

    if duration > max_duration:
        st.info(
            f"Audio is {duration:.0f}s long. Maximum allowed is {max_duration}s. "
            "The audio was truncated. You can also adjust the max duration in the sidebar."
        )
        audio = audio[: int(max_duration * sr)]

    return audio, sr



def render_upload_tab(selected_model: str, max_duration: int) -> None:
    """Content for the '📁 Upload Audio' tab."""
    uploaded = st.file_uploader(
        "Choose an audio file (WAV or MP3)",
        type=["wav", "mp3"],
        help=f"Maximum duration: {max_duration} seconds",
    )

    if uploaded is not None:
        suffix = ".wav" if uploaded.name.endswith(".wav") else ".mp3"
        result = _load_audio_from_file(uploaded, suffix, max_duration)
        if result is None:
            return
        audio, sr = result

        st.audio(uploaded, format=f"audio/{suffix[1:]}")

        model, threshold, error = load_model(selected_model)
        if error:
            st.error(error)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            display_results(audio, sr, model, threshold, device)


def render_record_tab(selected_model: str, max_duration: int) -> None:
    """Content for the '🎙️ Record Audio' tab."""
    st.markdown("Record audio directly from your microphone.")
    recorded = st.audio_input(
        "Click to start recording",
        help="Record a short clip and analyze it.",
    )

    if recorded is not None:
        result = _load_audio_from_file(recorded, ".wav", max_duration)
        if result is None:
            return
        audio, sr = result
        duration = len(audio) / sr

        if duration < 2.5:
            st.warning("Recording is too short. Please record at least 3 seconds.")
            return

        model, threshold, error = load_model(selected_model)
        if error:
            st.error(error)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            display_results(audio, sr, model, threshold, device)


def render_compare_tab(max_duration: int) -> None:
    """Content for the '⚖️ Compare Models' tab."""
    st.markdown("Upload an audio file and compare predictions across two models side by side.")

    compare_file = st.file_uploader(
        "Choose audio for comparison", type=["wav", "mp3"], key="compare",
    )

    col_left, col_right = st.columns(2)
    with col_left:
        model_a = st.selectbox("Model A", list(MODEL_INFO.keys()), index=0, key="model_a")
    with col_right:
        model_b = st.selectbox("Model B", list(MODEL_INFO.keys()), index=2, key="model_b")

    if compare_file is not None:
        suffix = ".wav" if compare_file.name.endswith(".wav") else ".mp3"
        result = _load_audio_from_file(compare_file, suffix, max_duration)
        if result is None:
            return
        audio, sr = result

        st.audio(compare_file, format=f"audio/{suffix[1:]}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"### {model_a}")
            model_a, threshold_a, err = load_model(model_a)
            if err:
                st.error(err)
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                display_results(audio, sr, model_a, threshold_a, device)

        with col_b:
            st.markdown(f"### {model_b}")
            model_b, threshold_b, err = load_model(model_b)
            if err:
                st.error(err)
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                display_results(audio, sr, model_b, threshold_b, device)
