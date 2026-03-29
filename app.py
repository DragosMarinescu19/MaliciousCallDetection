import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import librosa
import matplotlib.pyplot as plt
import io
import os
import tempfile
from typing import Type, TypedDict, cast



class OldCNN(nn.Module):
    def __init__(self):
        super(OldCNN, self).__init__()
        self.features = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),#in_channels=1 because spectogram is grayscale not RGB
            nn.BatchNorm2d(16), #Normalization(mean=0,std=1)
            nn.ReLU(),
            nn.MaxPool2d(2),#reduce image in half, concentrating on the most imp pixel

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1)) #256 channels, 1 height, 1 width

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x



class CNN(nn.Module):
    def __init__(self,dropout=0.41):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            #we use padding because we don't want to loose pixels from the images margins, now the output matrix is exactly as the input
            #when working with spectrograms this is important
            #formula : out=(W−K+2P)/S ​+ 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),#in_channels=1 because spectogram is grayscale not RGB
            nn.BatchNorm2d(32), #Normalization(mean=0,std=1)
            nn.ReLU(),
            nn.MaxPool2d(2),#reduce image in half, concentrating on the most imp pixel

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # This forces the spatial dimensions down to 1x1, regardless of input length.
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) #256 channels, 1 height, 1 width

        # Classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256), # It's 256 because the last Conv2d output 256 channels
            nn.ReLU(),
            nn.Dropout(dropout), # Added dropout to prevent overfitting
            nn.Linear(256, 2) # 2=num_classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def is_silent(audio_segment, sr, top_db=30, min_active_time=2.0):
    if np.max(np.abs(audio_segment)) < 1e-5: #if whole sound is blank
        return True
    intervals = librosa.effects.split(audio_segment, top_db=top_db)

    if len(intervals) == 0:
        return True
    # Calculate is total duration of non silent aprts is less than 2 sec
    active_samples = sum([end - start for start, end in intervals])
    active_time = active_samples / sr

    return active_time < min_active_time


def create_spectrograms(audio, sr=16000, window_size=5.0, min_overlap=1.5):
    """Split audio into windows and create mel spectrograms."""
    window_size_samples = int(window_size * sr)
    min_overlap_samples = int(min_overlap * sr)
    total_samples = len(audio)

    if total_samples < int(2.5 * sr):
        return [], []

    # pick step size based on audio length
    if total_samples < int(8.5 * sr) and total_samples >= window_size_samples:
        step = max(1, total_samples - window_size_samples)
    else:
        step = window_size_samples - min_overlap_samples

    spectrograms = []
    time_ranges = []

    for start in range(0, total_samples, step):
        segment = audio[start: min(total_samples, start + window_size_samples)]

        # pad short segments to full window size
        if len(segment) < window_size_samples:
            segment = np.pad(segment, (0, window_size_samples - len(segment)), mode="constant")

        if is_silent(segment, sr):
            continue

        # create mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=segment, sr=sr, n_fft=2048, hop_length=512, n_mels=128
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel = np.clip(log_mel, -80, 0)
        log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-9)

        tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        spectrograms.append(tensor)
        time_ranges.append((start/sr, min(start + window_size_samples, total_samples) / sr))

    return spectrograms, time_ranges



class ModelEntry(TypedDict):
    file: str
    cls: Type[CNN]
    desc: str


class PredResult(TypedDict):
    malicious_prob: float
    is_malicious: bool


# available models and their descriptions
MODEL_INFO: dict[str, ModelEntry] = {
    "V8 — Balanced (Best for general use)": {
        "file": "malicious_call_detector_v8_best.pt",
        "cls": CNN,
        "desc": "Optuna Original model trained on YouTube + RFP data. Best balance between Normal and Malicious detection."
    },
    "V14 — PF Detector (Aggressive)": {
        "file": "malicious_call_detector_v14_best.pt",
        "cls": CNN,
        "desc": "Fine-tuned on Partially Fake audio. Catches 129/131 PF files but may flag some normal calls."
    },
    "V15 — Rebalanced (Latest)": {
        "file": "malicious_call_detector_v15_best.pt",
        "cls": CNN,
        "desc": "Trained with extra Normal + Malicious data. Good balance, improved Normal recall."
    },
    "V4 — Without Real Data": {
        "file": "malicious_call_detector_v4_best.pt",
        "cls": OldCNN,
        "desc": "Trained without real data. Worst performance."
    }
}


@st.cache_resource
def load_model(model_name):
    """Load model from disk and cache it so it only loads once."""
    info = MODEL_INFO[model_name]
    model_path = os.path.join("models", info["file"])

    if not os.path.exists(model_path):
        return None, None, f"Model file not found: {model_path}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = info["cls"]()

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        threshold = checkpoint.get("threshold", 0.5)
    else:
        model.load_state_dict(checkpoint)
        threshold = 0.5

    model.to(device).eval()
    return model, threshold, None


def predict(model, spectrograms, threshold, device) -> list[PredResult]:
    """Run prediction on a list of spectrogram tensors."""
    results: list[PredResult] = []
    with torch.no_grad():
        for spec in spectrograms:
            spec = spec.to(device)
            pred = model(spec)
            prob = torch.softmax(pred, dim=1)
            mal_prob = prob[0][1].item()
            results.append({
                "malicious_prob": mal_prob,
                "is_malicious": mal_prob >= threshold,
            })
    return results


# ============================================================
# STREAMLIT UI
# ============================================================

# page setup
st.set_page_config(
    page_title="Malicious Call Detector",
    page_icon="🛡️",
    layout="wide",
)

# custom styling (loaded from style.css)
with open("style.css") as _f:
    st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

# header
st.markdown("# 🛡️ Malicious Call Detector")
st.markdown("Upload an audio file or record from your microphone to detect fraudulent phone calls using deep learning.")
st.divider()

# sidebar: model selection
with st.sidebar:
    st.header("⚙️ Settings")

    selected_model = st.selectbox(
        "Select Model",
        list(MODEL_INFO.keys()),
        index=0,
        help="Choose which trained model to use for prediction."
    )
    st.caption(MODEL_INFO[selected_model]["desc"])

    st.divider()
    st.markdown("**Audio Limits**")
    max_duration = st.slider("Max audio duration (seconds)", 10, 600, 120, step=10)

    st.divider()
    st.markdown("**About**")
    st.markdown(
        "Built as part of a Bachelor's of Science thesis on detecting malicious phone calls "
        "using Convolutional Neural Networks and Mel Spectrograms."
    )
    st.markdown("**Author:** Marinescu Dragoș")

# main content: two tabs (upload and record)
tab_upload, tab_record, tab_compare = st.tabs(["📁 Upload Audio", "🎙️ Record Audio", "⚖️ Compare Models"])


def display_results(audio, sr, model, threshold, device):
    """Shared function to display prediction results."""
    duration = len(audio) / sr

    # create spectrograms
    with st.spinner("Creating spectrograms..."):
        spectrograms, time_ranges = create_spectrograms(audio, sr)

    if len(spectrograms) == 0:
        st.warning("Audio is too short or too silent for analysis. Please provide at least 2.5 seconds of speech.")
        return

    # run prediction
    with st.spinner("Analyzing audio..."):
        results = predict(model, spectrograms, threshold, device)

    # count flagged windows
    flagged = sum(1 for r in results if r["is_malicious"])
    total = len(results)
    mal_ratio = flagged / total
    overall = "🔴 MALICIOUS" if mal_ratio >= 0.5 else "🟢 NORMAL"
    avg_confidence = np.mean([r["malicious_prob"] for r in results]) * 100

    # display overall result
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if mal_ratio >= 0.5:
            st.error(f"**Verdict: MALICIOUS**")
        else:
            st.success(f"**Verdict: NORMAL**")
    with col2:
        st.metric("Flagged Windows", f"{flagged}/{total}", delta=f"{mal_ratio*100:.1f}%")
    with col3:
        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
    with col4:
        st.metric("Duration", f"{duration:.1f}s")

    # risk score bar
    risk = avg_confidence if mal_ratio >= 0.5 else 100 - avg_confidence
    st.markdown(f"**Fraud Risk Score:** {risk:.0f}/100")
    st.progress(min(risk / 100, 1.0))

    # per-window timeline
    st.subheader("Per-Window Timeline")
    cols_per_row = 10
    for row_start in range(0, total, cols_per_row):
        cols = st.columns(min(cols_per_row, total - row_start))
        for i, col in enumerate(cols):
            idx = row_start + i
            r: PredResult = cast(PredResult, results[idx])
            t = time_ranges[idx]
            if r["is_malicious"]:
                col.markdown(f"<div style='background:#ff1744;color:white;padding:4px;border-radius:4px;text-align:center;font-size:11px;'>{t[0]}s<br>{r['malicious_prob']:.0%}</div>", unsafe_allow_html=True)
            else:
                col.markdown(f"<div style='background:#00c853;color:white;padding:4px;border-radius:4px;text-align:center;font-size:11px;'>{t[0]}s<br>{1-r['malicious_prob']:.0%}</div>", unsafe_allow_html=True)

    # spectrogram grid
    st.subheader("Spectrogram Analysis")
    n_specs: int = min(len(spectrograms), 20)  # show max 20 for performance
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
                ax.set_title(f"⚠ MAL ({r['malicious_prob']:.2f})\n{t[0]}s-{t[1]}s", color="red", fontsize=9)
            else:
                ax.set_title(f"✓ NOR ({1-r['malicious_prob']:.2f})\n{t[0]}s-{t[1]}s", color="lime", fontsize=9)
            ax.set_facecolor("#0e1117")
        else:
            ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # show flagged windows detail
    if flagged > 0:
        st.subheader("⚠️ Flagged Segments")
        for i, r in enumerate(results):
            if r["is_malicious"]:
                t = time_ranges[i]
                st.markdown(f"- Window {i+1}: **{t[0]}s → {t[1]}s** — Malicious confidence: `{r['malicious_prob']:.1%}`")


# TAB 1: Upload
with tab_upload:
    uploaded = st.file_uploader(
        "Choose an audio file (WAV or MP3)",
        type=["wav", "mp3"],
        help=f"Maximum duration: {max_duration} seconds"
    )

    if uploaded is not None:
        # save to temp file for librosa to read
        suffix = ".wav" if uploaded.name.endswith(".wav") else ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        # load audio
        audio, sr = librosa.load(tmp_path, sr=16000)
        os.unlink(tmp_path)
        duration = len(audio) / sr

        # check duration limit
        if duration > max_duration:
            st.info(f"Audio is {duration:.0f}s long. Maximum allowed is {max_duration}s. The audio was truncated. You can also adjust the max duration in the sidebar.")
            audio = audio[:int(max_duration * sr)]   
        st.audio(uploaded, format=f"audio/{suffix[1:]}")

        # load model and predict
        model, threshold, error = load_model(selected_model)
        if error:
            st.error(error)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            display_results(audio, sr, model, threshold, device)


# TAB 2: Record
with tab_record:
    st.markdown("Record audio directly from your microphone.")
    recorded = st.audio_input("Click to start recording", help="Record a short clip and analyze it.")

    if recorded is not None:
        # save recorded audio to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(recorded.read())
            tmp_path = tmp.name

        audio, sr = librosa.load(tmp_path, sr=16000)
        os.unlink(tmp_path)
        duration = len(audio) / sr

        if duration > max_duration:
            st.info(f"Recording is {duration:.0f}s. Maximum is {max_duration}s. The audio was truncated. You can also adjust the max duration in the sidebar.")
            audio = audio[:int(max_duration * sr)]
            duration = max_duration
            
        if duration < 2.5:
            st.warning("Recording is too short. Please record at least 3 seconds.")
        else:
            model, threshold, error = load_model(selected_model)
            if error:
                st.error(error)
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                display_results(audio, sr, model, threshold, device)


# TAB 3: Compare Models
with tab_compare:
    st.markdown("Upload an audio file and compare predictions across two models side by side.")

    compare_file = st.file_uploader("Choose audio for comparison", type=["wav", "mp3"], key="compare")

    col_left, col_right = st.columns(2)
    with col_left:
        model_a = st.selectbox("Model A", list(MODEL_INFO.keys()), index=0, key="model_a")
    with col_right:
        model_b = st.selectbox("Model B", list(MODEL_INFO.keys()), index=2, key="model_b")

    if compare_file is not None:
        suffix = ".wav" if compare_file.name.endswith(".wav") else ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(compare_file.read())
            tmp_path = tmp.name

        audio, sr = librosa.load(tmp_path, sr=16000)
        os.unlink(tmp_path)
        duration = len(audio) / sr

        if duration > max_duration:
            st.info(f"Recording is {duration:.0f}s. Maximum is {max_duration}s. The audio was truncated. You can also adjust the max duration in the sidebar.")
            audio = audio[:int(max_duration * sr)]
        st.audio(compare_file, format=f"audio/{suffix[1:]}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"### {model_a}")
            m, t, err = load_model(model_a)
            if err:
                st.error(err)
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                display_results(audio, sr, m, t, device)

        with col_b:
            st.markdown(f"### {model_b}")
            m, t, err = load_model(model_b)
            if err:
                st.error(err)
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                display_results(audio, sr, m, t, device)
  