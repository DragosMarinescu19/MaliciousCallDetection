# 🛡️ Malicious Call Detection

> Detecting fraudulent phone calls using **Mel Spectrograms** and **Convolutional Neural Networks** — built as a Bachelor's thesis at Babeș-Bolyai University, Cluj-Napoca.


---
## Web App available at https://maliciouscalldetection-marinescudragos.streamlit.app/


## Overview

Malicious phone calls — robocalls, human-led scam calls, and partially fake audio — cause billions of dollars in losses every year. With AI-generated voice becoming nearly indistinguishable from real speech, traditional detection methods fall short.

This project tackles the problem through **acoustic analysis only**, independent of semantic content or call metadata:

1. Audio is segmented into **5-second overlapping windows**
2. Each window is transformed into a **128-band Log-Mel Spectrogram**
3. A **CNN classifies** each spectrogram as *normal* or *malicious*
4. Results are aggregated into a per-call verdict with a fraud risk score

The final model (**EnhancedCNN**, 2.46M parameters) achieves:

| Metric | Normal Calls | Malicious Calls |
|--------|:------------:|:---------------:|
| **Recall** | 96.76% | 99.83% |

*Evaluated on a fully external dataset unseen during training.*

---

## Architecture

The **EnhancedCNN** is a VGGNet-inspired architecture with 4 convolutional blocks:

```
Input (1 × 128 × 157)
  │
  ├─ Block 1: Conv2d(64) × 2 → BN → ReLU → MaxPool → Dropout(0.1)
  ├─ Block 2: Conv2d(128) × 2 → BN → ReLU → MaxPool → Dropout(0.15)
  ├─ Block 3: Conv2d(256) × 2 → BN → ReLU → MaxPool → Dropout(0.2)
  └─ Block 4: Conv2d(512) → BN → ReLU → MaxPool
       │
       AdaptiveAvgPool2d(1, 1)
       │
       FC(512 → 256) → ReLU → Dropout → FC(256 → 2)
       │
     Output: [P(normal), P(malicious)]
```

Key design choices:
- **Double convolutions** per block enable learning more complex non-linear transformations before spatial reduction
- **Progressive dropout** (0.1 → 0.15 → 0.2) regularizes deeper layers more aggressively
- **AdaptiveAvgPool2d** makes the network input-size agnostic and significantly reduces parameter count
- **Telephony simulation augmentation** (bandpass filter, resampling, noise injection, clipping, volume variation) prevents shortcut learning from audio quality artifacts

---

## Dataset

The training dataset contains **110,000+ spectrograms** compiled from diverse sources:

| Source | Type | Description |
|--------|------|-------------|
| [Robocall Audio Dataset](https://github.com/wspr-ncsu/robocall-audio-dataset) | Malicious | 1,101 real robocall recordings (16 kHz WAV) |
| [RFP Dataset](https://zenodo.org/records/10202142) | Malicious | Partially Fake & Audio-With-Noise samples |
| [CallHome](https://catalog.ldc.upenn.edu/LDC97S42) | Normal | 120+ genuine phone conversations |
| YouTube (curated) | Both | Scam call compilations + podcasts/interviews |
| gTTS + Splicing | Malicious | Partially Fake audio: synthetic voice inserted into real speech at random intervals |

All audio is resampled to **16 kHz mono** before spectrogram generation.

---

## Web Application

The Streamlit-based UI offers three modes:

| Tab | Description |
|-----|-------------|
| 📁 **Upload Audio** | Upload WAV/MP3 files for analysis |
| 🎙️ **Record Audio** | Record directly from your microphone |
| ⚖️ **Compare Models** | Run two models side-by-side on the same audio |

The interface displays:
- **Verdict** (Normal / Malicious) with fraud risk score
- **Per-window timeline** with color-coded confidence bars
- **Spectrogram visualizations** for each analyzed window
- **Flagged segments** with exact timestamps

---

### Local Setup

```bash
# Clone the repository
git clone https://github.com/DragosMarinescu19/MaliciousCallDetection.git
cd MaliciousCallDetection

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

> **Note:** Pre-trained model weights (`.pt` files) must be present in the `models/` directory. The repository includes 22 model versions spanning the full experimental evolution (v1 through v22).

---

## Project Structure

```
MaliciousCallDetection/
├── app.py                  # Streamlit entry point
├── models.py               # CNN architectures (OldCNN, CNN, EnhancedCNN)
├── audio_utils.py          # Spectrogram generation, model loading, inference
├── ui_parts.py             # UI tabs: upload, record, compare
├── style.css               # Custom Streamlit styling
├── requirements.txt        # Python dependencies
├── models/                 # Pre-trained weights (.pt files, v1–v22)
├── dataset/                # Dataset metadata and labels
├── test_model.py           # Model architecture debugging with shape tracing
├── test_batch_leakage.py   # Verifies no cross-sample gradient leakage
├── test_silence.py         # Silence detection unit tests
├── test_noise.py           # Noise-only spectrogram sanity check
├── test_simulate_telephony.py  # Telephony augmentation playback test
├── .devcontainer/          # GitHub Codespaces configuration
└── Licenta_Marinescu_Dragos.pdf  # Full bachelor thesis (Romanian)
```

---

## Available Models

| Model | Architecture | Params | Highlights |
|-------|-------------|--------|------------|
| **V17** | EnhancedCNN | ~2.46M | Telephony simulation augmentation |
| **V18** | EnhancedCNN | ~2.46M | Optuna-tuned, balanced dataset split |
| **V19** | EnhancedCNN | ~2.46M | Optuna hyperparameter optimization |
| **V22** | EnhancedCNN | ~2.46M | Most optimized + Audio-With-Noise data |

Models can be selected in the sidebar at runtime. Earlier versions (v1–v16) document the iterative evolution from a ~9,300-parameter OldCNN to the final EnhancedCNN.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch |
| Audio Processing | librosa, soundfile, pydub |
| Spectrogram Features | 128-band Log-Mel, Z-score normalized |
| Hyperparameter Tuning | Optuna |
| Experiment Tracking | Weights & Biases |
| Web Interface | Streamlit |
| Visualization | Matplotlib, Seaborn |
| Containerization | Dev Containers (Python 3.11) |

---

##Limitations 

Every device has it's own recording system which compresses sound differently, thus creating different spectral artifacts. Simulating the telephone channel made the model generalize better but it can still have a hard time detecting audios coming from non-proffesional microphones (such as laptops or mobile phones) 

---

## Citation

If you use this work, please cite the thesis:

```
Marinescu, D. (2026). Detectarea apelurilor malițioase folosind spectrograme audio
și rețele neuronale convoluționale. Bachelor's Thesis, Babeș-Bolyai University,
Faculty of Mathematics and Computer Science.
```

---

## License

This project is licensed under the [MIT License](LICENSE).

**Author:** Dragoș Marinescu · Babeș-Bolyai University, Cluj-Napoca · 2026
