import scipy.signal
import numpy as np
import librosa
import sounddevice as sd
import scipy.signal

def simulate_telephone(audio, sr=16000, prob=1.0):
    """Randomly degrade audio to simulate phone recording conditions."""
    if np.random.random() > prob:
        return audio  # 50% keep original clean audio

    augmented = audio.copy()

    # 1. Bandpass filter 300-3400 Hz (phone codec bandwidth)
    if np.random.random() < 0.0:
        low = np.random.uniform(250, 400) / (sr / 2)
        high = min(np.random.uniform(3200, 3800) / (sr / 2), 0.99)
        b, a = scipy.signal.butter(4, [low, high], btype='band')
        augmented = scipy.signal.lfilter(b, a, augmented).astype(np.float32)

    # 2. Downsample->upsample (codec quality loss)
    if np.random.random() < 0.0:
        target_sr = np.random.choice([8000, 11025])
        down = librosa.resample(augmented, orig_sr=sr, target_sr=target_sr)
        augmented = librosa.resample(down, orig_sr=target_sr, target_sr=sr)

    # 3. Background noise
    if np.random.random() < 0.0:
        snr_db = np.random.uniform(10, 25)
        noise = np.random.randn(len(augmented)).astype(np.float32)
        signal_power = np.mean(augmented ** 2) + 1e-10
        noise_power = signal_power / (10 ** (snr_db / 10))
        augmented = augmented + noise * np.sqrt(noise_power)

    # 4. Slight clipping (speaker distortion) # 60-90% from the max amplitude is clipped 
    if np.random.random() < 0.0: # it doesnt appear that often(speaker yells, microphone too close to mouth)
        clip_val = np.random.uniform(0.6, 0.9) * (np.max(np.abs(augmented)) + 1e-10) 
        augmented = np.clip(augmented, -clip_val, clip_val)

    # 5. Volume variation
    augmented = augmented * np.random.uniform(0.5, 1.5)
    max_val = np.max(np.abs(augmented))
    if max_val > 1.0:
        augmented = augmented / max_val

    return augmented

segment,_=librosa.load("C:\\Users\\marin\\OneDrive\\Desktop\\LICENTA\\MaliciousCallDetection\\RFP_VC_00002.wav",sr=16000)
simulate_telephone(segment)
