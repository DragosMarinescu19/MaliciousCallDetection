import numpy as np
import librosa
import matplotlib.pyplot as plt

sr = 16000
# Noise
y2 = np.random.randn(5 * sr) * 1e-5
print(y2)

mel_spec = librosa.feature.melspectrogram(y=y2, sr=sr, n_fft=2048,hop_length=512, n_mels=128)
log_mel = librosa.power_to_db(mel_spec, ref=np.max)
log_mel = np.clip(log_mel, -80, 0)
print("Before norm:", np.min(log_mel), np.max(log_mel), np.mean(log_mel), np.std(log_mel))
log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-9)
print("After norm:", np.min(log_mel), np.max(log_mel),np.mean(log_mel), np.std(log_mel))

plt.imshow(log_mel, aspect='auto', origin='lower', cmap='magma')
plt.colorbar()
plt.savefig('test_noise.png')
