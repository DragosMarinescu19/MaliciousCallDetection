import numpy as np
import librosa

def is_silent(audio_segment, sr, top_db=30, min_active_time=2.0):
    #using librosa maximum values are [-1,1]
    if np.max(np.abs(audio_segment)) < 1e-5: #if whole sound is blank
        return True
    
    intervals = librosa.effects.split(audio_segment, top_db=top_db)
    if len(intervals) == 0:
        return True
    active_samples = sum([end - start for start, end in intervals])
    active_time = active_samples / sr
    return active_time < min_active_time
sr = 16000
#Zeros
y1 = np.zeros(5 * sr)
print("Zeros is_silent:", is_silent(y1, sr))

# Low noise
y2 = np.random.randn(5 * sr) * 1e-5
print("Low noise is_silent:", is_silent(y2, sr))

# One spike then zeros
y3 = np.zeros(5 * sr)
y3[0] = 1.0 # One peak
print("Spike is_silent:", is_silent(y3, sr))
