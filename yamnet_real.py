import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sounddevice as sd
import time
import csv
from scipy import signal
from collections import deque

# 1. SETUP & AI ENGINE
print("Initializing YAMNet Engine...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')

class_names = []
with open(class_map_path, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        class_names.append(row['display_name'])

# 2. DIGITAL SIGNAL CONDITIONING (Claude's Bandpass Filter)
def bandpass_filter(audio, lowcut=2000, highcut=4000, fs=16000):
    nyq = fs / 2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(4, [low, high], btype='band')
    # filtfilt applies the filter forward and backward to prevent phase shifting
    return signal.filtfilt(b, a, audio)

# 3. CONFIGURATION
TARGET_SOUND = "Fire alarm"
DETECTION_THRESHOLD = 0.15 
buffer_size = 3
score_buffer = deque(maxlen=buffer_size)

def alarm_protocol(avg_score):
    print(f"\n🚨 BANDPASS ALARM TRIGGERED! Avg Conf: {avg_score:.2f}")

# 4. CONCURRENT CALLBACK
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    
    # --- DSP STAGE ---
    raw_waveform = np.squeeze(indata).astype(np.float32)
    filtered_waveform = bandpass_filter(raw_waveform).astype(np.float32)

    # FAST FOURIER TRANSFORM (FFT)
    fft_spectrum = np.abs(np.fft.rfft(raw_waveform))
    
    # FIX: The second argument 'd' must be the time per sample (1 / sample_rate)
    fft_freqs = np.fft.rfftfreq(len(raw_waveform), d=1.0/16000.0)
    
    # Ignore DC offset (0 Hz) and room hum (< 100 Hz)
    valid_indices = np.where(fft_freqs > 100)[0]
    peak_freq_index = valid_indices[np.argmax(fft_spectrum[valid_indices])]
    peak_freq = fft_freqs[peak_freq_index]

    # --- AI STAGE ---
    scores, embeddings, spectrogram = yamnet_model(filtered_waveform)
    mean_scores = tf.reduce_mean(scores, axis=0).numpy()
    
    top_class_index = np.argmax(mean_scores)
    main_prediction = class_names[top_class_index]
    
    target_idx = class_names.index(TARGET_SOUND)
    current_target_score = mean_scores[target_idx]
    
    score_buffer.append(current_target_score)
    avg_score = sum(score_buffer) / len(score_buffer)

    # UI Feedback
    print(f"Winner: {main_prediction:12s} | Alarm Conf: {avg_score:.2f} | Peak: {peak_freq:.0f} Hz    ", end='\r')

    # --- TRIGGER LOGIC WITH REFINED FFT VETO ---
    if len(score_buffer) == buffer_size and avg_score > DETECTION_THRESHOLD:
        
        # VETO GATE: Human voices peak < 500 Hz. Alarms peak > 800 Hz.
        if peak_freq > 2000: 
            alarm_protocol(avg_score)
            score_buffer.clear()
        else:
            # Veto applied. 
            pass

# 5. MONITORING LOOP
block_duration = 0.5 
try:
    with sd.InputStream(samplerate=16000, channels=1, callback=audio_callback, 
                       blocksize=int(16000 * block_duration)):
        print(f"\nSystem Active. Bandpass Filter (2kHz-4kHz) enabled.")
        print(f"Monitoring for {TARGET_SOUND}...")
        while True:
            time.sleep(0.1)
except KeyboardInterrupt:
    print("\nMonitor Stopped.")