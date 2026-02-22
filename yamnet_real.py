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
    # Convert hardware buffer to float32
    waveform = np.squeeze(indata).astype(np.float32)
    
    # Apply the Bandpass Filter to aggressively mute human speech
    waveform = bandpass_filter(waveform).astype(np.float32)

    # --- AI STAGE ---
    scores, embeddings, spectrogram = yamnet_model(waveform)
    mean_scores = tf.reduce_mean(scores, axis=0).numpy()
    
    # Find the "Winner" (this should no longer say "Speech" very often!)
    top_class_index = np.argmax(mean_scores)
    main_prediction = class_names[top_class_index]
    main_conf = mean_scores[top_class_index]

    # Target Detection
    target_idx = class_names.index(TARGET_SOUND)
    current_target_score = mean_scores[target_idx]
    
    score_buffer.append(current_target_score)
    avg_score = sum(score_buffer) / len(score_buffer)

    print(f"Loudest Filtered: {main_prediction:12s} ({main_conf:.2f}) | {TARGET_SOUND} Current: {current_target_score:.2f} | Avg: {avg_score:.2f}    ", end='\r')

    # Trigger Logic
    if len(score_buffer) == buffer_size and avg_score > DETECTION_THRESHOLD:
        alarm_protocol(avg_score)
        score_buffer.clear()

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