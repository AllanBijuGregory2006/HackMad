import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import threading
import time
import csv
from scipy import signal
from collections import deque
from pycaw.pycaw import AudioUtilities

# 1. SETUP & AI ENGINE
print("Initializing YAMNet Engine...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')

class_names = []
with open(class_map_path, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        class_names.append(row['display_name'])

# 2. PRE-LOAD WARNING AUDIO INTO RAM
try:
    WARNING_FS, WARNING_DATA = wavfile.read('warning.wav')
    print("✅ Successfully loaded 'warning.wav' into memory.")
except FileNotFoundError:
    print("❌ ERROR: 'warning.wav' not found! Please place it in the same folder.")
    WARNING_DATA = None

is_alarming = False

# 3. DIGITAL SIGNAL CONDITIONING
def bandpass_filter(audio, lowcut=2000, highcut=4000, fs=16000):
    nyq = fs / 2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.filtfilt(b, a, audio)

# 4. CONFIGURATION
TARGET_SOUND = "Fire alarm"
DETECTION_THRESHOLD = 0.15 
buffer_size = 3
score_buffer = deque(maxlen=buffer_size)

# 5. THE WINDOWS SYSTEM INTERRUPT THREAD
def play_warning_audio(duration):
    global is_alarming
    is_alarming = True
    print(f"\n🔊 SYSTEM INTERRUPT: Muting OS and playing warning for {duration} seconds...")
    
    try:
        # A. Force Mute all other Windows applications
        for session in AudioUtilities.GetAllSessions():
            if session.Process and session.Process.name() != "python.exe":
                session.SimpleAudioVolume.SetMute(1, None)
                
        # B. Play the warning sound safely in this thread
        if WARNING_DATA is not None:
            sd.play(WARNING_DATA, WARNING_FS, loop=True)
            
        # C. Keep the thread alive while the audio plays
        time.sleep(duration)
        
    finally:
        # D. SAFE RECOVERY: Stop the warning and UNMUTE the OS
        sd.stop()
        for session in AudioUtilities.GetAllSessions():
            if session.Process:
                session.SimpleAudioVolume.SetMute(0, None)
        
        print("\n🔇 Warning complete. Windows audio restored. Resuming active monitoring...")
        time.sleep(2) # Cooldown to prevent instant re-triggering
        is_alarming = False

# 6. TRIGGER PROTOCOL
def alarm_protocol(avg_score):
    global is_alarming
    if not is_alarming:
        print(f"\n🚨 VERIFIED ALARM TRIGGERED! Avg Conf: {avg_score:.2f}")
        # Spawn the background worker to handle the OS muting and playback
        threading.Thread(target=play_warning_audio, args=(15,), daemon=True).start()

# 7. CONCURRENT CALLBACK (The ISR)
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    
    # --- DSP STAGE ---
    raw_waveform = np.squeeze(indata).astype(np.float32)
    filtered_waveform = bandpass_filter(raw_waveform).astype(np.float32)

    # --- FFT VETO STAGE ---
    fft_spectrum = np.abs(np.fft.rfft(raw_waveform))
    fft_freqs = np.fft.rfftfreq(len(raw_waveform), d=1.0/16000.0)
    
    # Ignore DC offset and room hum (< 100 Hz)
    valid_indices = np.where(fft_freqs > 100)[0]
    if len(valid_indices) > 0:
        peak_freq_index = valid_indices[np.argmax(fft_spectrum[valid_indices])]
        peak_freq = fft_freqs[peak_freq_index]
    else:
        peak_freq = 0

    # --- AI STAGE ---
    scores, embeddings, spectrogram = yamnet_model(filtered_waveform)
    mean_scores = tf.reduce_mean(scores, axis=0).numpy()
    
    top_class_index = np.argmax(mean_scores)
    main_prediction = class_names[top_class_index]
    
    target_idx = class_names.index(TARGET_SOUND)
    current_target_score = mean_scores[target_idx]
    
    score_buffer.append(current_target_score)
    avg_score = sum(score_buffer) / len(score_buffer)

    print(f"Winner: {main_prediction:12s} | Alarm Conf: {avg_score:.2f} | Peak: {peak_freq:.0f} Hz    ", end='\r')

    # --- TRIGGER LOGIC ---
    if len(score_buffer) == buffer_size and avg_score > DETECTION_THRESHOLD:
        # Physical Veto: Ensure the loudest room sound is actually high-pitched
        if peak_freq > 800: 
            alarm_protocol(avg_score)
            score_buffer.clear()

# 8. MONITORING LOOP
block_duration = 0.5 
try:
    with sd.InputStream(samplerate=16000, channels=1, callback=audio_callback, 
                       blocksize=int(16000 * block_duration)):
        print(f"\nSystem Active. DSP, FFT Veto, and Windows API enabled.")
        print(f"Monitoring for {TARGET_SOUND}...")
        while True:
            time.sleep(0.1)
except KeyboardInterrupt:
    print("\nMonitor Stopped.")