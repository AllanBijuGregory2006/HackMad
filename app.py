import threading
import csv
import time
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
from scipy import signal
from collections import deque
from flask import Flask, jsonify, send_from_directory
import os
from flask_cors import CORS
import tensorflow as tf
import tensorflow_hub as hub
from pycaw.pycaw import AudioUtilities

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────
#  1. LOAD YAMNET
# ─────────────────────────────────────────
print("Loading YAMNet model...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
class_names = []
with open(class_map_path, 'r') as f:
    for row in csv.DictReader(f):
        class_names.append(row['display_name'])
print("YAMNet ready.")

# ─────────────────────────────────────────
#  PRE-LOAD WARNING AUDIO INTO RAM
# ─────────────────────────────────────────
try:
    _folder = os.path.dirname(os.path.abspath(__file__))
    WARNING_FS, WARNING_DATA = wavfile.read(os.path.join(_folder, 'warning.wav'))
    print("✅ Successfully loaded 'warning.wav' into memory.")
except FileNotFoundError:
    print("❌ ERROR: 'warning.wav' not found! Place it in the same folder as app.py.")
    WARNING_DATA = None
    WARNING_FS   = 16000

is_alarming = False

# ─────────────────────────────────────────
#  2. CONFIGURATION
# ─────────────────────────────────────────
TARGET_SOUND       = "Fire alarm"
DETECTION_THRESHOLD = 0.15
BUFFER_SIZE        = 3
BLOCK_DURATION     = 0.5   # seconds per audio chunk

CRITICAL_SOUNDS = [
    'Fire alarm', 'Smoke detector', 'Alarm',
    'Siren', 'Civil defense siren', 'Ambulance (siren)'
]

# ─────────────────────────────────────────
#  3. SHARED STATE (read by browser via /status)
# ─────────────────────────────────────────
state = {
    'is_alert':       False,
    'top_sound':      'Ambient noise',
    'top_confidence': 0.0,
    'target_score':   0.0,
    'avg_score':      0.0,
    'results':        [],        # top-5 classifications for the live feed
    'latency_ms':     0,
}
state_lock = threading.Lock()

score_buffer = deque(maxlen=BUFFER_SIZE)

# ─────────────────────────────────────────
#  4. DSP — BANDPASS FILTER
# ─────────────────────────────────────────
def bandpass_filter(audio, lowcut=2000, highcut=4000, fs=16000):
    nyq  = fs / 2
    b, a = signal.butter(4, [lowcut / nyq, highcut / nyq], btype='band')
    return signal.filtfilt(b, a, audio)

# ─────────────────────────────────────────
#  5. WINDOWS SYSTEM INTERRUPT — WARNING PLAYBACK
# ─────────────────────────────────────────
def play_warning_audio(duration=15):
    global is_alarming
    is_alarming = True
    print(f"\n🔊 SYSTEM INTERRUPT: Muting OS and playing warning for {duration} seconds...")

    try:
        # A. Force mute all other Windows applications
        for session in AudioUtilities.GetAllSessions():
            try:
                if session.Process and session.Process.name() != "python.exe":
                    session.SimpleAudioVolume.SetMute(1, None)
            except Exception:
                pass  # Skip dead/inaccessible processes

        # B. Play the warning sound on loop
        if WARNING_DATA is not None:
            sd.play(WARNING_DATA, WARNING_FS, loop=True)

        # C. Keep the thread alive while audio plays
        time.sleep(duration)

    finally:
        # D. SAFE RECOVERY: Stop warning and unmute OS
        sd.stop()
        for session in AudioUtilities.GetAllSessions():
            try:
                if session.Process:
                    session.SimpleAudioVolume.SetMute(0, None)
            except Exception:
                pass  # Skip dead/inaccessible processes

        print("\n🔇 Warning complete. Windows audio restored. Resuming active monitoring...")
        time.sleep(2)  # Cooldown to prevent instant re-triggering
        is_alarming = False

def alarm_protocol(avg_score):
    global is_alarming
    if not is_alarming:
        print(f"\n🚨 VERIFIED ALARM TRIGGERED! Avg Conf: {avg_score:.2f}")
        # Spawn background worker to handle OS muting and playback
        threading.Thread(target=play_warning_audio, args=(15,), daemon=True).start()

# ─────────────────────────────────────────
#  6. AUDIO CALLBACK (runs on every mic chunk)
# ─────────────────────────────────────────
def audio_callback(indata, frames, time_info, status):
    t0 = time.time()

    if status:
        print(status)

    # --- DSP STAGE ---
    raw_waveform      = np.squeeze(indata).astype(np.float32)
    filtered_waveform = bandpass_filter(raw_waveform).astype(np.float32)

    # FAST FOURIER TRANSFORM (FFT)
    fft_spectrum = np.abs(np.fft.rfft(raw_waveform))
    # 'd' must be the time per sample (1 / sample_rate)
    fft_freqs = np.fft.rfftfreq(len(raw_waveform), d=1.0 / 16000.0)

    # Ignore DC offset (0 Hz) and room hum (< 100 Hz)
    valid_indices   = np.where(fft_freqs > 100)[0]
    peak_freq_index = valid_indices[np.argmax(fft_spectrum[valid_indices])]
    peak_freq       = fft_freqs[peak_freq_index]

    # --- AI STAGE ---
    # Feed raw (unfiltered) audio to YAMNet so all sounds classify correctly.
    # Bandpass filter is only used for the FFT veto gate above.
    scores, _, _ = yamnet_model(raw_waveform)
    mean_scores  = tf.reduce_mean(scores, axis=0).numpy()

    # Top-5 for the live feed panel
    top5_idx = np.argsort(mean_scores)[::-1][:5]
    results  = [
        {'sound': class_names[i], 'confidence': round(float(mean_scores[i]), 3)}
        for i in top5_idx
    ]

    # Winner
    top_idx   = top5_idx[0]
    top_sound = class_names[top_idx]
    top_conf  = float(mean_scores[top_idx])

    # Target tracking
    target_idx   = class_names.index(TARGET_SOUND)
    target_score = float(mean_scores[target_idx])
    score_buffer.append(target_score)
    avg_score = sum(score_buffer) / len(score_buffer)

    # UI Feedback
    print(
        f"Winner: {top_sound:12s} | Alarm Conf: {avg_score:.2f} | Peak: {peak_freq:.0f} Hz    ",
        end='\r'
    )

    # --- TRIGGER LOGIC WITH REFINED FFT VETO ---
    # VETO GATE: Human voices peak < 500 Hz. Alarms peak > 800 Hz.
    is_alert = False
    if len(score_buffer) == BUFFER_SIZE and avg_score > DETECTION_THRESHOLD:
        if peak_freq > 800:
            is_alert = True
            score_buffer.clear()
            alarm_protocol(avg_score)
        else:
            print(f"\n[VETO] Score {avg_score:.2f} exceeded threshold but peak_freq={peak_freq:.0f}Hz is too low — blocked")
    else:
        if avg_score > 0.05:
            print(f"\n[DEBUG] avg={avg_score:.2f} threshold={DETECTION_THRESHOLD} buffer={len(score_buffer)}/{BUFFER_SIZE} peak={peak_freq:.0f}Hz is_alarming={is_alarming}")

    if is_alert:
        # Use the highest-confidence critical sound as the label
        alert_sound = TARGET_SOUND
        alert_conf  = target_score
        for item in results:
            if any(c.lower() in item['sound'].lower() for c in CRITICAL_SOUNDS):
                alert_sound = item['sound']
                alert_conf  = item['confidence']
                break
    else:
        alert_sound = top_sound
        alert_conf  = top_conf

    latency = int((time.time() - t0) * 1000)

    with state_lock:
        state['is_alert']       = is_alert
        state['top_sound']      = alert_sound
        state['top_confidence'] = round(alert_conf * 100, 1)
        state['target_score']   = round(target_score, 3)
        state['avg_score']      = round(avg_score, 3)
        state['results']        = results
        state['latency_ms']     = latency

# ─────────────────────────────────────────
#  7. START MIC STREAM IN BACKGROUND THREAD
# ─────────────────────────────────────────
def run_mic_stream():
    print(f"\nMic stream active — Bandpass filter (2kHz–4kHz) ON")
    print(f"Monitoring for: {TARGET_SOUND}\n")
    with sd.InputStream(
        samplerate=16000,
        channels=1,
        callback=audio_callback,
        blocksize=int(16000 * BLOCK_DURATION)
    ):
        while True:
            time.sleep(0.1)

mic_thread = threading.Thread(target=run_mic_stream, daemon=True)
mic_thread.start()

# ─────────────────────────────────────────
#  8. FLASK ROUTES — browser polls these
# ─────────────────────────────────────────
@app.route('/status')
def status():
    """
    Browser polls this every second.
    Returns current classification state as JSON.
    """
    with state_lock:
        return jsonify({
            'is_alert':       state['is_alert'],
            'top_sound':      state['top_sound'],
            'top_confidence': state['top_confidence'],
            'results':        state['results'],
            'latency_ms':     state['latency_ms'],
        })

@app.route('/')
def index():
    folder = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(folder, 'soundguard.html')

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': 'YAMNet'})

# ─────────────────────────────────────────
#  9. RUN
# ─────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 50)
    print("  SoundGuard Backend")
    print("  http://localhost:5000")
    print("  Open soundguard.html in browser")
    print("=" * 50)
    app.run(port=5000, debug=False)
