import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import soundfile as sf
import csv
import os
import matplotlib.pyplot as plt
import scipy.signal


# Load model
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
class_names = []
with open(class_map_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        class_names.append(row['display_name'])

TARGET = "Fire alarm"
target_idx = class_names.index(TARGET)
THRESHOLD = 0.15

#Test files
test_files = {
    # Class A - Alarms/Emergency Soundes
    'fire_alarm.wav': 1,
    'fire_alarm_2.wav': 1,
    'school_alarm.wav': 1,
    'smoke_detector.wav': 1,
    'siren.wav': 1,
    'chernobyl_alarm.wav': 1, 
    # Class B - Fake Alarms like sounds/False Positives
    'Cymbals_alarm_like.wav': 0,
    'metal_clang.wav': 0,
    'power_drill.wav': 0,
    'hammer.wav': 0,
    'factory.wav': 0,
    # Class C - Normal Sounds
    'speech.wav': 0,
    'music.wav': 0,
    'dog.wav': 0,
    'keyboard.wav': 0,
    'crowd_noise.wav': 0,
    'factory_atmosphere.wav': 0,
}

results = []

for filename, is_alarm in test_files.items():
    path = f"test_sounds/{filename}"
    audio, sr = sf.read(path)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)

    if sr != 16000:
        target_length = int(len(audio) * 16000 / sr)
        audio = scipy.signal.resample(audio, target_length)

    # Normalize audio
    audio = audio / (np.max(np.abs(audio)) + 1e-9)
    scores, _, _ = yamnet_model(audio)
    mean_scores = tf.reduce_mean(scores, axis=0).numpy()
    fire_score = mean_scores[target_idx]
    triggered = fire_score > THRESHOLD
    correct = (triggered == bool(is_alarm))
    
    results.append({
        'file': filename,
        'is_alarm': is_alarm,
        'score': fire_score,
        'triggered': triggered,
        'correct': correct
    })
    
    status = "Success" if correct else "Fail"
    print(f"{status} {filename:25s} | Score: {fire_score:.2f} | Triggered: {triggered}")

# Calculate precision and recall
true_positives = sum(1 for r in results if r['is_alarm'] and r['triggered'])
false_positives = sum(1 for r in results if not r['is_alarm'] and r['triggered'])
false_negatives = sum(1 for r in results if r['is_alarm'] and not r['triggered'])

recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

print(f"\nRecall (catches real alarms): {recall*100:.0f}%")
print(f"Precision (avoids false alarms): {precision*100:.0f}%")

# Make chart
files = [r['file'].replace('.wav','') for r in results]
scores = [r['score'] for r in results]
colors = ['red' if r['is_alarm'] else 'steelblue' for r in results]

plt.figure(figsize=(14,5))
plt.bar(files, scores, color=colors)
plt.axhline(y=THRESHOLD, color='black', linestyle='--', label=f'Threshold ({THRESHOLD})')
plt.title('SoundGuard Validation: YAMNet Fire Alarm Confidence by Sound Type')
plt.ylabel('Confidence Score')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('validation_results.png')
print("\nChart saved!")







