import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sounddevice as sd
import time
import csv

# 1. SETUP: Load the model once at the start (The "Brain")
print("Initializing YAMNet...")
# We load the weights into memory; this is the core neural network.
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Get the path to the class map file from the model itself
class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')

# Properly parse the CSV to get human-readable names
class_names = []
with open(class_map_path, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        class_names.append(row['display_name'])
        
print(f"Loaded {len(class_names)} audio classes.")

# 2. CONFIGURATION: Alarm Thresholds
TARGET_SOUND = "Fire alarm"  # You can change this to "Siren" or "Screaming" if needed
THRESHOLD = 0.30             # Lowered for real-world microphone capture

def alarm_protocol(score):
    print(f"\n🚨 ALARM TRIGGERED! {TARGET_SOUND} detected with {score:.2f} confidence!")
    # Future expansion: Trigger hardware GPIO or send a notification

# 3. REAL-TIME PROCESSING LOOP (The Callback / ISR)
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    
    # Pre-processing: Convert hardware buffer to float32 for the AI
    waveform = np.squeeze(indata).astype(np.float32)

    # Inference: The mathematical core of the model
    scores, embeddings, spectrogram = yamnet_model(waveform)
    
    # Average the scores over the 0.5s block
    mean_scores = tf.reduce_mean(scores, axis=0)
    
    # DIAGNOSTIC: Get top 3 predictions to see what the AI "hears"
    top_indices = np.argsort(mean_scores.numpy())[::-1][:3]
    top_predictions = [f"{class_names[i]}: {mean_scores[i].numpy():.2f}" for i in top_indices]
    
    # Display the real-time stream
    print(f"Top 3: {' | '.join(top_predictions)}", end='\r')

    # Alarm Logic: Check if our TARGET_SOUND index exceeds the threshold
    try:
        target_index = class_names.index(TARGET_SOUND)
        if mean_scores[target_index] > THRESHOLD:
            alarm_protocol(mean_scores[target_index].numpy())
    except ValueError:
        pass # Target sound not in class list

# 4. START THE STREAM
# 16kHz mono is the strict requirement for YAMNet
block_duration = 0.5 
print(f"Monitoring for {TARGET_SOUND}...")

try:
    with sd.InputStream(samplerate=16000, channels=1, callback=audio_callback, 
                       blocksize=int(16000 * block_duration)):
        print("Press Ctrl+C to stop.")
        while True:
            time.sleep(0.1)
except KeyboardInterrupt:
    print("\nStopping monitor...")
except Exception as e:
    print(f"\n❌ Hardware Error: {e}")