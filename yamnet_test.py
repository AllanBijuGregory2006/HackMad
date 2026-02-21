import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import scipy.io.wavfile as wavfile
import scipy.signal

# 1. Load the YAMNet model directly from the Google "App Store"
print("Downloading and loading YAMNet model (this might take a few seconds)...")
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

# 2. Load the class mapping (the 521 human-readable labels like "Guitar" or "Dog")
class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
class_names = []
with open(class_map_path, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        class_names.append(row['display_name'])

# 3. Helper function to ensure the audio is exactly 16kHz using SciPy
def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
    if original_sample_rate != desired_sample_rate:
        print(f"Resampling audio from {original_sample_rate}Hz to {desired_sample_rate}Hz...")
        desired_length = int(round(float(len(waveform)) / original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform

# 4. Load and process the audio file
# Make sure this file exists in the same folder as this script!
audio_file = 'test_audio.wav' 

try:
    print(f"Loading '{audio_file}'...")
    sample_rate, wav_data = wavfile.read(audio_file)
    
    # If the audio is stereo (2 channels), average it to mono (1 channel)
    if len(wav_data.shape) > 1:
        wav_data = np.mean(wav_data, axis=1)
        
    sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)
    
    # Normalize the waveform to be between -1.0 and 1.0 using NumPy
    waveform = wav_data / tf.int16.max
    waveform = waveform.astype(np.float32)

    # 5. Run the audio through the TensorFlow engine!
    print("Analyzing soundwaves...")
    scores, embeddings, spectrogram = yamnet_model(waveform)
    
    # YAMNet scores every 0.48 seconds. We average them to get the overall clip prediction.
    class_scores = tf.reduce_mean(scores, axis=0)
    top_class_index = tf.math.argmax(class_scores)
    inferred_class = class_names[top_class_index]

    print(f'\n🎉 Success! The main sound detected is: {inferred_class}')

except FileNotFoundError:
    print(f"\n❌ Error: Could not find '{audio_file}'. Please make sure you have a .wav file in the same folder.")