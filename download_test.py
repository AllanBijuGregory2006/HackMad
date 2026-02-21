import tensorflow as tf

# Downloads a 16kHz mono meow sound from Google's AudioSet
testing_wav_file_name = tf.keras.utils.get_file(
    'test_audio.wav',
    'https://storage.googleapis.com/audioset/miaow_16k.wav',
    cache_dir='./',
    cache_subdir='.'
)

print(f"File downloaded successfully to: {testing_wav_file_name}")