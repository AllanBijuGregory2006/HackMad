import sounddevice as sd
import numpy as np



channels = 1
samplerate = 16000





def callback(indata, frames, time, status):
    if status:
        print(status)
    print (indata)
    # Process the audio data in 'indata' as needed
    

with sd.InputStream(samplerate = 16000, channels = 1, callback=callback):
    #Not closing the stream.
    input("press enter to stop")



   





