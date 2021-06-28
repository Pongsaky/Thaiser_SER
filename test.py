import sounddevice as sd
import soundfile as sf

samplerate = 16000  # Sample rate
seconds = 3  # Duration of recording

myrecording = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=2)
print("Record start")
sd.wait()  # Wait until recording is finished
print(myrecording)
sf.write("Recording.flac", myrecording, 16000)
print("Record finished")
