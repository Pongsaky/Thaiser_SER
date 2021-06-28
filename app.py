import time
import sounddevice as sd
import soundfile as sf
import torch
from vistec_ser.data.features.padding import pad_dup
from mymodel.my_model import all_model
from src.components.data import PrepareData

file_path = "Recording.flac"
sample_rate = 16000
num_mel = 60
max_len = 10
frame_len = 50
frame_shift = 10

def record(duration, samplerate):
    myrecording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    print('Record start')
    for i in range(duration):
        print(i, end=" ")
        time.sleep(1)
    sd.wait()  # Wait until recording is finished
    print('Record finsihed')
    sf.write("Recording.flac", myrecording, samplerate=samplerate)

def predict(result):
    print(result.shape)
    result = torch.mean(result, axis=0)
    result_arg = result.argmax()
    #print(result, result.shape)
    if result_arg == 0:
        return "Neutral"
    elif result_arg == 1:
        return "Angry"
    elif result_arg == 2:
        return "Happy"
    else:
        return "Sad"

print("Welcome to Thai Speech emotion recognition")

while True:

    mode = int(input("Mode \n- enter 1 Recording\n- enter 2 exit\n "))

    if mode ==1:
        duration = int(input("Duration to recode (int only) : "))
        if type(duration) != int:
            print("Please type integer number") 
        record(duration, sample_rate)
        data_transform = PrepareData(file_path=file_path, sampling_rate=sample_rate, 
                num_mel=num_mel, max_len=max_len, frame_len=frame_len, frame_shift=frame_shift, pad_fn=pad_dup)
        feature = data_transform._load_audio()
        a_model = all_model(feature)
        result_cnnlstm = predict(a_model.cnnlstm())
        result_cnnblstm = predict(a_model.cnnblstm())
        result_csablstm = predict(a_model.csablstm())

        print(f'Emotion CNNLSTM : {result_cnnlstm}')
        print(f'Emotion CNNBLSTM : {result_cnnblstm}')
        print(f'Emotion CSABLSTM : {result_csablstm}')
    elif (mode==2):
        break
    else:
        print("Please type correct number")




    

