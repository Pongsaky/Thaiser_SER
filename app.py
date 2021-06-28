import numpy as np
import streamlit as st
import sounddevice as sd
import soundfile as sf
import torch
from torch.utils import data
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
    st.text('Record start')
    sd.wait()  # Wait until recording is finished
    st.text('Record finished')
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

st.title("Thai Speech emotion recognition")

samplerate = st.number_input("Samplerate : ", min_value=16000, step=1)
duration = st.number_input("Record duration : ", min_value=1, step=1)

if st.button("Record"):
    record(duration, samplerate)
    data_transform = PrepareData(file_path=file_path, sampling_rate=sample_rate, 
                num_mel=num_mel, max_len=max_len, frame_len=frame_len, frame_shift=frame_shift, pad_fn=pad_dup)
    feature = data_transform._load_audio()
    a_model = all_model(feature)
    result_cnnlstm = predict(a_model.cnnlstm())
    result_cnnblstm = predict(a_model.cnnblstm())
    result_csablstm = predict(a_model.csablstm())
    st.write(f'Emotion CNNLSTM : {result_cnnlstm}')
    st.write(f'Emotion CNNBLSTM : {result_cnnblstm}')
    st.write(f'Emotion CSABLSTM : {result_csablstm}')
    #st.text(a_model.cnnlstm)
    




    

