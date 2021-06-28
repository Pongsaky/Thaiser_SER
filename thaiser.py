import torch
import seaborn as sns
import matplotlib.pyplot as plt
from mymodel.csablstm import CSABLSTM
from vistec_ser.data.features.padding import pad_dup
from vistec_ser.models.network import CNN1DLSTMSlice
from src.components.data import PrepareData

file_path = "src/data/neutral.flac"
sample_rate = 16000
num_mel = 60
max_len = 10
frame_len = 50
frame_shift = 10
data_transform = PrepareData(file_path=file_path, sampling_rate=sample_rate, 
                num_mel=num_mel, max_len=max_len, frame_len=frame_len, frame_shift=frame_shift, pad_fn=pad_dup)

# Data
d1 = torch.rand(60, 300)
print(d1.shape)
data_1 = PrepareData(audio=d1, sampling_rate=sample_rate, 
                num_mel=num_mel, max_len=max_len, frame_len=frame_len, frame_shift=frame_shift, pad_fn=pad_dup)

feature = data_1._load_audio(read_file=False)
print(feature.shape)

'''
hparams = {
    "n_channels": [64, 64, 128, 128],
    "kernel_size": [5, 3, 3, 3],
    "pool_size": [4, 2, 2, 2],
    "lstm_unit": 128,
    "in_channel": num_mel, 
    "sequence_length": max_len * 100, 
    "n_classes": 4,
    "learning_rate": 1e-4
}
model = CSABLSTM(hparams)

model.load_state_dict(torch.load("mymodel/weights/csablstm.pt", map_location=torch.device('cpu')))
model.eval()

result = model(feature)

print(result)
'''