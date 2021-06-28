import torch
from .csablstm import CSABLSTM
from vistec_ser.models.network import CNN1DLSTMSlice, CNN1DLSTMAttentionSlice

class all_model:
    def __init__(self, feature):
        self.feature = feature

    def cnnlstm(self):
        hparams = {"n_channels": [64, 64, 128, 128],
                        "kernel_size": [5, 3, 3, 3],
                        "pool_size": [4, 2, 2, 2],
                        "lstm_unit": 128,
                        "in_channel": 60, 
                        "sequence_length": 10 * 100, 
                        "n_classes": 4,
                        "learning_rate": 1e-4}
        m_cnnlstm = CNN1DLSTMSlice(hparams)
        m_cnnlstm.load_state_dict(torch.load("mymodel/weights/cnnlstm.pt", map_location=torch.device('cpu')))
        m_cnnlstm.eval()
        result = m_cnnlstm(self.feature)
        return result

    def cnnblstm(self):
        hparams = {"n_channels": [64, 64, 128, 128],
                        "kernel_size": [5, 3, 3, 3],
                        "pool_size": [4, 2, 2, 2],
                        "lstm_unit": 128,
                        "in_channel": 60, 
                        "sequence_length": 10 * 100, 
                        "n_classes": 4,
                        "learning_rate": 1e-4}
        m_cnnblstm = CNN1DLSTMAttentionSlice(hparams)
        m_cnnblstm.load_state_dict(torch.load("mymodel/weights/cnnblstm.pt", map_location=torch.device('cpu')))
        m_cnnblstm.eval()
        result = m_cnnblstm(self.feature)
        return result

    def csablstm(self):
        hparams = {"n_channels": [64, 64, 128, 128],
                        "kernel_size": [5, 3, 3, 3],
                        "pool_size": [4, 2, 2, 2],
                        "lstm_unit": 128,
                        "in_channel": 60, 
                        "sequence_length": 10 * 100, 
                        "n_classes": 4,
                        "learning_rate": 1e-4}
        m_csablstm = CSABLSTM(hparams)
        m_csablstm.load_state_dict(torch.load("mymodel/weights/csablstm.pt", map_location=torch.device('cpu')))
        m_csablstm.eval()
        result = m_csablstm(self.feature)
        return result