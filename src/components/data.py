from typing import Callable, Union, List, Dict

import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.compliance import kaldi
from tqdm import tqdm
from torchvision.transforms.transforms import Compose

from vistec_ser.data.features.padding import pad_dup
from .feature import NormalizeSample, FilterBank

class PrepareData(Dataset):
    """Speech Emotion Recognition Dataset
    Storing all dataset in RAM
    """

    def __init__(
            self,
            max_len: int,
            frame_len: int = 50,
            frame_shift: int = 10,
            num_mel: int = 60,
            file_path=None,
            audio=None,
            len_thresh: float = 0.5,
            pad_fn: Callable = pad_dup,
            sampling_rate: int = 16000,
            center_feats: bool = True,
            scale_feats: bool = True,
            emotions=None,
            transform=None,
            ):

        if emotions is None:
            self.emotions = ["neutral", "anger", "happiness", "sadness"]
        else:
            self.emotions = emotions
        self.n_classes = len(self.emotions)
        self.file_path = file_path
        self.audio = audio
        assert isinstance(max_len, int)
        assert isinstance(sampling_rate, int)
        self.sampling_rate = sampling_rate
        self.max_len = max_len * 100
        self.len_thresh = len_thresh
        self.pad_fn = pad_fn
        self.transform = transform
        self.normalize = NormalizeSample(center_feats, scale_feats)
        self.transform = Compose([FilterBank(
            frame_length=frame_len,
            frame_shift=frame_shift,
            num_mel_bins=num_mel)])

    def _chop_sample(self, sample: torch.Tensor) -> List[torch.Tensor]:
        x = sample
        _, time_dim = x.shape
        x_chopped = list()
        for i in range(time_dim):
            if i % self.max_len == 0 and i != 0:  # if reach self.max_len
                xi = x[:, i - self.max_len:i]
                assert xi.shape[-1] == self.max_len, xi.shape
                x_chopped.append(self.normalize(xi))
        if time_dim < self.max_len:  # if file length not reach self.max_len
            if self.pad_fn:
                xi = self.pad_fn(x, max_len=self.max_len)
                assert xi.shape[-1] == self.max_len
            else:
                xi = x
            x_chopped.append(self.normalize(xi))
        else:  # if file is longer than n_frame, pad remainder
            remainder = x[:, x.shape[-1] - x.shape[0] % self.max_len:]
            if not remainder.shape[-1] <= self.len_thresh:
                if self.pad_fn:
                    xi = self.pad_fn(remainder, max_len=self.max_len)
                else:
                    xi = x
                x_chopped.append(self.normalize(xi))
        return x_chopped

    def _load_feature(self, audio_path: str) -> torch.Tensor:
        audio, sample_rate = torchaudio.load(audio_path)
        # initial preprocess
        # convert to mono, resample, truncate
        audio = torch.unsqueeze(audio.mean(dim=0), dim=0)  # convert to mono
        if sample_rate != self.sampling_rate:
            audio = kaldi.resample_waveform(audio, orig_freq=sample_rate, new_freq=self.sampling_rate)
        return audio

    def _load_audio(self, read_file=True):
        if read_file:
            fbank = self.transform(self._load_feature(self.file_path))
        else:
            fbank = self.transform(self.audio)
        print(fbank.shape)
        samples = self._chop_sample(fbank)
        stack_samples = torch.stack(samples)
        return stack_samples