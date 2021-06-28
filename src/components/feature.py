import torch
import torchaudio
from torchaudio.compliance import kaldi

class FilterBank(object):
    def __init__(
            self,
            frame_length: float = 50.,
            frame_shift: float = 10.,
            num_mel_bins: int = 40,
            preemphasis_coefficient: float = 0.97,
            window_type: str = "hanning",
            sample_frequency: float = 16000.,
            dither: float = 0.,
            low_freq: float = None,
            high_freq: float = None,
            vtln_max: float = 1.0,
            vtln_min: float = 1.0):
        if high_freq is None:
            high_freq = sample_frequency // 2
        if low_freq is None:
            low_freq = 0.
        self.vtln_range = torch.arange(vtln_min, vtln_max, 0.01)
        self.spectrogram_params = {
            "frame_length": frame_length,
            "frame_shift": frame_shift,
            "num_mel_bins": num_mel_bins,
            "preemphasis_coefficient": preemphasis_coefficient,
            "window_type": window_type,
            "sample_frequency": sample_frequency,
            "dither": dither,
            "high_freq": high_freq,
            "low_freq": low_freq
        }

    def _sample_vtln_factor(self):
        if len(self.vtln_range) == 0:
            return 1.0
        else:
            idx = torch.randint(0, len(self.vtln_range), (1,))[0]
            return self.vtln_range[idx]

    def __call__(self, sample):
        audio = sample
        alpha = self._sample_vtln_factor()
        fbank = kaldi.fbank(audio, vtln_warp=alpha, **self.spectrogram_params)
        fbank = torch.transpose(fbank, 0, 1)
        return fbank

class NormalizeSample(object):
    def __init__(self,
                 center_feats: bool = True,
                 scale_feats: bool = False):
        self.center_feats = center_feats
        self.scale_feats = scale_feats

    def __call__(self, sample):
        feature = sample
        if self.center_feats:
            # feature = feature - feature.mean(dim=0)
            feature = feature - torch.unsqueeze(feature.mean(dim=-1), dim=-1)
        if self.scale_feats:
            # feature = feature / torch.sqrt(feature.var(dim=0) + 1e-8)
            feature = feature / torch.sqrt(torch.unsqueeze(feature.var(dim=-1), dim=-1) + 1e-8)
        return feature

