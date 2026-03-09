from typing import Optional

from torchaudio.datasets import SPEECHCOMMANDS
import torch
import os
import librosa
import torchaudio
from torch import nn
from torchaudio import functional as F
import matplotlib.pyplot as plt


melspec = torchaudio.transforms.MelSpectrogram(
    hop_length=160,
    n_mels=80
)


class LogMelFilterBanks(nn.Module):
    def __init__(
            self,
            n_fft: int = 400,
            samplerate: int = 16000,
            hop_length: int = 160,
            n_mels: int = 80,
            pad_mode: str = 'reflect',
            power: float = 2.0,
            normalize_stft: bool = False,
            onesided: bool = True,
            center: bool = True,
            return_complex: bool = True,
            f_min_hz: float = 0.0,
            f_max_hz: Optional[float] = None,
            norm_mel: Optional[str] = None,
            mel_scale: str = 'htk'
    ):
        super(LogMelFilterBanks, self).__init__()
        # general params and params defined by the exercise
        self.n_fft = n_fft
        self.samplerate = samplerate
        self.window_length = n_fft
        self.window = torch.hann_window(self.window_length)
        # Do correct initialization of stft params below:
        # hop_length, n_mels, center, return_complex, onesided, normalize_stft, pad_mode, power
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.center = center
        self.return_complex = return_complex
        self.onesided = onesided
        self.normalize_stft = normalize_stft
        self.pad_mode = pad_mode
        self.power = power

        # Do correct initialization of mel fbanks params below:
        # f_min_hz, f_max_hz, norm_mel, mel_scale
        self.f_min_hz = f_min_hz
        self.f_max_hz = f_max_hz
        self.norm_mel = norm_mel
        self.mel_scale = mel_scale
        # finish parameters initialization
        self.mel_fbanks = self._init_melscale_fbanks()

    def _init_melscale_fbanks(self):
        # To access attributes, use self.<parameter_name>
        return F.melscale_fbanks(
            # Turns a normal STFT into a mel frequency STFT with triangular filter banks
            # make a full and correct function call
            n_freqs=int(self.n_fft // 2 + 1),
            f_min=self.f_min_hz,
            f_max=self.f_max_hz if self.f_max_hz is not None else self.samplerate / 2,
            n_mels=self.n_mels,
            sample_rate=self.samplerate,
            norm=self.norm_mel,
            mel_scale=self.mel_scale
        )

    def spectrogram(self, x):
        # x - is an input signal
        return torch.stft(x,
                          n_fft=self.n_fft,
                          win_length=self.window_length,
                          hop_length=self.hop_length,
                          window=self.window,
                          center=self.center,
                          pad_mode=self.pad_mode,
                          normalized=self.normalize_stft,
                          onesided=self.onesided,
                          return_complex=self.return_complex,
                          )

    def forward(self, x):
        """
        Args:
            x (Torch.Tensor): Tensor of audio of dimension (batch, time), audiosignal
        Returns:
            Torch.Tensor: Tensor of log mel filterbanks of dimension (batch, n_mels, n_frames),
                where n_frames is a function of the window_length, hop_length and length of audio
        """
        spec = self.spectrogram(x)

        spec = spec.abs().pow(self.power)

        mel = torch.einsum("bft,fm->bmt", spec, self.mel_fbanks)
        logmel = torch.log(mel + 1e-6)
        return logmel

    def plot_fbank(self, title=None):
        fig, axs = plt.subplots(1, 1)
        axs.set_title("Filter bank")
        axs.imshow(self.mel_fbanks, aspect="auto")
        axs.set_ylabel("frequency bin")
        axs.set_xlabel("mel bin")

    def plot_spectrogram(self, specgram, title="Спектрограмма сигнала"):
        fig, ax = plt.subplots(1, 1)
        ax.set_title(title)
        ax.set_ylabel("freq_bin")

        ax.imshow(
            librosa.power_to_db(specgram),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )

        fig.tight_layout()
        plt.show()