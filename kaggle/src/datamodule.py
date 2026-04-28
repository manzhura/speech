from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset

__all__ = ["SpokenNumbersDataset"]


class SpokenNumbersDataset(Dataset):
    def __init__(
        self,
        dataframe,
        root_dir,
        target_sample_rate=16000,
        n_mels=80,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        target_frames=320,
        train=False,
        use_wave_augment=True,
        use_spec_augment=True,
        noise_prob=0.3,
        gain_prob=0.4,
        specaug_prob=0.5,
        noise_std=0.03,
        gain_db_range=(-6.0, 6.0),
        freq_mask_param=10,
        time_mask_param=25,
    ):
        self.root_dir = Path(root_dir)
        self.data = dataframe.reset_index(drop=True).copy()

        self.target_sample_rate = target_sample_rate
        self.target_frames = target_frames
        self.train = train

        self.use_wave_augment = use_wave_augment
        self.use_spec_augment = use_spec_augment

        self.noise_prob = noise_prob
        self.gain_prob = gain_prob
        self.specaug_prob = specaug_prob

        self.noise_std = noise_std
        self.gain_db_range = gain_db_range
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            center=True,
            power=2.0,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()
        self.resampler_cache = {}

        self.blank_id = 0
        self.char2idx = {str(i): i + 1 for i in range(10)}
        self.idx2char = {i + 1: str(i) for i in range(10)}

        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=self.freq_mask_param
        )
        self.time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=self.time_mask_param
        )

    def __len__(self):
        return len(self.data)

    def _to_mono(self, waveform):
        if waveform.size(0) == 1:
            return waveform
        return waveform.mean(dim=0, keepdim=True)

    def _resample_if_needed(self, waveform, sample_rate):
        if sample_rate == self.target_sample_rate:
            return waveform
        if sample_rate not in self.resampler_cache:
            self.resampler_cache[sample_rate] = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.target_sample_rate,
            )
        return self.resampler_cache[sample_rate](waveform)

    def _pad_or_trim_mel(self, mel):
        t = mel.size(1)
        if t < self.target_frames:
            mel = torch.nn.functional.pad(mel, (0, self.target_frames - t))
        elif t > self.target_frames:
            mel = mel[:, :self.target_frames]
        return mel

    def _encode_target(self, text):
        return torch.tensor([self.char2idx[ch] for ch in str(text)], dtype=torch.long)

    def _apply_wave_augment(self, waveform):
        if not self.train or not self.use_wave_augment:
            return waveform

        if torch.rand(1).item() < self.noise_prob:
            noise = torch.randn_like(waveform) * self.noise_std
            waveform = waveform + noise

        if torch.rand(1).item() < self.gain_prob:
            min_db, max_db = self.gain_db_range
            gain_db = torch.empty(1).uniform_(min_db, max_db).item()
            gain = 10 ** (gain_db / 20.0)
            waveform = waveform * gain

        waveform = waveform.clamp(-1.0, 1.0)
        return waveform

    def _apply_spec_augment(self, mel):
        if not self.train or not self.use_spec_augment:
            return mel

        if torch.rand(1).item() < self.specaug_prob:
            mel = self.freq_mask(mel)
            mel = self.time_mask(mel)

        return mel

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        waveform, sample_rate = torchaudio.load(self.root_dir / row["filename"])

        waveform = self._to_mono(waveform)
        waveform = self._resample_if_needed(waveform, sample_rate)
        waveform = self._apply_wave_augment(waveform)

        mel = self.mel_transform(waveform)
        mel = self.db_transform(mel + 1e-8)
        mel = mel.squeeze(0)
        mel = self._pad_or_trim_mel(mel)
        mel = self._apply_spec_augment(mel)

        target_text = str(row["transcription"])
        target_ids = self._encode_target(target_text)

        metadata = {
            "filename": row["filename"],
            "target_text": target_text,
            "spk_id": row["spk_id"],
        }

        return mel, target_ids, metadata