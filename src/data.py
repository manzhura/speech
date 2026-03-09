import os

import torch
import torchaudio
import lightning as L
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader

from melbanks import LogMelFilterBanks


class MelDataset(SPEECHCOMMANDS):
    def __init__(self, subset: str = None, root: str = "./dataset", n_mels: int = 80):
        super().__init__(root, subset=subset, download=True)

        self.labels = ["no", "yes"]
        self.label2id = {"no": 0, "yes": 1}

        self.mel = LogMelFilterBanks(n_mels=n_mels)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath, encoding="utf-8") as f:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in f]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = set(load_list("validation_list.txt") + load_list("testing_list.txt"))
            self._walker = [w for w in self._walker if w not in excludes]

        self._walker = [w for w in self._walker if "_background_noise_" not in w]
        self._walker = [w for w in self._walker if os.path.basename(os.path.dirname(w)) in ("yes", "no")]

    def _pad_trim_1s(self, wav, sr=16000):
        T = wav.size(-1)
        if T > sr:
            return wav[..., :sr]
        if T < sr:
            return torch.nn.functional.pad(wav, (0, sr - T))
        return wav

    def __getitem__(self, idx):
        waveform, sample_rate, label, *_ = super().__getitem__(idx)

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

        waveform = self._pad_trim_1s(waveform, sr=16000)

        mel = self.mel(waveform)                 # [1, n_mels, T]
        y = torch.tensor(self.label2id[label], dtype=torch.long)  # 0/1
        return mel, y


class MelDM(L.LightningDataModule):
    def __init__(self, data_dir: str = "./dataset", batch_size: int = 32, num_workers: int = 0, n_mels: int = 80):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_mels = n_mels

    def setup(self, stage: str = None):
        if stage in (None, "fit"):
            self.ds_train = MelDataset("training", root=self.data_dir, n_mels=self.n_mels)
            self.ds_val = MelDataset("validation", root=self.data_dir, n_mels=self.n_mels)

        if stage in (None, "test"):
            self.ds_test = MelDataset("testing", root=self.data_dir, n_mels=self.n_mels)

        if stage in (None, "predict"):
            self.ds_predict = MelDataset("testing", root=self.data_dir, n_mels=self.n_mels)

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(self.ds_predict, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)
