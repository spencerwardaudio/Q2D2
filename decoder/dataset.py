from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

import soundfile
# import librosa

# Add project root to path for shared utilities
_PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from dataloader_aug.audio_preprocessing import normalize_rms_snr
from dataloader_aug.dataset_paths import get_dataset_config

# Validate dataset paths on module load
_dataset_config = get_dataset_config()
assert _dataset_config.q2d2_train.exists(), \
    f"❌ Q2D2 training filelist missing: {_dataset_config.q2d2_train}"
assert _dataset_config.q2d2_val.exists(), \
    f"❌ Q2D2 validation filelist missing: {_dataset_config.q2d2_val}"

torch.set_num_threads(1)


@dataclass
class DataConfig:
    filelist_path: str
    sampling_rate: int
    num_samples: int
    batch_size: int
    num_workers: int


class VocosDataModule(LightningDataModule):
    def __init__(self, train_params: DataConfig, val_params: DataConfig):
        super().__init__()
        self.train_config = train_params
        self.val_config = val_params

    def _get_dataloder(self, cfg: DataConfig, train: bool):
        dataset = VocosDataset(cfg, train=train)
        dataloader = DataLoader(
            dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=train, pin_memory=True,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.train_config, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.val_config, train=False)


class VocosDataset(Dataset):
    def __init__(self, cfg: DataConfig, train: bool):
        with open(cfg.filelist_path) as f:
            self.filelist = f.read().splitlines()
        self.sampling_rate = cfg.sampling_rate
        self.num_samples = cfg.num_samples
        self.train = train

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, index: int) -> torch.Tensor:
        audio_path = self.filelist[index]
        # y, sr = torchaudio.load(audio_path)
        # print(audio_path,"111")
        y1, sr = soundfile.read(audio_path)
        # y1, sr = librosa.load(audio_path,sr=None)
        y = torch.tensor(y1).float().unsqueeze(0)
        # if y.size(0) > 1:
        #     # mix to mono
        #     y = y.mean(dim=0, keepdim=True)
        if y.ndim > 2:
            # mix to mono
            y = y.mean(dim=-1, keepdim=False)
        
        # COMMENTED OUT: RMS/SNR normalization produces excessive console spam
        # and Q2D2 trains successfully without it. Re-enable if needed for stability.
        # y = normalize_rms_snr(
        #     y,
        #     target_snr_db=40.0,
        #     train_mode=self.train,
        #     snr_variation_db=5.0,
        #     audio_path=audio_path,
        #     source_identifier="Q2D2/VocosDataset"
        # )
        
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)
        if y.size(-1) < self.num_samples:
            pad_length = self.num_samples - y.size(-1)
            padding_tensor = y.repeat(1, 1 + pad_length // y.size(-1))
            y = torch.cat((y, padding_tensor[:, :pad_length]), dim=1)
        elif self.train:
            start = np.random.randint(low=0, high=y.size(-1) - self.num_samples + 1)
            y = y[:, start : start + self.num_samples]
        else:
            # During validation, take always the first segment for determinism
            y = y[:, : self.num_samples]

        return y[0]
