import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pytorch_lightning as pl
from torch.utils.data import Dataset
from thop import profile
from melbanks import LogMelFilterBanks

class SpeechCommands(torchaudio.datasets.SPEECHCOMMANDS):
    def __init__(self, subset: str = None, data_path: str = "./data"):
        super().__init__(data_path, download=True)

        if subset == "validation":
            self._walker = self.load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = self.load_list("testing_list.txt")
        elif subset == "training":
            excludes = self.load_list("validation_list.txt") + self.load_list("testing_list.txt")
            walker = set(self._walker)
            self._walker = list(walker - set(excludes))

    def load_list(self, filename: str):
        filepath = os.path.join(self._path, filename)
        with open(filepath) as f:
            return [os.path.join(self._path, line.strip()) for line in f]

class BinarySpeechCommands(Dataset):
    def __init__(self, subset: str):
        self.dataset = SpeechCommands(subset=subset)
        self.target_classes = {"yes": 0, "no": 1}
        self.binary_indices = []

        for i in range(len(self.dataset)):
            _, _, label, _, _ = self.dataset[i]
            if label.lower() in self.target_classes:
                self.binary_indices.append(i)

    def __len__(self):
        return len(self.binary_indices)

    def __getitem__(self, idx):
        real_idx = self.binary_indices[idx]
        waveform, _, label, _, _ = self.dataset[real_idx]
        waveform = waveform.squeeze(0)
        label = self.target_classes[label]
        return waveform, label

class CNN(nn.Module):
    def __init__(self, n_classes=2):
        super(CNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=80, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

class SpeechCommandBinaryClassifier(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super(SpeechCommandBinaryClassifier, self).__init__()
        self.save_hyperparameters()
        self.feature_extractor = LogMelFilterBanks()
        self.cnn = CNN(n_classes=2)
        self.criterion = nn.CrossEntropyLoss()
        self.train_epoch_start_time = None
        self.lr = lr

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.cnn(features)
        return logits

    def training_step(self, batch, batch_idx):
        waveform, label = batch
        logits = self.forward(waveform)
        loss = self.criterion(logits, label)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def training_epoch_start(self):
        self.train_epoch_start_time = time.time()

    def on_training_epoch_end(self, outputs):
        epoch_time = time.time() - self.train_epoch_start_time
        self.log("epoch_time", epoch_time)

    def validation_step(self, batch, batch_idx):
        waveform, label = batch
        logits = self.forward(waveform)
        loss = self.criterion(logits, label)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == label).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return acc

    def test_step(self, batch, batch_idx):
        waveform, label = batch
        logits = self.forward(waveform)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == label).float().mean()
        self.log("test_acc", acc)
        return acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def compute_flops(self, input_size=(1, 16000)):
        dummy_input = torch.randn(*input_size)
        features = self.feature_extractor(dummy_input)
        macs, params = profile(self.cnn, inputs=(features,))
        flops = macs * 2

        print("CNN FLOPs: %s   MACs: %s   Params: %s \n" %(flops, macs, params))

        return flops, macs, params
        

def collate_fn(batch):
    waveforms, labels = zip(*batch)
    lengths = [w.shape[0] for w in waveforms]
    max_length = max(lengths)
    padded_waveforms = [F.pad(w, (0, max_length - w.shape[0])) for w in waveforms]
    padded_waveforms = torch.stack(padded_waveforms)
    labels = torch.tensor(labels)
    return padded_waveforms, labels
