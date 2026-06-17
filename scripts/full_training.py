#!/usr/bin/env python3

"""
full_training.py

This file provides everything you need to train a model.
Supply the file via command line arguments with a json param file in the following format:

{
  "RUN_NAME": "TEST_RUN",            # name of folder to put the logs in
  "METADATA": <path to metadata>,    # metadata
  "SPEC_TYPE": "logmel-balanced",    # stft/logmel-(time, balanced, freq), mfcc1-(small, medium, large)
  "USE_AUG": true,                   # Whether to use augmentation pipeline
  "TRAINING": {
    "SEED": 9,                       # random seed
    "EPOCHS": 2,                     # number of epochs to train
    "BATCH_SIZE": 32,                # batch size
    "LR": 0.001,                     # learning rate
    "CONTINUE_RUN": false,           # whether to train from the previous run of the same 'RUN_NAME'
    "TIME_LIMIT": 8                  # time limit to training (hours)
  },
  "MODEL": {
    "CLASSES": 16,                   # number of classes
    "TYPE": "resnet",                # model to train with, currently supported models: 'resnet'
    "VARIANT": 50                    # use this to get specific version of model
  }
}
"""

###############################################################################
# Global Imports
###############################################################################
import os                                                                                       # Path operations
import json                                                                                     # Json Processing (load/dump)
from time import strftime                                                                       # Time processing
from typing import Optional                                                                     # Optional function args
from warnings import filterwarnings                                                             # Remove common warnings

###############################################################################
# 3PP Imports
###############################################################################
import click                                                                                    # Command line arguments
import torch                                                                                    # Tensor and various functions
import numpy as np                                                                              # Numpy mean
from tqdm import tqdm                                                                           # Progress bar
from pandas import read_csv                                                                     # Open csv metadata
from timm import create_model                                                                   # Create and init DL models
from torch.nn import CrossEntropyLoss                                                           # Loss function
from torchaudio import transforms, load                                                         # Spectrograms and load audio
from lightning.pytorch.callbacks import Timer                                                   # Limit training by time
from lightning.pytorch.loggers import CSVLogger                                                 # Model logging
from torch.utils.data import DataLoader, Dataset                                                # Training related stuff
from lightning.pytorch import Callback, LightningModule                                         # Training helpers
from audiomentations import Compose, Gain, Shift, AddGaussianNoise                              # Audio transformations
from sklearn.metrics import confusion_matrix, classification_report                             # Better metrics
from lightning.pytorch import Trainer, LightningDataModule, seed_everything                     # Training helpers
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, Precision, Recall                  # Torch metric calculations


# ======================================================================================================
# =============  Spectrogram Processing  ===============================================================
# ======================================================================================================
def extract_features(wave: torch.Tensor, data_params: dict, aug=None):
    spec_kind = data_params["SPEC_TYPE"]
    parts = spec_kind.split("-")
    spec_type = parts[0]
    spec_variant = parts[1]

    if spec_type == "stft":
        spec = make_stft(wave, spec_variant, aug)
    elif spec_type == "logmel":
        spec = make_logmel(wave, spec_variant, aug)
    elif spec_type == "mfcc":
        spec = make_mfcc(wave, spec_variant, aug)
    else:
        raise ValueError(spec_type)

    return spec.unsqueeze(dim=0)


def clamp_spec(spec: torch.Tensor, min_out_val=0.0, max_out_val=1.0) -> torch.Tensor:
    """clamp spectrogram between min and max value"""
    # Input scaling
    min_in_val = torch.min(spec).item()
    max_in_val = torch.max(spec).item()
    in_span = max_in_val - min_in_val
    # Output scaling
    out_span = max_out_val - min_out_val
    assert in_span != 0, "SPECTROGRAM CONTAINS NO DATA"
    scale_factor = out_span / in_span
    return (spec - min_in_val) * scale_factor


def make_stft(wave: torch.Tensor, variant: str, aug: Optional[Compose] = None):
    """Create STFT spectrogram"""
    spec_params = {
        "freq": {
            "SR": 44100,
            "DURATION": 1,
            "NORM_AUD": True,
            "NORM_SPEC": False,
            "N_FFT": 2048,
            "HOP_LEN": 1024,
            "SCALING": "POWER",
        },
        "linear": {
            "SR": 44100,
            "DURATION": 1,
            "NORM_AUD": True,
            "NORM_SPEC": False,
            "N_FFT": 1024,
            "HOP_LEN": 512,
            "SCALING": "LINEAR",
        },
        "balanced": {
            "SR": 44100,
            "DURATION": 1,
            "NORM_AUD": True,
            "NORM_SPEC": False,
            "N_FFT": 1024,
            "HOP_LEN": 512,
            "SCALING": "POWER",
        },
        "hannideal": {
            "SR": 44100,
            "DURATION": 1,
            "NORM_AUD": True,
            "NORM_SPEC": False,
            "N_FFT": 1024,
            "HOP_LEN": 256,
            "SCALING": "POWER",
        },
        "time": {
            "SR": 44100,
            "DURATION": 1,
            "NORM_AUD": True,
            "NORM_SPEC": False,
            "N_FFT": 512,
            "HOP_LEN": 256,
            "SCALING": "POWER",
        },
    }[variant]

    num_samples = int(spec_params["SR"] * spec_params["DURATION"])
    padding_size = max(num_samples - wave.size(1), 0)
    if padding_size > 0:
        wave = torch.cat([wave, torch.zeros(1, padding_size)], dim=1)
    wave = wave[:, :num_samples]
    clean_wave = wave.clone()

    if aug is not None:
        am_wave = wave.numpy()[0]
        am_wave = aug(samples=am_wave, sample_rate=spec_params["SR"])
        wave = torch.from_numpy(am_wave).unsqueeze(0)

    stft_transform = transforms.Spectrogram(
        n_fft=spec_params["N_FFT"],
        hop_length=spec_params["HOP_LEN"],
        normalized=spec_params["NORM_SPEC"],
        power=2.0,
    )

    try:
        stft_output = stft_transform(wave).squeeze(0)
        if spec_params["SCALING"] == "MAG":
            stft_output = transforms.AmplitudeToDB(stype="magnitude", top_db=80)(stft_output)
        if spec_params["SCALING"] == "POWER":
            stft_output = transforms.AmplitudeToDB(stype="power", top_db=80)(stft_output)
        stft_output = clamp_spec(stft_output)
        return stft_output
    except ZeroDivisionError:
        stft_output = stft_transform(clean_wave).squeeze(0)
        stft_output = transforms.AmplitudeToDB(top_db=80)(stft_output)
        stft_output = clamp_spec(stft_output)
        return stft_output


def make_logmel(wave: torch.Tensor, variant: str, aug: Optional[Compose] = None):
    """create log-mel spectrogram"""
    spec_params = {
        "freq": {
            "SR": 44100,
            "DURATION": 1,
            "NORM_AUD": True,
            "NORM_MEL": False,
            "N_FFT": 2048,
            "N_MELS": 256,
            "HOP_LEN": 1024,
            "SCALING": "POWER",
        },
        "balanced": {
            "SR": 44100,
            "DURATION": 1,
            "NORM_AUD": True,
            "NORM_MEL": False,
            "N_FFT": 1024,
            "N_MELS": 128,
            "HOP_LEN": 512,
            "SCALING": "POWER",
        },
        "hannideal": {
            "SR": 44100,
            "DURATION": 1,
            "NORM_AUD": True,
            "NORM_MEL": False,
            "N_FFT": 1024,
            "N_MELS": 128,
            "HOP_LEN": 256,
            "SCALING": "POWER",
        },
        "time": {
            "SR": 44100,
            "DURATION": 1,
            "NORM_AUD": True,
            "NORM_MEL": False,
            "N_FFT": 512,
            "N_MELS": 64,
            "HOP_LEN": 256,
            "SCALING": "POWER",
        },
    }[variant]

    num_samples = int(spec_params["SR"] * spec_params["DURATION"])
    padding_size = max(num_samples - wave.size(1), 0)
    if padding_size > 0:
        wave = torch.cat([wave, torch.zeros(1, padding_size)], dim=1)
    wave = wave[:, :num_samples]
    clean_wave = wave.clone()

    mel_spec = transforms.MelSpectrogram(
        sample_rate=spec_params["SR"],
        n_fft=spec_params["N_FFT"],
        n_mels=spec_params["N_MELS"],
        hop_length=spec_params["HOP_LEN"],
        normalized=spec_params["NORM_MEL"],
    )

    if aug is not None:
        am_wave = wave.numpy()[0]
        am_wave = aug(samples=am_wave, sample_rate=spec_params["SR"])
        wave = torch.from_numpy(am_wave).unsqueeze(0)

    try:
        mel_output = mel_spec(wave).squeeze(0)
        scaling_type = spec_params["SCALING"]

        if scaling_type == "MAG":
            log_mel_output = transforms.AmplitudeToDB(stype="magnitude", top_db=80)(mel_output)
        elif scaling_type == "POWER":
            log_mel_output = transforms.AmplitudeToDB(stype="power", top_db=80)(mel_output)
        else:
            raise ValueError(scaling_type)

        log_mel_output = clamp_spec(log_mel_output)
        return log_mel_output
    except ZeroDivisionError:
        mel_output = mel_spec(clean_wave).squeeze(0)
        log_mel_output = transforms.AmplitudeToDB(top_db=80)(mel_output)
        log_mel_output = clamp_spec(log_mel_output)
        return log_mel_output


def make_mfcc(wave: torch.Tensor, variant: str, aug: Optional[Compose] = None):
    """create MFCC spectrogram"""
    spec_params = {
        "small": {
            "SR": 44100,
            "DURATION": 1,
            "NORM_AUD": True,
            "NORM_MEL": False,
            "N_FFT": 1024,
            "N_MELS": 128,
            "HOP_LEN": 512,
            "N_MFCC": 20,
        },
        "medium": {
            "SR": 44100,
            "DURATION": 1,
            "NORM_AUD": True,
            "NORM_MEL": False,
            "N_FFT": 1024,
            "N_MELS": 128,
            "HOP_LEN": 512,
            "N_MFCC": 30,
        },
        "large": {
            "SR": 44100,
            "DURATION": 1,
            "NORM_AUD": True,
            "NORM_MEL": False,
            "N_FFT": 1024,
            "N_MELS": 128,
            "HOP_LEN": 512,
            "N_MFCC": 40,
        },
    }[variant]

    num_samples = int(spec_params["SR"] * spec_params["DURATION"])
    padding_size = max(num_samples - wave.size(1), 0)
    if padding_size > 0:
        wave = torch.cat([wave, torch.zeros(1, padding_size)], dim=1)
    wave = wave[:, :num_samples]
    clean_wave = wave.clone()

    mfcc_transform = transforms.MFCC(
        sample_rate=spec_params["SR"],
        n_mfcc=spec_params["N_MFCC"],
        melkwargs={
            "n_fft": spec_params["N_FFT"],
            "n_mels": spec_params["N_MELS"],
            "hop_length": spec_params["HOP_LEN"],
            "normalized": spec_params["NORM_MEL"],
        },
    )

    if aug is not None:
        am_wave = wave.numpy()[0]
        am_wave = aug(samples=am_wave, sample_rate=spec_params["SR"])
        wave = torch.from_numpy(am_wave).unsqueeze(0)

    try:
        mfcc_output = mfcc_transform(wave).squeeze(0)
        mfcc_output = clamp_spec(mfcc_output)
        return mfcc_output
    except ZeroDivisionError:
        mfcc_output = mfcc_transform(clean_wave).squeeze(0)
        mfcc_output = clamp_spec(mfcc_output)
        return mfcc_output


# ======================================================================================================
# =============  Logging and Checkpointing  ============================================================
# ======================================================================================================
def print_confmatrix(cm, labels=None):
    """Print confusion matrix in human readable format"""
    labels = [f'Cl {i}' for i in range(len(cm))] if labels is None else labels

    # Print column headers
    print("\n", flush=True)
    print(' ' * 6, end='')
    print('  '.join(f'{label:>8}' for label in labels))

    # Print rows with labels
    for i, (row, label) in enumerate(zip(cm, labels)):
        print(f'{label:<6}', end='')
        # Format row with diagonal in []
        formatted_row = []
        for j, val in enumerate(row):
            if i == j:
                formatted_row.append(f'[{val:>6}]')
            else:
                formatted_row.append(f' {val:>6} ')
        print('  '.join(formatted_row))
    print("\n", flush=True)


def save_metrics(model, my_path):
    """Save metrics in better format upon finishing training"""
    results = {
        "loss": model.epoch_losses,
        "accs": model.epoch_accs,
        "metric_f1": model.metrics_all["f1"],
        "metric_precision": model.metrics_all["precision"],
        "metric_recall": model.metrics_all["recall"],
        "confusion": model.conf_matrices,
    }

    # === Save json metrics ===
    filename = os.path.join(my_path, "full_metrics.json")
    with open(filename, "w") as file:
        json.dump(results, file)

    # === Return all metrics for introspection ===
    return results


def save_export_model(model, params, my_path):
    """Export model upon finishing training"""
    # Save metrics
    my_metrics = save_metrics(model, my_path)

    # Save pt file
    filename = "model_final.pt"
    torch.save(model.model.state_dict(), os.path.join(my_path, filename))

    # Export to onnx
    model.eval()
    example = torch.rand(params["INPUT_SIZE"]).unsqueeze(dim=0)
    output_path = os.path.join(my_path, "model_final.onnx")
    dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    torch.onnx.export(
        model.model,                  # Model being exported
        (example,),                   # Sample input the model needs
        output_path,                  # Output file path
        export_params=True,           # Store the trained weights
        opset_version=14,             # ONNX version to use
        do_constant_folding=True,     # Optimize constant-folding
        input_names=['input'],        # Names for inputs
        output_names=['output'],      # Names for outputs
        dynamic_axes=dynamic_axes,    # Dynamic axes specification
        verbose=False
    )

    # Reopen file to get filesize and convert to MB
    file_size = os.path.getsize(output_path) / 1024 / 1024
    return file_size, my_metrics


def load_newest_checkpoint(model, base_dir, model_name, version=None):
    """Load previous run of the model"""
    # === Find checkpoint ===
    my_path = os.path.join(base_dir, model_name)
    if not os.path.exists(my_path):
        print("First run of current model type")
        return None
    vers = version
    ckpt_path = ""
    if vers is None:
        all_versions = [f for f in os.listdir(my_path) if os.path.isdir(os.path.join(my_path, f))]
        # Checkpoints are timestamped, sorting gets newest one
        vers = sorted(all_versions)[-1]
    ckpt_path = os.path.join(my_path, vers, "model_best.ckpt")
    if not os.path.exists(ckpt_path):
        print("Error no checkpoints found in ", vers)
        return None

    # === Load checkpoint ===
    ckpt_data = torch.load(ckpt_path)
    print("Continuing run from ", vers[8:])
    print(f"Model got to epoch {ckpt_data['epoch']}")
    print(f"Model got metric {ckpt_data['best_metric']}")
    model.model.load_state_dict(ckpt_data['model_state_dict'])


class CustomModelCheckpoint(Callback):
    def __init__(self, save_dir, params, monitor='val_loss', mode='min'):
        super().__init__()
        self.save_dir = save_dir
        self.monitor = monitor
        self.params = params
        self.mode = mode
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def on_validation_end(self, trainer, pl_module):
        # Get the current validation metric
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            print("ERROR")
            return
        if (self.mode == 'min' and current < self.best) or (self.mode == 'max' and current > self.best):
            self.save_best = True
            self.best = current
            self._save_checkpoint(trainer, pl_module)
        else:
            self.save_best = False
            self._save_checkpoint(trainer, pl_module)

    def on_load_checkpoint(self, trainer, pl_module, checkpoint: dict):
        # Optionally handle loading of checkpoint state
        self.best = checkpoint.get('best_metric', float('inf') if self.mode == 'min' else float('-inf'))

    def _save_checkpoint(self, trainer, pl_module):
        # Define checkpoint file paths
        filename = "model_best" if self.save_best else "model_curr"
        checkpoint_path = os.path.join(self.save_dir, f"{filename}.ckpt")
        checkpoint_onnx_path = os.path.join(self.save_dir, f"{filename}.onnx")

        # Save the model state
        torch.save({
            'model_state_dict': pl_module.model.state_dict(),
            'optimizer_state_dict': trainer.optimizers[0].state_dict(),
            'epoch': trainer.current_epoch,
            'best_metric': self.best
        }, checkpoint_path)
        print(f"Saved new best model to {checkpoint_path}")

        # Save the best model as onnx
        pl_module.model.eval()
        example = torch.rand(self.params["INPUT_SIZE"]).unsqueeze(dim=0).to(self.device)

        # Export to onnx
        dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        torch.onnx.export(
            pl_module.model,              # Model being exported
            (example,),                      # Sample input the model needs
            checkpoint_onnx_path,         # Output file path
            export_params=True,           # Store the trained weights
            opset_version=14,             # ONNX version to use
            do_constant_folding=True,     # Optimize constant-folding
            input_names=['input'],        # Names for inputs
            output_names=['output'],      # Names for outputs
            dynamic_axes=dynamic_axes,    # Dynamic axes specification
            verbose=False
        )


# ======================================================================================================
# =============  Data Processing  ======================================================================
# ======================================================================================================
class MemDataset(Dataset):
    def __init__(self, meta, params, transform=None, target_transform=None, augmentations=None):
        self.meta = meta
        self.tx = transform
        self.aug = augmentations
        self.data_params = params
        self.target_tx = target_transform
        # Load all data and store it in RAM
        self.data = {}
        self.data_ref = []
        for i in tqdm(range(meta.shape[0])):
            row_data = meta.iloc[i]
            datum_id = row_data["datum_id"]
            if datum_id not in self.data:
                fname = row_data["file_path"]
                self.data[datum_id] = self._load_file(fname)
            self.data_ref.append(datum_id)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        y = self.meta["class_id"].iloc[idx]
        # Transformations
        try:
            my_data = self.data[self.data_ref[idx]]
            X = self.tx(my_data, self.data_params, self.aug) if self.tx else my_data
            if self.target_tx is not None:
                y = self.target_tx(y)
            return [X, y]
        except:
            print(self.meta.iloc[idx])
            raise Exception("Failure")

    def _load_file(self, file_path) -> torch.Tensor:
        wav, _ = load(file_path)
        return wav


class SimpleDataModule(LightningDataModule):
    def __init__(self, metadata, classname, data_params, meta_bg=None, batch_size=32, workers=0,
                 augmentations=None, transform=None, target_transform=None):
        super().__init__()
        self.batch_size = batch_size
        self.workers = workers
        self.meta = metadata
        self.meta_bg = meta_bg
        self.dataset_class = classname
        self.params = data_params

        # Augmentations
        self.aug = augmentations

        # Transformations
        self.train_tx = transform
        self.test_tx = transform
        self.val_tx = transform
        self.predict_tx = transform
        self.train_target_tx = target_transform
        self.test_target_tx = target_transform
        self.val_target_tx = target_transform
        self.predict_target_tx = target_transform

    def setup(self, stage: str):
        train_meta = self.meta[self.meta.split == "train"]
        val_meta = self.meta[self.meta.split == "val"]
        test_meta = self.meta[self.meta.split == "test"]
        predict_meta = test_meta

        # Save into Datasets
        if stage == "fit":
            self.train_dataset = self.dataset_class(
                train_meta, self.params, augmentations=self.aug,
                transform=self.train_tx, target_transform=self.train_target_tx)
            self.val_dataset = self.dataset_class(
                val_meta, self.params, transform=self.val_tx, target_transform=self.val_target_tx)
        if stage == "test":
            self.test_dataset = self.dataset_class(
                test_meta, self.params, transform=self.test_tx, target_transform=self.test_target_tx)
        if stage == "predict":
            self.predict_dataset = self.dataset_class(
                predict_meta, self.params, transform=self.predict_tx, target_transform=self.predict_target_tx)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.workers)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.workers)

    def get_dataloader(self, name):
        if name == "train":
            self.setup("fit")
            return self.train_dataloader()
        elif name == "val":
            self.setup("fit")
            return self.val_dataloader()
        elif name == "test":
            self.setup("test")
            return self.test_dataloader()
        elif name == "predict":
            self.setup("predict")
            return self.predict_dataloader()
        else:
            raise ValueError(name)


# ======================================================================================================
# =============  Models  ===============================================================================
# ======================================================================================================
class MyResNet(LightningModule):
    def __init__(self, num_channels, num_classes, depth=50, pretrained=False):
        super(MyResNet, self).__init__()
        self.name = f"ResNet_{depth}"
        self.num_channels = num_channels
        self.num_classes = num_classes

        # Create ResNet model using timm
        model_name = f"resnet{depth}"
        self.model = create_model(
            model_name,
            pretrained=pretrained,
            num_classes=self.num_classes,
            in_chans=self.num_channels,
        )

    def forward(self, x):
        x = self.model(x)
        x = torch.sigmoid(x)
        return x


class LightningModel(LightningModule):
    def __init__(self, model, hparams):
        super(LightningModel, self).__init__()
        self.model = model
        self.name = self.model.name
        self.lr = hparams["LR"]
        self.lr_min = 1e-6
        self.num_classes = self.model.num_classes

        self.accuracy = Accuracy(num_classes=self.num_classes, task="multiclass")
        self.confmat = ConfusionMatrix(num_classes=self.num_classes, task="multiclass")

        # Metrics
        self.metric_f1 = F1Score(num_classes=self.num_classes, task="multiclass", average="macro")
        self.metric_prec = Precision(num_classes=self.num_classes, task="multiclass", average="macro")
        self.metric_recall = Recall(num_classes=self.num_classes, task="multiclass", average="macro")

        # Loss Function
        self.crit = CrossEntropyLoss()

        # Losses and Accuracy for plotting
        self.losses = {"train": [], "test": [], "val": [], "pred": []}
        self.accs = {"train": [], "test": [], "val": [], "pred": []}

        self.metrics_all = {
            "f1": {"train": [], "test": [], "val": [], "pred": []},
            "precision": {"train": [], "test": [], "val": [], "pred": []},
            "recall": {"train": [], "test": [], "val": [], "pred": []},
        }
        self.epoch_losses = {"train": [], "test": [], "val": [], "pred": []}
        self.epoch_accs = {"train": [], "test": [], "val": [], "pred": []}

        self.conf_matrices = {"train": [], "test": [], "val": [], "pred": []}

        self.y_true, self.y_pred = [], []

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return opt

    def forward(self, x):
        y = self.model(x)
        return y

    def basic_step(self, batch, batch_idx, mode):
        x, y = batch
        y_hat = self.forward(x)             # Logits
        y_val = torch.argmax(y_hat, dim=1)  # Class
        loss = self.crit(y_hat, y)  # Calculate loss
        acc = self.accuracy(y_hat, y)  # Calculate accuracy using logits
        # === Logging ===
        self.losses[mode].append(loss.item())
        self.accs[mode].append(acc.item())
        # Store predictions and targets for confusion matrix and classification report
        self.y_true.extend(y.cpu().numpy())
        self.y_pred.extend(y_val.cpu().numpy())
        return loss

    def basic_reset(self, mode):
        self.losses[mode] = []
        self.accs[mode] = []
        self.y_true.clear()
        self.y_pred.clear()

    def basic_metrics(self, mode):
        # === Reports ===
        if mode != "train":
            conf_matrix = confusion_matrix(self.y_true, self.y_pred)
            class_report = classification_report(self.y_true, self.y_pred, zero_division=0.0)
            # Log or print confusion matrix and classification report
            print_confmatrix(conf_matrix)
            self.conf_matrices[mode].append(conf_matrix.tolist())
            print(f"Classification Report - {mode}:")
            print(class_report)
            # Metrics
            ten_y_pred = torch.tensor(self.y_pred).to(self.device)
            ten_y_true = torch.tensor(self.y_true).to(self.device)
            score_f1 = self.metric_f1(ten_y_pred, ten_y_true).item()
            score_pr = self.metric_prec(ten_y_pred, ten_y_true).item()
            score_re = self.metric_recall(ten_y_pred, ten_y_true).item()
            self.metrics_all["f1"][mode].append(score_f1)
            self.metrics_all["precision"][mode].append(score_pr)
            self.metrics_all["recall"][mode].append(score_re)

        # === Calculate Loss and Accuracy for Epoch ===
        tmp_epoch_loss = np.mean(self.losses[mode])
        tmp_epoch_acc = np.mean(self.accs[mode])
        self.epoch_losses[mode].append(tmp_epoch_loss)
        self.epoch_accs[mode].append(tmp_epoch_acc)

        # === Logging ===
        self.log(f"{mode}_loss", tmp_epoch_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{mode}_accuracy", tmp_epoch_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'], prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.basic_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.basic_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.basic_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx):
        return self.basic_step(batch, batch_idx, "pred")

    def on_train_epoch_start(self):
        self.basic_reset("train")

    def on_train_epoch_end(self):
        self.basic_metrics("train")

    def on_validation_epoch_start(self):
        self.basic_reset("val")

    def on_validation_epoch_end(self):
        self.basic_metrics("val")

    def on_test_epoch_start(self):
        self.basic_reset("test")

    def on_test_epoch_end(self):
        self.basic_metrics("test")

    def on_predict_epoch_start(self):
        self.basic_reset("pred")

    def on_predict_epoch_end(self):
        self.basic_metrics("pred")


# ======================================================================================================
# =============  Main Function  ========================================================================
# ======================================================================================================
@click.command()
@click.argument("param_filename")
def main(param_filename):
    for s in range(5):
        full_training(param_filename, s)


def full_training(param_filename, seed):
    # ====== Setup ======

    # Load param file
    print("Running model on", param_filename)
    with open(param_filename, "r", encoding="utf-8") as file:
        all_params = json.load(file)

    # Setup
    torch.set_default_dtype(torch.float)
    torch.set_float32_matmul_precision("medium")
    seed_everything(seed)

    # Ignore common warnnings for clarity
    filterwarnings("ignore", message=".*At least one mel filterbank has all zero values.*")
    filterwarnings("ignore", message=".*is too large for input signal.*")
    filterwarnings("ignore", message=".*Consider increasing the value of the `num_workers` argument*")

    # Load metadata
    metadata = read_csv(all_params["METADATA"])

    # Augmentation pipeline
    aug_pipeline = None
    if all_params["USE_AUG"]:
        aug_pipeline = Compose([
            # Time Shifting
            Shift(
                min_shift=-0.05,
                max_shift=0.7,
                shift_unit="seconds",
                rollover=False,
                p=0.9
            ),
            # Background Noise
            AddGaussianNoise(
                min_amplitude=0.001,
                max_amplitude=0.015,
                p=0.5
            ),
            # Gain
            Gain(
                min_gain_db=-6,
                max_gain_db=6,
                p=0.75
            ),
        ])

    # ====== Training ======
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = LightningModel(hparams=all_params["TRAINING"],
                           model=MyResNet(
                               num_channels=1, num_classes=all_params["MODEL"]["CLASSES"],
                               depth=all_params["MODEL"]["VARIANT"], pretrained=True))
    model.to(device)

    # Data
    datamodule = SimpleDataModule(
        metadata, MemDataset, all_params, transform=extract_features,
        augmentations=aug_pipeline, batch_size=all_params["TRAINING"]["BATCH_SIZE"])

    # Preflight Data Check
    input_size = None
    for stage in ["train", "val", "test", "predict"]:
        tmp_data = datamodule.get_dataloader(stage)
        for batch in tmp_data:
            input_size = list(batch[0].shape[1:])

    # Update param data for logging purposes
    if input_size is None:
        raise ValueError("No input size identified")
    all_params["INPUT_SIZE"] = input_size

    # Set up Logging
    log_base_dir = "logs/"
    timestamp = strftime("%Y%m%d_%H%M%S")
    run_name = all_params["RUN_NAME"]
    logger_name = f"{run_name}_{model.model.name}"

    # Load previous model
    if all_params["TRAINING"]["CONTINUE_RUN"]:
        load_newest_checkpoint(model, log_base_dir, logger_name)

    # Set up the logger
    csv_logger = CSVLogger(save_dir=log_base_dir, name=logger_name, version=f"version_{timestamp}")
    csv_logger.log_hyperparams(all_params)

    # Set up the checkpointer
    checkpoint_path = os.path.join(log_base_dir, logger_name, f"version_{timestamp}")
    checkpoint_callback = CustomModelCheckpoint(
        save_dir=checkpoint_path, params=all_params, monitor='val_accuracy', mode='max')

    # Time model training
    time_limit = all_params["TRAINING"]["TIME_LIMIT"]
    timer = Timer(duration=dict(hours=time_limit))

    # Train the model
    trainer = Trainer(max_epochs=all_params["TRAINING"]["EPOCHS"], log_every_n_steps=1, logger=csv_logger,
                      accelerator="auto", callbacks=[timer, checkpoint_callback])
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)

    # Save and export metrics and models
    file_size, total_metrics = save_export_model(model, all_params, checkpoint_path)

    # Log training durations
    print("Training Time   ", timer.time_elapsed("train"))
    print("Validation Time ", timer.time_elapsed("validate"))
    print("Testing Time    ", timer.time_elapsed("test"))


if __name__ == "__main__":
    main()
