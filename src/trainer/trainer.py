import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
import librosa
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm
import torchaudio

from src.base import BaseTrainer
from src.base.base_text_encoder import BaseTextEncoder
from src.logger.utils import plot_spectrogram_to_buf
from src.metric.utils import calc_cer, calc_wer
from src.utils import inf_loop, MetricTracker
from src.text import text_to_sequence

import numpy as np

from waveglow.utils import get_WaveGlow
import waveglow as waveglow


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        config,
        device,
        dataloaders,
        lr_scheduler=None,
        len_epoch=None,
        skip_oom=True,
    ):
        super().__init__(model, criterion, metrics, optimizer, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }
        self.lr_scheduler = lr_scheduler
        self.log_step = 50

        self.train_metrics = MetricTracker(
            "loss", "mel_loss", "pitch_loss", "energy_loss", "duration_loss", "grad norm", *[m.name for m in self.metrics], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            *[m.name for m in self.metrics], writer=self.writer
        )
        self.device = device

        self.waveglow = get_WaveGlow()
        self.waveglow = self.waveglow.to(device)

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["src_seq", "src_pos", "mel_target", "duration_target", "pitch_target", "energy_target", "mel_pos"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.optimizer.get_last_lr()
                )

                #self._log_spectrogram(batch["mel_predictions"])
                self._log_audio(batch["mel_predictions"])
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()
        outputs = self.model(**batch)
        if type(outputs) is dict:
            batch.update(outputs)
        else:
            batch["logits"] = outputs

        if is_train:
            batch["loss"], batch["mel_loss"], batch["pitch_loss"], batch["energy_loss"], batch["duration_loss"] = self.criterion(**batch)
            metrics.update("loss", batch["loss"].item())
            metrics.update("mel_loss", batch["mel_loss"].item())
            metrics.update("pitch_loss", batch["pitch_loss"].item())
            metrics.update("energy_loss", batch["energy_loss"].item())
            metrics.update("duration_loss", batch["duration_loss"].item())
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step_and_update_lr()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        for met in self.metrics:
            metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_audio(batch["mel_predictions"])
            self._log_test_synthesis()

        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))

    def _log_audio(self, spectrogram_batch):
        melspec = random.choice(spectrogram_batch.cpu())
        mel = melspec.unsqueeze(0).contiguous().transpose(1, 2).to(self.device)
        waveglow.inference.inference(
            mel, self.waveglow,
            f"results/s={1}_{1}_waveglow.wav"
        )
        audio, sr = torchaudio.load(f"results/s={1}_{1}_waveglow.wav")
        self.writer.add_audio("audio", audio, sample_rate=sr)
    
    def _log_test_synthesis(self):
        texts = "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest"
        src_seq = torch.from_numpy(np.array(text_to_sequence(texts, ["english_cleaners"])))

        src_pos = list()
        src_pos.append(np.arange(1, int(src_seq.size(0)) + 1))
        src_pos = torch.from_numpy(np.array(src_pos)).to(self.device)
        src_seq = src_seq.unsqueeze(0).to(self.device)
        
        self.model.eval()
        output = self.model(src_seq, src_pos)
        melspec = output["mel_predictions"].squeeze()
        mel = melspec.unsqueeze(0).contiguous().transpose(1, 2).to(self.device)
        waveglow.inference.inference(
            mel, self.waveglow,
            f"results/s={1}_{1}_waveglow.wav"
        )
        audio, sr = torchaudio.load(f"results/s={1}_{1}_waveglow.wav")
        self.writer.add_audio("test_audio", audio, sample_rate=sr)

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
