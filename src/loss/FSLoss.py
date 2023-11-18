import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


class FastSpeech2Loss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(
        self,
        mel_predictions,
        pitch_predictions,
        energy_predictions,
        log_duration_predictions,
        mel_target,
        pitch_target,
        energy_target,
        duration_target,
        **kwargs,
    ):
        log_duration_targets = torch.log(duration_target.float() + 1e-8)
        print(mel_predictions.shape)
        print(mel_target.shape)
        mel_loss = self.mae_loss(mel_predictions, mel_target)

        pitch_loss = self.mse_loss(pitch_predictions, pitch_target)
        energy_loss = self.mse_loss(energy_predictions, energy_target)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        total_loss = mel_loss + duration_loss + pitch_loss + energy_loss

        return total_loss, mel_loss, pitch_loss, energy_loss, duration_loss
