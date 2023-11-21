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
        log_pitch_predictions,
        log_energy_predictions,
        log_duration_predictions,
        mel_target,
        pitch_target,
        energy_target,
        duration_target,
        **kwargs,
    ):
        mel_loss = self.mae_loss(mel_predictions, mel_target)
        log_duration_targets = torch.log(duration_target.float() + 1)
        log_pitch_targets = torch.log(pitch_target.float() + 1)
        log_energy_targets = torch.log(energy_target + 1)

        pitch_loss = self.mse_loss(log_pitch_predictions, log_pitch_targets)
        energy_loss = self.mse_loss(log_energy_predictions, log_energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        total_loss = mel_loss + duration_loss + pitch_loss + energy_loss

        return total_loss, mel_loss, pitch_loss, energy_loss, duration_loss
