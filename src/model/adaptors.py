import torch
from torch import nn
from torch.nn import Sequential
import torch.nn.functional as F
from src.base import BaseModel
import numpy as np

def create_alignment(base_mat, duration_predictor_output):
    N, L = duration_predictor_output.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(duration_predictor_output[i][j]):
                base_mat[i][count+k][j] = 1
            count = count + duration_predictor_output[i][j]
    return base_mat

def pad(input_ele, mel_max_length=None):  # TODO change code
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


class VariancePredictor(nn.Module):
    """
    Predicts the pitch/duration/energy
    """

    def __init__(
        self,
        input_channels: int = 256,
        output_channels: int = 256,
        kernel_size: int = 3,
        dropout: float = 0,
    ):
        super().__init__()
        self.conv_1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            padding=1
        )
        self.block_1 = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm(output_channels),
            nn.Dropout(dropout),
        )
        self.conv_2 = nn.Conv1d(
            in_channels=output_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            padding=1
        )
        self.block_2 = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm(output_channels),
            nn.Dropout(dropout),
        )
        self.linear = nn.Linear(output_channels, 1)
        self.relu = nn.ReLU()

    def forward(self, x, mask=None):  # x - (B, T, C)
        x = self.conv_1(x.transpose(1, 2)).transpose(1, 2) # (B, T, C)
        x = self.block_1(x)
        x = self.conv_2(x.transpose(1, 2)).transpose(1, 2)
        x = self.block_2(x)
        x = self.linear(x)
        x = self.relu(x)

        x = x.squeeze()
        
        if not self.training:
            x = x.unsqueeze(0)
        return x


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self, 
        input_channels: int = 256,
        output_channels: int = 256,
        kernel_size: int = 3,
        dropout: float = 0,
        ):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = VariancePredictor(input_channels, 
            output_channels,
            kernel_size,
            dropout
        )

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)
        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output

    def forward(self, x, target=None, mel_max_length=None, alpha=1.0):
        duration_predictor_output = self.duration_predictor(x)

        if target is not None:
            output = self.LR(x, target, mel_max_length)
            return output, duration_predictor_output
        else:
            # we remove 1 from exp because we estimate (target + 1), also we ensure that min is 0
            duration_predictor_output = (((torch.exp(duration_predictor_output) - 1) * alpha) + 0.5).int()
            duration_predictor_output[duration_predictor_output < 0] = 0

            output = self.LR(x, duration_predictor_output)
            mel_pos = torch.stack(
                [torch.Tensor([i+1  for i in range(output.size(1))])]
            ).long().to(x.device)
            return output, mel_pos


class VarianceAdaptor(nn.Module):
    def __init__(
        self,
        input_channels: int = 64,
        output_channels: int = 64,
        kernel_size: int = 3,
        dropout: float = 0,
        n_bins: int = 256,
        encoder_hidden: int = 256,
        pitch_min: float = 60.0,
        pitch_max: float = 455.0,
        energy_min: float = 0.0,
        energy_max: float = 150.0,
    ):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.length_regulator = LengthRegulator(
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.pitch_predictor = VariancePredictor(
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.energy_predictor = VariancePredictor(
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.pitch_buckets = nn.Parameter(
            torch.exp(torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)),
            requires_grad=False,
        )
        self.energy_buckets = nn.Parameter(
            torch.exp(
                torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
            ),
            requires_grad=False,
        )

        self.pitch_embbeding = nn.Embedding(n_bins, encoder_hidden)
        self.energy_embbeding = nn.Embedding(n_bins, encoder_hidden)

    def forward(
        self,
        x,
        mel_max_length=None,
        duration_target=None,
        pitch_target=None,
        energy_target=None,
        duration_control=1.0,
        pitch_control=1.0,
        energy_control=1.0,
    ):
        x, duration_prediction = self.length_regulator(x, target=duration_target, mel_max_length=mel_max_length, alpha=duration_control)
        pitch_prediction = self.pitch_predictor(x, None)
        energy_prediction = self.energy_predictor(x, None)
        if pitch_target is None:
            pitch_prediction = pitch_prediction * pitch_control
            pitch_embedding = self.pitch_embbeding(
                torch.bucketize(pitch_prediction, self.pitch_buckets)
            )
        else:
            pitch_embedding = self.pitch_embbeding(
                torch.bucketize(pitch_target, self.pitch_buckets)
            )

        if energy_target is None:
            energy_prediction = energy_prediction * energy_control
            energy_embedding = self.energy_embbeding(
                torch.bucketize(energy_prediction, self.energy_buckets)
            )
        else:
            energy_embedding = self.energy_embbeding(
                torch.bucketize(energy_target, self.energy_buckets)
            )

        x = x + pitch_embedding
        x = x + energy_embedding
        return x, pitch_prediction, energy_prediction, duration_prediction
