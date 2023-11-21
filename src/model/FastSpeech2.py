import torch
from torch import nn
import torch.nn.functional as F
from src.base import BaseModel
from src.model.adaptors import VarianceAdaptor
from src.model.coders import Encoder, Decoder

def get_mask_from_lengths(lengths, max_len=None):
    if max_len == None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, 1, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()

    return mask

class FastSpeech2(nn.Module):
    """FastSpeech2"""

    def __init__(self, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(**model_config["variance_adaptor"])
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["decoder_dim"],
            model_config["num_mels"],
        )

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(
        self,
        src_seq,
        src_pos,
        mel_pos=None,
        mel_max_length=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        pitch_control=1.0,
        energy_control=1.0,
        duration_control=1.0,
        **kwargs
    ):
        x, _ = self.encoder(src_seq, src_pos)
        if self.training:
            output = self.variance_adaptor(
                x,
                mel_max_length=mel_max_length,
                duration_target=duration_target,
                pitch_target=pitch_target,
                energy_target=energy_target,
                duration_control=duration_control,
                pitch_control=pitch_control,
                energy_control=energy_control,
            )
            x, log_pitch_prediction, log_energy_prediction, log_duration_prediction, _ = output
            output = self.decoder(x, mel_pos)
            output = self.mask_tensor(output, mel_pos, mel_max_length)
            output = self.mel_linear(output)
        else:
            output = self.variance_adaptor(
                x,
                duration_control=duration_control,
                pitch_control=pitch_control,
                energy_control=energy_control,
            )
            x, log_pitch_prediction, log_energy_prediction, log_duration_prediction, mel_pos = output
            output = self.decoder(x, mel_pos)
            output = self.mel_linear(output)

        return {"mel_predictions": output, 
                "log_duration_predictions": log_duration_prediction,
                "log_pitch_predictions": log_pitch_prediction,
                "log_energy_predictions": log_energy_prediction}
