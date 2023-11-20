import torch
from torch import nn
import torch.nn.functional as F
from src.base import BaseModel
from src.model.adaptors import VarianceAdaptor
from src.model.coders import Encoder, Decoder


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
        x, mask = self.encoder(src_seq, src_pos)

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

        x, pitch_prediction, energy_prediction, log_duration_prediction = output
        output = self.decoder(x, mel_pos)
        output = self.mel_linear(output)

        return {"mel_predictions": output, 
                "log_duration_predictions": log_duration_prediction,
                "pitch_predictions": pitch_prediction,
                "energy_predictions": energy_prediction}
