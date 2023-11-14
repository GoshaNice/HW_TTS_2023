from torch import nn
from torch.nn import Sequential

from src.base import BaseModel


class DeepSpeech2(BaseModel):
    def __init__(
        self, input_dim: int, n_class: int, gru_hidden: int, fc_hidden: int, **batch
    ):
        super().__init__(input_dim, n_class, **batch)
        self.conv = Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=(11, 41), padding="same"
            ),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=(11, 21), padding="same"
            ),
            nn.BatchNorm2d(num_features=32),
        )
        self.rnn = nn.GRU(
            input_size=input_dim * 32,
            hidden_size=gru_hidden,
            num_layers=5,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = Sequential(
            nn.Linear(in_features=gru_hidden * 2, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=n_class),
        )

    def forward(self, spectrogram, **batch):
        x = self.conv(spectrogram.unsqueeze(1))
        x = x.permute(0, 3, 1, 2)
        x = x.flatten(2)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return {"logits": x}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
