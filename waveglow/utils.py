import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

import hparams


def get_WaveGlow():
    waveglow_path = os.path.join("waveglow", "pretrained_model")
    waveglow_path = os.path.join(waveglow_path, "waveglow_256channels.pt")
    print(waveglow_path)
    wave_glow = torch.load(waveglow_path)['model']
    wave_glow = wave_glow.remove_weightnorm(wave_glow)
    wave_glow.cuda().eval()
    for m in wave_glow.modules():
        if 'Conv' in str(type(m)):
            setattr(m, 'padding_mode', 'zeros')

    return wave_glow