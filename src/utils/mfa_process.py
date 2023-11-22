import tgt
import numpy as np
import torch
from src.text import text_to_sequence


def process_mfa(mfa_path):
    tg = tgt.io.read_textgrid(mfa_path)
    tier = tg.get_tier_by_name("phones")
    durations = []
    phonemes = []
    silent = ["sil", "sp", "spn"]
    text = "{"
    for item in tier._objects:
        start, end, phoneme = item.start_time, item.end_time, item.text

        if phoneme not in silent:
            phonemes.append(phoneme)
        else:
            text += " ".join(phonemes)
            text += "} {"
            phonemes = []
        durations.append(
            int(
                np.round(end * 22050. / 256.)
                - np.round(start * 22050. / 256.)
            )
        )
    text += " ".join(phonemes)
    text += "}"
    character = np.array(text_to_sequence(text, ["english_cleaners"]))
    durations = durations[:character.shape[0]]
    duration = np.array(durations)
    duration = torch.from_numpy(duration)
    character = torch.from_numpy(character)
    return character, duration
