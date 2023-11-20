import logging
from typing import List
import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    len_arr = np.array([d["text"].size(0) for d in dataset_items])
    cut_list = np.argsort(-len_arr)
    return reprocess_tensor(dataset_items, cut_list=cut_list)


def reprocess_tensor(batch, cut_list):
    texts = [batch[ind]["text"] for ind in cut_list]
    mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
    durations = [batch[ind]["duration"] for ind in cut_list]
    pitches = [batch[ind]["pitch"] for ind in cut_list]
    energies = [batch[ind]["energy"] for ind in cut_list]

    length_text = np.array([])
    for text in texts:
        length_text = np.append(length_text, text.size(0))

    length_duration = np.array([])
    for duration in durations:
        length_duration = np.append(length_duration, duration.size(0))

    src_pos = list()
    max_len = int(max(length_text))
    max_len_duration = int(max(length_duration))
    max_len = max(max_len, max_len_duration)
    for length_src_row in length_text:
        src_pos.append(
            np.pad(
                [i + 1 for i in range(int(length_src_row))],
                (0, max_len - int(length_src_row)),
                "constant",
            )
        )
    src_pos = torch.from_numpy(np.array(src_pos))

    length_mel = np.array(list())
    for mel in mel_targets:
        length_mel = np.append(length_mel, mel.size(0))

    mel_pos = list()
    max_mel_len = int(max(length_mel))
    for length_mel_row in length_mel:
        mel_pos.append(
            np.pad(
                [i + 1 for i in range(int(length_mel_row))],
                (0, max_mel_len - int(length_mel_row)),
                "constant",
            )
        )
    mel_pos = torch.from_numpy(np.array(mel_pos))

    src_seq = pad_1D_tensor(texts)
    durations = pad_1D_tensor(durations, max_pad_len=max_len)
    pitches = pad_1D_tensor(pitches)
    energies = pad_1D_tensor(energies)
    mel_targets = pad_2D_tensor(mel_targets)
    src_seq = pad_1D_tensor(texts, max_pad_len=max_len)

    out = {
        "src_seq": src_seq,
        "src_pos": src_pos,
        "mel_target": mel_targets,
        "duration_target": durations,
        "pitch_target": pitches,
        "energy_target": energies,
        "mel_pos": mel_pos,
        "mel_max_length": max_mel_len,
    }

    return out


def pad_1D_tensor(inputs, PAD=0, max_pad_len=None):
    def pad_data(x, length, PAD):
        x_padded = F.pad(x, (0, length - x.shape[0]))
        return x_padded

    max_len = max((len(x) for x in inputs))
    if max_pad_len:
        max_len = max(max_len, max_pad_len)
    padded = torch.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D_tensor(inputs, maxlen=None):
    def pad(x, max_len):
        if x.size(0) > max_len:
            raise ValueError("not max_len")

        s = x.size(1)
        x_padded = F.pad(x, (0, 0, 0, max_len - x.size(0)))
        return x_padded[:, :s]

    if maxlen:
        output = torch.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(x.size(0) for x in inputs)
        output = torch.stack([pad(x, max_len) for x in inputs])

    return output
