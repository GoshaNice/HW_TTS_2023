import json
import logging
import os
import shutil
from curses.ascii import isascii
from pathlib import Path

import torchaudio
from src.base.base_dataset import BaseDataset
from src.utils import ROOT_PATH
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm
import numpy as np
import gdown
import torch
import pyworld as pw
import librosa
from scipy import interpolate

logger = logging.getLogger(__name__)

URL_LINKS = {
    "dataset": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
    "mels": "https://drive.google.com/u/0/uc?id=1cJKJTmYd905a-9GFoo5gKjzhKjUVj83j",
    "alignments": "https://github.com/xcmyz/FastSpeech/raw/master/alignments.zip",
    "mfa": "https://drive.google.com/u/0/uc?id=16tJW_myv6DEXmtaPDuj3hDU0rO61hE7A"
}


class LJspeechDataset(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "ljspeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _load_dataset(self):
        arch_path = self._data_dir / "LJSpeech-1.1.tar.bz2"
        print("Loading LJSpeech")
        download_file(URL_LINKS["dataset"], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in tqdm((self._data_dir / "LJSpeech-1.1").iterdir()):
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LJSpeech-1.1"))

        files = [file_name for file_name in (self._data_dir / "wavs").iterdir()]
        train_length = int(0.85 * len(files))  # hand split, test ~ 15%
        (self._data_dir / "train").mkdir(exist_ok=True, parents=True)
        (self._data_dir / "test").mkdir(exist_ok=True, parents=True)
        for i, fpath in tqdm(enumerate((self._data_dir / "wavs").iterdir())):
            if i < train_length:
                shutil.copy(str(fpath), str(self._data_dir / "train" / fpath.name))
            else:
                shutil.copy(str(fpath), str(self._data_dir / "test" / fpath.name))
        # shutil.rmtree(str(self._data_dir / "wavs"))

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _load_mel(self):
        print("Loading Mel")
        output = self._data_dir / "mel.tar.gz"
        gdown.download(URL_LINKS["mels"], str(output), quiet=False)
        shutil.unpack_archive(output, self._data_dir)
        os.remove(str(output))

    def _load_aligments(self):
        print("Loading Alignments")
        arch_path = self._data_dir / "alignments.zip"
        download_file(URL_LINKS["alignments"], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        os.remove(str(arch_path))

    def _count_energy(self):
        print("Counting Energy")
        energy_dir = self._data_dir / "energy"
        mel_dir = self._data_dir / "mels"
        energy_dir.mkdir(exist_ok=True, parents=True)

        for mel_path in mel_dir.iterdir():
            mel = np.load(mel_path)
            energy = np.linalg.norm(mel, axis=-1)
            energy_name = mel_path.name.replace("mel", "energy")
            np.save(energy_dir / energy_name, energy)

    def _count_pitch(self):
        print("Counting Pitch")
        wav_dir = self._data_dir / "wavs"
        mel_dir = self._data_dir / "mels"
        pitch_dir = self._data_dir / "pitch"
        pitch_dir.mkdir(exist_ok=True, parents=True)

        for i, wav_path in tqdm(enumerate(wav_dir.iterdir())):
            audio, sr = librosa.load(wav_path, dtype=np.float64)
            mel_name = f"ljspeech-mel-{(i+1):05d}.npy"
            mel = np.load(mel_dir / mel_name)

            frame_period = (audio.shape[0] / sr * 1000) / mel.shape[0]
            _f0, t = pw.dio(audio, sr, frame_period=frame_period)
            f0 = pw.stonemask(audio, _f0, t, sr)[: mel.shape[0]].astype(np.float32)

            x = np.arange(f0.shape[0])[f0 != 0]
            y = f0[f0 != 0]
            below, above = f0[f0 != 0][0], f0[f0 != 0][-1]
            transform = interpolate.interp1d(
                x, y, bounds_error=False, fill_value=(below, above)
            )
            f0 = transform(np.arange(f0.shape[0]))
            pitch_name = f"ljspeech-pitch-{(i+1):05d}.npy"
            np.save(pitch_dir / pitch_name, f0)

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._load_dataset()

        mels_dir = self._data_dir / "mels"
        if not mels_dir.exists():
            self._load_mel()

        aligments_dir = self._data_dir / "alignments"
        if not aligments_dir.exists():
            self._load_aligments()

        pitch_dir = self._data_dir / "pitch"
        if not pitch_dir.exists():
            self._count_pitch()

        energy_dir = self._data_dir / "energy"
        if not energy_dir.exists():
            self._count_energy()

        wav_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            if any([f.endswith(".wav") for f in filenames]):
                wav_dirs.add(dirpath)
        for wav_dir in tqdm(list(wav_dirs), desc=f"Preparing ljspeech folders: {part}"):
            wav_dir = Path(wav_dir)
            trans_path = list(self._data_dir.glob("*.csv"))[0]
            with trans_path.open() as f:
                for i, line in enumerate(f):
                    w_id = line.split("|")[0]
                    w_text = line.split("|")[2].strip()
                    wav_path = wav_dir / f"{w_id}.wav"
                    if not wav_path.exists():  # elem in another part
                        continue
                    t_info = torchaudio.info(str(wav_path))
                    length = t_info.num_frames / t_info.sample_rate
                    if w_text.isascii():
                        index.append(
                            {
                                "path": str(wav_path.absolute().resolve()),
                                "text": w_text.lower(),
                                "audio_len": length,
                                "aligment_path": str(
                                    self._data_dir / "alignments" / f"{i}.npy"
                                ),
                                "mel_path": str(
                                    self._data_dir
                                    / "mels"
                                    / f"ljspeech-mel-{(i+1):05d}.npy"
                                ),
                                "energy_path": str(
                                    self._data_dir
                                    / "energy"
                                    / f"ljspeech-energy-{(i+1):05d}.npy"
                                ),
                                "pitch_path": str(
                                    self._data_dir
                                    / "pitch"
                                    / f"ljspeech-pitch-{(i+1):05d}.npy"
                                ),
                            }
                        )
        return index
