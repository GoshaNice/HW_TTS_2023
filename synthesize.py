import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

import src.model as module_model
from src.trainer import Trainer
from src.utils import ROOT_PATH
from src.utils.object_loading import get_dataloaders
from src.utils.parse_config import ConfigParser
import pyloudnorm as pyln
import torch.nn.functional as F
import numpy as np
from src import text
from waveglow.utils import get_WaveGlow
import waveglow
import torch
import torchaudio

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"

def main(config, out_file):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"We are running on {device}")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    vocoder = get_WaveGlow()
    vocoder = vocoder.to(device)
    
    #texts = "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest"
    texts = "Massachusetts Institute of Technology may be best known for its math, science and engineering education"
    src_seq = torch.from_numpy(np.array(text.text_to_sequence(texts, ["english_cleaners"])))

    src_pos = list()
    src_pos.append(np.arange(1, int(src_seq.size(0)) + 1))
    src_pos = torch.from_numpy(np.array(src_pos)).to(device)
    src_seq = src_seq.unsqueeze(0).to(device)
    
    
    for duration_control in [0.8, 1, 1.2]:
        for pitch_control in [0.8, 1, 1.2]:
            for energy_control in [0.8, 1, 1.2]:
                model.eval()
                output = model(src_seq, 
                               src_pos,
                               duration_control = duration_control,
                               pitch_control = pitch_control,
                               energy_control = energy_control)
                melspec = output["mel_predictions"].squeeze()
                mel = melspec.unsqueeze(0).contiguous().transpose(1, 2).to(device)
                waveglow.inference.inference(
                    mel, vocoder,
                    f"results/d={duration_control}_p={pitch_control}_e={energy_control}_waveglow.wav"
                )


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output",
        type=str,
        help="Dir to write results",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to texts.txt to synthesize",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)
    
    main(config, args.output)
