from typing import List

import torch
from torch import Tensor

from src.base.base_metric import BaseMetric
from src.base.base_text_encoder import BaseTextEncoder
from src.metric.utils import calc_cer


class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        cers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            else:
                pred_text = self.text_encoder.decode(log_prob_vec[:length])
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)


class BeamsearchCERMetric(BaseMetric):
    def __init__(
        self, text_encoder: BaseTextEncoder, use_lm: bool = False, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.use_lm = use_lm

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        cers = []
        if self.use_lm:
            probs = log_probs.exp().cpu()
        else:
            probs = log_probs.exp().cpu().detach().numpy()
        lengths = log_probs_length.cpu().detach().numpy()
        for batch, target_text in enumerate(text):
            length = lengths[batch]
            target_text = BaseTextEncoder.normalize_text(target_text)
            if self.use_lm:
                hypos = self.text_encoder.ctc_beam_search_lm(
                    probs[batch][:length], probs_length=length
                )
            else:
                hypos = self.text_encoder.ctc_beam_search(
                    probs[batch][:length], probs_length=length
                )
            pred_text = hypos[0].text
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)
