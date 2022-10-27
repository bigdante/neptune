# -*- encoding: utf-8 -*-
'''
@File    :   base_strategy.py
@Time    :   2021/10/08 22:22:42
@Author  :   Ming Ding
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch
import torch.nn.functional as F


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-65504):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # convert to 1D
        logits = logits.view(logits.size()[1]).contiguous()
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        # going back to 2D
        logits = logits.view(1, -1).contiguous()

    return logits


class ConstrainedBaseStrategyCalibrate:
    def __init__(self, invalid_slices=[], temperature=1., top_k=200, eps=1e-4, top_p=0.0, end_tokens=None, tokenizer=None):
        self.invalid_slices = invalid_slices
        self.temperature = temperature
        self.topk = top_k
        self.top_p = top_p
        self.eps = eps
        if end_tokens is None:
            end_tokens = []
        self.end_tokens = end_tokens
        self._is_done = False

        self.command_tokens = [ct.Id for ct in tokenizer._command_tokens]
        self.target_token = tokenizer.EncodeAsIds(" Republican")[0]
        self.context_tokens, self.indices, self.mask, self.confidence, self.tokens, self.new_seq = None, None, None, [], [], True
        self.calibrate_weight = None

    def refresh(self):
        self.context_tokens, self.indices, self.mask, self.confidence, self.tokens, self.new_seq = None, None, None, [], [], True
        self.calibrate_weight = None

    def create_mask(self, batch_size, vocab_size, indices, device):
        mask = torch.full((batch_size, vocab_size), -65504).float().to(device)
        for idx in range(batch_size):
            mask[idx, indices[idx]] = 0
        self.mask = mask#.unsqueeze(1).repeat(1, self.num_beams, 1).view(batch_size * self.num_beams, -1)

    def init_constraints(self):
        self.new_seq = False
        self.calibrate_weight = None
        self.indices = []
        for batch_idx in range(self.context_tokens.shape[0]):
            self.indices.append(torch.LongTensor(
                sorted(list(set(self.context_tokens.detach().cpu().tolist()[batch_idx])) +
                       self.command_tokens)
            ).to(self.context_tokens.device))

    @property
    def is_done(self) -> bool:
        return self._is_done

    def get_calibrated_weight(self, logits, tokens, mems, temperature=None):
        batch_size, vocab_size = logits.shape
        if self.new_seq:
            self.init_constraints()
            self.create_mask(batch_size, vocab_size, self.indices, tokens.device)

        if temperature is None:
            temperature = self.temperature
        logits = logits / temperature
        for invalid_slice in self.invalid_slices:
            logits[..., invalid_slice] = -65504
        logits += self.mask

        softmaxed_logits = F.softmax(logits.float(), dim=-1)
        eos_prob = softmaxed_logits[:, self.end_tokens].sum().item()
        print("Calibrate eos prob:", eos_prob)
        weight = torch.ones_like(softmaxed_logits) * (0.5 / (1 - eos_prob))
        weight[:, self.end_tokens] = 1 / eos_prob
        # weight = torch.ones_like(softmaxed_logits) / (top_k_logits(softmaxed_logits, 40, filter_value=1))
        self.calibrate_weight = weight
        # print("Calibrate coefficient:", 1 / eos_prob)

    def forward(self, logits, tokens, mems, temperature=None, is_first_token=False):
        batch_size, vocab_size = logits.shape
        if self.new_seq:
            self.init_constraints()
            self.create_mask(batch_size, vocab_size, self.indices, tokens.device)

        if temperature is None:
            temperature = self.temperature
        logits = logits / temperature
        for invalid_slice in self.invalid_slices:
            logits[..., invalid_slice] = -65504
        logits += self.mask
        if self.calibrate_weight is not None and is_first_token:
            softmaxed_logits = F.softmax(logits.float(), dim=-1) * self.calibrate_weight
        else:
            softmaxed_logits = F.softmax(logits.float(), dim=-1)
        print("End token probs:", softmaxed_logits[:, self.end_tokens].sum().item())
        print("Target token prob:", softmaxed_logits[:, self.target_token].item())
        # print("End token probs before top_k:", softmaxed_logits[:, self.end_tokens].sum().item())
        # print("Target token prob before top_k:", softmaxed_logits[:, self.target_token].item())
        logits = top_k_logits(softmaxed_logits, self.topk, self.top_p)
        probs = F.softmax(logits.float(), dim=-1)  # float is essetial, due to a bug in Pytorch
        pred = torch.multinomial(probs, num_samples=1)
        self.confidence.append(softmaxed_logits[:, pred.cpu().item()].cpu().item())
        self.tokens.append(pred.cpu().item())
        if pred.numel() == 1 and pred.item() in self.end_tokens:
            self._is_done = True
        tokens = torch.cat((tokens, pred.view(tokens.shape[0], 1)), dim=1)
        return tokens, mems

    def finalize(self, tokens, mems):
        self._is_done = False
        return tokens, mems
