#!/usr/bin/env python3
"""
lm-evaluation-harness models for eMG-RNN (expectation-based Minimalist
Grammar Recurrent Neural Network).

Registered models
-----------------
emg_lm
    Token-level eMG-RNN checkpoints in the format of the 2024 BabyLM release
    (Chesi et al. 2024): a folder with ``model.pt`` and the tokenizer files.
    ``model.pt`` may hold a plain state dict (also one saved from
    DataParallel/DistributedDataParallel) or the training-script dict with a
    "state_dict" key. Both 2024 cell variants are supported; vocabulary size,
    dimensions and number of layers are read from the checkpoint.
emg_incremental
    The tokenizer-free eMG-RNN of this repository
    (model_eMG_incremental.IncrementalEMGModel): a folder written by
    train_eMG_incremental.py or IncrementalEMGModel.save_pretrained().

Token-level models of this repository (model_eMG.EMGLanguageModel) need no
wrapper: save them with save_pretrained() and use lm-eval's built-in "hf"
model.

Scoring
-------
For a request (context, continuation) every model returns

    log P(continuation | context) = sum_i log P(u_i | everything before u_i)

summed over the units u_i of the continuation (tokens for emg_lm, words for
emg_incremental). The probability of u_i is read from the model output at
the position of the unit before it, i.e. the next-unit prediction the models
are trained for. lm-eval compares these sums; on BLiMP it checks that the
grammatical sentence of each pair gets the higher one.

The 2024 checkpoints were trained on lines without a start symbol, so they
never learned to predict the first token of a line. When the context is
empty (as in BLiMP), emg_lm therefore conditions on the first token of the
continuation and scores the following ones; ``prefix_token`` names a token
to prepend instead. emg_incremental was trained with <s> and scores every
word.

Model arguments
---------------
emg_lm: pretrained (folder), checkpoint (default model.pt), tokenizer
    (default: the same folder), normalize (default True: the 2024
    preprocessing, i.e. lowercasing, spaces around . , ! ? and removal of the
    speaker labels a:, b:, david:, sarah:), prefix_token (default none),
    max_gen_toks (default 256).
emg_incremental: pretrained (folder), include_eos (default False: add the
    probability of </s> to every continuation; for sentence-level tasks only).
Both: device, batch_size.

Usage
-----
    python lm_eval_emg.py --model emg_lm --model_args pretrained=PATH --tasks blimp --batch_size 32
    python lm_eval_emg.py --model emg_incremental --model_args pretrained=PATH --tasks blimp

or from Python:

    import lm_eval, lm_eval_emg
    lm_eval.simple_evaluate(model="emg_lm", model_args="pretrained=PATH", tasks=["blimp"])

Author: Cristiano Chesi (IUSS Pavia, NeTS Lab).
Development, debugging and optimization assisted by Claude (Anthropic).
Released under the MIT License (see LICENSE).
"""

import logging
import os
import re
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

logger = logging.getLogger(__name__)


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y")
    return bool(value)


# ============================================================================
# 2024 architecture (same parameter names as the 2024 release)
# ============================================================================

class LegacyEMGCell(nn.Module):
    """
    Cell of the 2024 release (03-models/model_eMG_RNN.py):
        x~ = W_x x;  rho, mu = sigmoid(W_h m_{t-1});
        c_t = tanh(c_{t-1} + rho * x~);  m_t = (1 - mu) * x~ + mu * c_t
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_transform = nn.Linear(input_size, hidden_size)
        self.hidden_transform = nn.Linear(hidden_size, 2 * hidden_size)

    def forward(self, x, state):
        m_prev, c_prev = state
        x_tilde = self.input_transform(x)
        retain, merge = self.hidden_transform(m_prev).chunk(2, dim=-1)
        retain, merge = torch.sigmoid(retain), torch.sigmoid(merge)
        c = torch.tanh(c_prev + retain * x_tilde)
        return (1 - merge) * x_tilde + merge * c, c


class LegacyEMGBaseCell(nn.Module):
    """
    Cell of the 2024 release (03-models/model_eMG_RNN_base.py), with gates
    computed from [x; m_{t-1}]:
        c_t = tanh(c_{t-1} + rho * x);  m_t = mu * x + (1 - mu) * c_t
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.combined_transform = nn.Linear(input_size + hidden_size, 2 * hidden_size)

    def forward(self, x, state):
        m_prev, c_prev = state
        retain, merge = self.combined_transform(torch.cat((x, m_prev), dim=-1)).chunk(2, dim=-1)
        retain, merge = torch.sigmoid(retain), torch.sigmoid(merge)
        c = torch.tanh(c_prev + retain * x)
        return merge * x + (1 - merge) * c, c


class LegacyEMG(nn.Module):
    def __init__(self, cell_class, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cells = nn.ModuleList([cell_class(input_size if i == 0 else hidden_size, hidden_size)
                                    for i in range(num_layers)])

    def forward(self, x, hidden=None):
        """x: (batch, seq_len, input); returns every layer's output (batch, seq_len, layers, hidden) and the state."""
        batch = x.size(0)
        if hidden is None:
            zeros = x.new_zeros(batch, self.hidden_size)
            hidden = [(zeros, zeros) for _ in self.cells]
        hidden = list(hidden)
        outputs = []
        for x_t in x.unbind(1):
            layer_input, per_layer = x_t, []
            for i, cell in enumerate(self.cells):
                hidden[i] = cell(layer_input, hidden[i])
                layer_input = hidden[i][0]
                per_layer.append(layer_input)
            outputs.append(torch.stack(per_layer, dim=1))
        return torch.stack(outputs, dim=1), hidden


class LegacyEMGLanguageModel(nn.Module):
    """
    Language model of the 2024 release, with the same forward pass: the
    output layer is applied to the output of every eMG layer and the
    per-layer logits are concatenated on the last axis. With one layer (the
    released models) these are the usual (batch, seq_len, vocab) logits; with
    more layers training saw the concatenation, so scoring uses it too.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int,
                 variant: str = "separate"):
        super().__init__()
        cell = LegacyEMGCell if variant == "separate" else LegacyEMGBaseCell
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.emg = LegacyEMG(cell, embedding_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        out, hidden = self.emg(self.embedding(x), hidden)
        return self.fc(out).reshape(x.size(0), x.size(1), -1), hidden

    @classmethod
    def from_checkpoint(cls, path: str, map_location="cpu") -> "LegacyEMGLanguageModel":
        state = torch.load(path, map_location=map_location)
        if isinstance(state.get("state_dict"), dict):
            state = state["state_dict"]
        state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
        vocab_size, embedding_dim = state["embedding.weight"].shape
        hidden_dim = state["fc.weight"].shape[1]
        num_layers = 1 + max(int(k.split(".")[2]) for k in state if k.startswith("emg.cells."))
        variant = "base" if any("combined_transform" in k for k in state) else "separate"
        model = cls(vocab_size, embedding_dim, hidden_dim, num_layers, variant)
        model.load_state_dict(state)
        logger.info("2024 eMG checkpoint: vocab %d, embedding %d, hidden %d, %d layer(s), %s cell",
                    vocab_size, embedding_dim, hidden_dim, num_layers, variant)
        return model


_PUNCTUATION_2024 = re.compile(r"([.,!?])")
_SPEAKERS_2024 = re.compile(r"\b(?:a|b|david|sarah):")


def normalize_2024(text: str) -> str:
    """Preprocessing of the 2024 evaluation: lowercase, spaced . , ! ?, no speaker labels."""
    text = _PUNCTUATION_2024.sub(r" \1 ", text.lower())
    text = _SPEAKERS_2024.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


# ============================================================================
# Shared harness logic
# ============================================================================

class _EMGHarnessLM(LM):
    """Batching and request handling shared by the eMG models."""

    def __init__(self, device: Optional[str] = None, batch_size=16, max_batch_size=None, **kwargs):
        super().__init__()
        if kwargs:
            logger.warning("Ignoring unknown model arguments: %s", sorted(kwargs))
        if device in (None, "", "auto"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._batch_size = 16 if str(batch_size).startswith("auto") else int(batch_size)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def batch_size(self) -> int:
        return self._batch_size

    # --------------------------------------------------- model-specific hooks

    def _units(self, text: str) -> list:
        """Preprocess and split a string into scoring units."""
        raise NotImplementedError

    def _prefix(self) -> list:
        """Units prepended to every sequence."""
        return []

    def _score(self, sequences: List[list], spans: List[Tuple[int, int]]) -> List[Tuple[float, bool]]:
        """For each sequence, the summed log-probability of units[start:end] and whether they are all greedy."""
        raise NotImplementedError

    # ------------------------------------------------------------- requests

    def _item(self, context: str, continuation: str, extra: Sequence = ()) -> Tuple[list, Tuple[int, int]]:
        sequence = self._prefix() + self._units(context)
        start = len(sequence)
        sequence = sequence + self._units(continuation) + list(extra)
        return sequence, (max(start, 1), len(sequence))

    def _run(self, items: List[Tuple[list, Tuple[int, int]]]) -> List[Tuple[float, bool]]:
        order = sorted(range(len(items)), key=lambda i: -len(items[i][0]))
        results: List[Optional[Tuple[float, bool]]] = [None] * len(items)
        for b in range(0, len(order), self.batch_size):
            chunk = order[b:b + self.batch_size]
            scored = [i for i in chunk if items[i][1][0] < items[i][1][1]]
            for i in chunk:
                if i not in scored:
                    results[i] = (0.0, True)
            if scored:
                outputs = self._score([items[i][0] for i in scored], [items[i][1] for i in scored])
                for i, out in zip(scored, outputs):
                    results[i] = out
        return results

    def _continuation_extra(self) -> Sequence:
        return ()

    def loglikelihood(self, requests, disable_tqdm: bool = False) -> List[Tuple[float, bool]]:
        items = [self._item(*req.args[:2], extra=self._continuation_extra()) for req in requests]
        return self._run(items)

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> List[float]:
        items = [self._item("", req.args[0], extra=self._continuation_extra()) for req in requests]
        return [score for score, _ in self._run(items)]

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        raise NotImplementedError(f"{type(self).__name__} supports log-likelihood tasks only.")


# ============================================================================
# 2024 checkpoints
# ============================================================================

@register_model("emg_lm")
class EMGLM(_EMGHarnessLM):
    """lm-eval model for eMG-RNN checkpoints of the 2024 BabyLM release (see module docstring)."""

    def __init__(self, pretrained: str, checkpoint: str = "model.pt", tokenizer: Optional[str] = None,
                 normalize=True, prefix_token: Optional[str] = None, max_gen_toks: int = 256,
                 device: Optional[str] = None, batch_size=16, max_batch_size=None, **kwargs):
        super().__init__(device=device, batch_size=batch_size, max_batch_size=max_batch_size, **kwargs)
        from transformers import AutoTokenizer

        self.model = LegacyEMGLanguageModel.from_checkpoint(
            os.path.join(pretrained, checkpoint), map_location="cpu").to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer or pretrained)
        self.normalize = _as_bool(normalize)
        self.max_gen_toks = int(max_gen_toks)
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        self.eot_id = self.tokenizer.eos_token_id
        if self.eot_id is None and "<eos>" in self.tokenizer.get_vocab():
            self.eot_id = self.tokenizer.convert_tokens_to_ids("<eos>")
        self.prefix_ids: List[int] = []
        if prefix_token:
            vocab = self.tokenizer.get_vocab()
            if prefix_token not in vocab:
                raise ValueError(f"prefix_token {prefix_token!r} is not in the tokenizer vocabulary")
            self.prefix_ids = [vocab[prefix_token]]
        if self.tokenizer.vocab_size > self.model.vocab_size:
            raise ValueError("The tokenizer has more tokens than the model's vocabulary.")

    def _units(self, text: str) -> List[int]:
        text = normalize_2024(text) if self.normalize else text.strip()
        return self.tokenizer.encode(text, add_special_tokens=False) if text else []

    def _prefix(self) -> List[int]:
        return list(self.prefix_ids)

    @torch.no_grad()
    def _score(self, sequences, spans):
        width = max(len(s) for s in sequences)
        ids = torch.full((len(sequences), width), self.pad_id, dtype=torch.long)
        for row, seq in enumerate(sequences):
            ids[row, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        ids = ids.to(self.device)
        logits, _ = self.model(ids[:, :-1])           # right padding never affects earlier positions
        logp = torch.log_softmax(logits.float(), dim=-1)
        targets = ids[:, 1:]
        token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        greedy = logp.argmax(-1) == targets
        # unit k of a sequence is predicted at input position k - 1
        return [(float(token_logp[row, start - 1:end - 1].sum()), bool(greedy[row, start - 1:end - 1].all()))
                for row, (start, end) in enumerate(spans)]

    @torch.no_grad()
    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        """Greedy decoding with the carried recurrent state, stopped at any ``until`` string."""
        outputs = []
        vocab = self.model.vocab_size
        for req in requests:
            context, gen_kwargs = req.args[0], (req.args[1] if len(req.args) > 1 else {})
            until = gen_kwargs.get("until") or []
            until = [until] if isinstance(until, str) else list(until)
            max_new = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks))
            ids = self._prefix() + self._units(context) or [self.pad_id]
            logits, hidden = self.model(torch.tensor([ids], device=self.device))
            new, text = [], ""
            for _ in range(max_new):
                token = int(logits[0, -1, :vocab].argmax())
                if token == self.eot_id:
                    break
                new.append(token)
                text = self.tokenizer.decode(new)
                stops = [text.index(u) for u in until if u in text]
                if stops:
                    text = text[:min(stops)]
                    break
                logits, hidden = self.model(torch.tensor([[token]], device=self.device), hidden)
            outputs.append(text)
        return outputs


# ============================================================================
# Tokenizer-free model
# ============================================================================

@register_model("emg_incremental")
class EMGIncrementalLM(_EMGHarnessLM):
    """lm-eval model for model_eMG_incremental.IncrementalEMGModel (see module docstring)."""

    def __init__(self, pretrained: str, include_eos=False, device: Optional[str] = None,
                 batch_size=32, max_batch_size=None, **kwargs):
        super().__init__(device=device, batch_size=batch_size, max_batch_size=max_batch_size, **kwargs)
        from data_preparation_charwise import BOS_WORD, EOS_WORD, preprocess_line
        from model_eMG_incremental import IncrementalEMGModel

        self._bos, self._eos, self._preprocess = BOS_WORD, EOS_WORD, preprocess_line
        self.model = IncrementalEMGModel.from_pretrained(pretrained, map_location="cpu").to(self.device).eval()
        self.data_config = self.model.config.data_config()
        self.include_eos = _as_bool(include_eos)

    def _units(self, text: str) -> List[str]:
        return self._preprocess(text, self.data_config)

    def _prefix(self) -> List[str]:
        return [self._bos]

    def _continuation_extra(self) -> Sequence:
        return (self._eos,) if self.include_eos else ()

    @torch.no_grad()
    def _score(self, sequences, spans):
        out = self.model(**self.model.encode_sequences(sequences, self.device), return_word_logprobs=True)
        word_logp = out.word_logprobs.float()
        # word k of a sequence is predicted at input position k - 1; is_greedy is not
        # defined for an open vocabulary and is reported as False
        return [(float(word_logp[row, start - 1:end - 1].sum()), False)
                for row, (start, end) in enumerate(spans)]


if __name__ == "__main__":
    from lm_eval.__main__ import cli_evaluate
    cli_evaluate()
