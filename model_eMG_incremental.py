#!/usr/bin/env python3
"""
Tokenizer-free eMG-RNN (expectation-based Minimalist Grammar Recurrent Neural
Network) with an incremental MorPiece vocabulary.

The model reads words as character sequences and predicts the next word
character by character, so it assigns a probability to any sentence (open
vocabulary) and can be evaluated on BLiMP-style minimal pairs without a
tokenizer. A morpheme vocabulary grows during training following MorPiece
and Yang's Tolerance Principle, and it reaches the network in two ways:

1. Morpheme route: each input word is segmented into vocabulary pieces
   (stem, ##ending, ...) whose embeddings are added, through a learned gate,
   to the character-CNN word vector.
2. TP supervision: the encoder's morpheme-boundary predictor is trained to
   reproduce the boundaries that the Tolerance Principle places in each
   word, wherever the counts are large enough to decide.

Contents
--------
IncrementalMorPieceConfig   hyperparameters (model and text preprocessing)
ToleranceTrie               dual-trie TP boundary discovery, incremental vocabulary
MorphemeIndexer             cached morpheme IDs and boundary targets per word type
prepare_batch               batch from CharBatchCollator -> model inputs
IncrementalMorPieceCNN      character encoder with boundary predictor and morpheme route
WordSpeller                 character-level next-word distribution
IncrementalEMGModel         encoder -> eMG layers -> speller
minimal_pair_accuracy       BLiMP-style evaluation

References
    Yang, C. (2016). The Price of Linguistic Productivity. MIT Press.
    MorPiece: https://github.com/cristianochesi/morpiece

Author: Cristiano Chesi (IUSS Pavia, NeTS Lab).
Development, debugging and optimization assisted by Claude (Anthropic).
Released under the MIT License (see LICENSE).
"""

import gzip
import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_preparation_charwise import (BOS_WORD, EOS_WORD, SPECIAL_WORDS, CharacterVocab,
                                       CharLevelConfig, preprocess_line)
from model_eMG import (HighwayNetwork, LockedDropout, OptimizedEMG, _positions,
                       _scatter_words, _select_words)

logger = logging.getLogger(__name__)

WEIGHTS_FILE = "pytorch_model.pt"
VOCAB_FILE = "vocabulary.json"
TRIES_FILE = "tries.json.gz"
CONFIG_FILE = "config.json"


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class IncrementalMorPieceConfig:
    """
    Hyperparameters of :class:`IncrementalEMGModel`.

    Characters and preprocessing (must match the training data)
        char_vocab_size, char_embedding_dim, max_word_length, lowercase,
        split_punctuation.
    Character encoder
        char_cnn_filters, char_cnn_kernels: filters per kernel width.
        use_highway: highway layer on the word vector.
        attention_heads: heads of the segment attention (must divide the
            sum of char_cnn_filters).
    Tolerance Principle and vocabulary (see ToleranceTrie)
        tp_cutoff: minimum mother-node count before the TP is evaluated.
        min_morpheme_freq: minimum count for a piece to enter the vocabulary.
        min_suffix_stems: distinct stems an ending needs to enter it.
        max_vocab_size: size of the morpheme embedding table.
        morpheme_embedding_dim, max_morphemes_per_word: morpheme route.
    eMG core
        hidden_dim, num_layers, dropout, use_layer_norm, move_gate_input,
        use_gradient_checkpointing, checkpoint_chunk (see model_eMG.EMGConfig).
    Word speller
        speller_hidden_dim, speller_embedding_dim.
    Losses
        tp_loss_weight: weight of the TP boundary supervision.
    """
    char_vocab_size: int = 256
    char_embedding_dim: int = 32
    max_word_length: int = 50
    lowercase: bool = True
    split_punctuation: bool = True

    char_cnn_filters: Tuple[int, ...] = (64, 128, 128, 192)
    char_cnn_kernels: Tuple[int, ...] = (1, 2, 3, 4)
    use_highway: bool = True
    attention_heads: int = 4

    tp_cutoff: int = 100
    min_morpheme_freq: int = 10
    min_suffix_stems: int = 3
    max_vocab_size: int = 50000
    morpheme_embedding_dim: int = 128
    max_morphemes_per_word: int = 8

    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.1
    use_layer_norm: bool = False
    move_gate_input: bool = False
    use_gradient_checkpointing: bool = False
    checkpoint_chunk: int = 32

    speller_hidden_dim: int = 256
    speller_embedding_dim: int = 64

    tp_loss_weight: float = 0.1

    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = BOS_WORD
    eos_token: str = EOS_WORD
    morpheme_boundary: str = "##"

    def __post_init__(self):
        self.char_cnn_filters = tuple(int(f) for f in self.char_cnn_filters)
        self.char_cnn_kernels = tuple(int(k) for k in self.char_cnn_kernels)
        if len(self.char_cnn_filters) != len(self.char_cnn_kernels):
            raise ValueError("char_cnn_filters and char_cnn_kernels must have the same length")
        if sum(self.char_cnn_filters) % self.attention_heads:
            raise ValueError("attention_heads must divide the sum of char_cnn_filters")

    def data_config(self) -> CharLevelConfig:
        """Preprocessing settings matching this model (used for scoring raw text)."""
        return CharLevelConfig(max_word_length=self.max_word_length,
                               char_vocab_size=self.char_vocab_size,
                               lowercase=self.lowercase,
                               split_punctuation=self.split_punctuation,
                               use_cache=False)


# ============================================================================
# Tolerance Principle tries and vocabulary
# ============================================================================

class ToleranceTrie:
    """
    Morpheme discovery with two tries, after MorPiece.

    Every observed word is added to a root trie (left to right) and an
    inflection trie (right to left); each node counts the words that pass
    through it (token counts, the setting MorPiece recommends for Italian).
    Writing f(x) for the count of the node reached by string x, a boundary is
    placed between the stem w[:s] and the ending w[s:] when the Tolerance
    Principle holds on both sides, as in MorPiece's boundary discovery:

        root side:        m = f(w[:s-1]),  d = f(w[:s])
        inflection side:  m = f(w[s+1:]),  d = f(w[s:])
        TP(m, d)  iff  m > cutoff,  d != m  and  d > m / ln(m)

    i.e. both pieces are sufficiently attested continuations of their
    shorter mother strings. Stems have at least two characters; endings too,
    because the empty string carries no count.

    Every boundary found in an observed word counts the stem w[:s] and the
    ending w[s:]. :meth:`update_vocabulary` adds to the token vocabulary,
    most frequent first and up to max_vocab_size: words, stems and endings
    ("##" + ending) seen at least min_morpheme_freq times, endings only if
    they attach to at least min_suffix_stems different stems, plus frequent
    single characters as fallback pieces. Token IDs never change once given.
    """

    def __init__(self, config: IncrementalMorPieceConfig):
        self.config = config
        self.cutoff = config.tp_cutoff
        self.min_freq = config.min_morpheme_freq
        self.min_suffix_stems = config.min_suffix_stems
        self.roots: Dict = {}
        self.infls: Dict = {}
        self.word_counts: Dict[str, int] = {}
        self.stem_counts: Dict[str, int] = {}
        self.ending_counts: Dict[str, int] = {}
        self.ending_stems: Dict[str, object] = {}     # ending -> set of stems, or True once enough
        self.char_counts: Dict[str, int] = {}
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.total_words = 0
        self.total_splits = 0
        self.version = 0
        for token in (config.pad_token, config.unk_token, config.bos_token, config.eos_token):
            self._add_token(token)

    # ------------------------------------------------------------ counting

    def _add_token(self, token: str) -> None:
        idx = len(self.token_to_id)
        self.token_to_id[token] = idx
        self.id_to_token[idx] = token

    @staticmethod
    def _insert(word: str, trie: Dict) -> None:
        node = trie
        for ch in word:
            child = node.get(ch)
            if child is None:
                child = node[ch] = {"##": 0}
            child["##"] += 1
            node = child

    @staticmethod
    def _path_counts(word: str, trie: Dict) -> List[int]:
        """counts[k] = f(word[:k]) for k = 1..len(word); counts[0] = 0."""
        counts = [0] * (len(word) + 1)
        node = trie
        for k, ch in enumerate(word, start=1):
            node = node.get(ch)
            if node is None:
                break
            counts[k] = node["##"]
        return counts

    def _tp(self, mother: int, daughter: int) -> bool:
        return mother > self.cutoff and daughter != mother and daughter > mother / math.log(mother)

    def boundary_labels(self, word: str) -> List[Tuple[int, int]]:
        """
        (s, label) for every candidate boundary s (2 <= s <= len - 1):
        1 = TP boundary, 0 = no boundary, -1 = not enough evidence on one side.
        Reads the current counts without changing them.
        """
        n = len(word)
        if n < 3 or word in SPECIAL_WORDS:
            return []
        rc = self._path_counts(word, self.roots)
        ic = self._path_counts(word[::-1], self.infls)
        labels = []
        for s in range(2, n):
            root_m, root_d = rc[s - 1], rc[s]
            infl_m, infl_d = ic[n - s - 1], ic[n - s]
            if root_m <= self.cutoff or infl_m <= self.cutoff:
                labels.append((s, -1))
            else:
                labels.append((s, int(self._tp(root_m, root_d) and self._tp(infl_m, infl_d))))
        return labels

    def boundaries(self, word: str) -> List[int]:
        """Positions s of the TP boundaries of ``word`` under the current counts."""
        return [s for s, label in self.boundary_labels(word) if label == 1]

    def process_word(self, word: str) -> List[int]:
        """Count one occurrence of ``word``; return the TP boundaries found in it."""
        if not word or word in SPECIAL_WORDS:
            return []
        self.total_words += 1
        self.word_counts[word] = self.word_counts.get(word, 0) + 1
        for ch in word:
            self.char_counts[ch] = self.char_counts.get(ch, 0) + 1
        self._insert(word, self.roots)
        self._insert(word[::-1], self.infls)
        found = self.boundaries(word)
        for s in found:
            stem, ending = word[:s], word[s:]
            self.stem_counts[stem] = self.stem_counts.get(stem, 0) + 1
            self.ending_counts[ending] = self.ending_counts.get(ending, 0) + 1
            stems = self.ending_stems.setdefault(ending, set())
            if stems is not True:
                stems.add(stem)
                if len(stems) >= self.min_suffix_stems:
                    self.ending_stems[ending] = True
        self.total_splits += len(found)
        return found

    def process_text(self, text: str) -> None:
        for word in text.split():
            self.process_word(word)

    def update_vocabulary(self) -> int:
        """Promote frequent pieces into the vocabulary; returns the number of tokens added."""
        boundary = self.config.morpheme_boundary
        candidates: Dict[str, int] = {}

        def offer(token: str, count: int) -> None:
            if count >= self.min_freq and token not in self.token_to_id:
                candidates[token] = max(count, candidates.get(token, 0))

        for word, count in self.word_counts.items():
            offer(word, count)
        for stem, count in self.stem_counts.items():
            offer(stem, count)
        for ending, count in self.ending_counts.items():
            if self.ending_stems.get(ending) is True:
                offer(boundary + ending, count)
        for ch, count in self.char_counts.items():
            offer(ch, count)
            offer(boundary + ch, count)

        added = 0
        for token, _ in sorted(candidates.items(), key=lambda kv: (-kv[1], kv[0])):
            if len(self.token_to_id) >= self.config.max_vocab_size:
                break
            self._add_token(token)
            added += 1
        self.version += 1
        return added

    # ---------------------------------------------------------- encoding

    def segment_word(self, word: str) -> List[str]:
        """
        Greedy longest-match segmentation into vocabulary pieces: the first
        piece is a word, stem or character, the following ones are "##"
        endings; characters outside the vocabulary become <unk>.
        """
        if not word:
            return [self.config.unk_token]
        if word in self.token_to_id:
            return [word]
        boundary = self.config.morpheme_boundary
        pieces, pos, n = [], 0, len(word)
        while pos < n:
            prefix = "" if pos == 0 else boundary
            for end in range(n, pos, -1):
                piece = prefix + word[pos:end]
                if piece in self.token_to_id:
                    pieces.append(piece)
                    pos = end
                    break
            else:
                pieces.append(self.config.unk_token)
                pos += 1
        return pieces

    def encode(self, word: str) -> List[int]:
        return [self.token_to_id[p] if p in self.token_to_id else self.token_to_id[self.config.unk_token]
                for p in self.segment_word(word)]

    def decode(self, ids: Sequence[int]) -> str:
        tokens = [self.id_to_token.get(int(i), self.config.unk_token) for i in ids]
        return "".join(t[len(self.config.morpheme_boundary):] if t.startswith(self.config.morpheme_boundary)
                       else t for t in tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    # ------------------------------------------------------------ storage

    def save(self, path: str) -> None:
        """Vocabulary, counts and settings (JSON)."""
        state = {
            "tokens": [self.id_to_token[i] for i in range(len(self.id_to_token))],
            "word_counts": self.word_counts,
            "stem_counts": self.stem_counts,
            "ending_counts": self.ending_counts,
            "ending_stems": {k: (True if v is True else sorted(v)) for k, v in self.ending_stems.items()},
            "char_counts": self.char_counts,
            "total_words": self.total_words,
            "total_splits": self.total_splits,
            "version": self.version,
            "settings": {"tp_cutoff": self.cutoff, "min_morpheme_freq": self.min_freq,
                         "min_suffix_stems": self.min_suffix_stems,
                         "max_vocab_size": self.config.max_vocab_size},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)

    def save_tries(self, path: str) -> None:
        """The two tries (gzip-compressed JSON), needed to continue counting."""
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump({"roots": self.roots, "infls": self.infls}, f, ensure_ascii=False)

    def load(self, path: str, tries_path: Optional[str] = None) -> None:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        self.token_to_id = {tok: i for i, tok in enumerate(state["tokens"])}
        self.id_to_token = dict(enumerate(state["tokens"]))
        self.word_counts = state["word_counts"]
        self.stem_counts = state["stem_counts"]
        self.ending_counts = state["ending_counts"]
        self.ending_stems = {k: (True if v is True else set(v)) for k, v in state["ending_stems"].items()}
        self.char_counts = state["char_counts"]
        self.total_words = state["total_words"]
        self.total_splits = state["total_splits"]
        self.version = state["version"] + 1          # invalidate cached segmentations
        if tries_path is not None and os.path.exists(tries_path):
            with gzip.open(tries_path, "rt", encoding="utf-8") as f:
                tries = json.load(f)
            self.roots, self.infls = tries["roots"], tries["infls"]


class MorphemeIndexer:
    """
    Morpheme IDs and TP boundary targets for the word types of a dataset,
    recomputed lazily for the types of each batch whenever the trie changes.

    morph_ids[t]      (max_morphemes_per_word,) vocabulary IDs, 0 = padding
    split_targets[t]  (max_word_length,) 1 boundary after character j,
                      0 no boundary, -1 undecided or not a candidate
    """

    def __init__(self, trie: ToleranceTrie, types: Sequence[str], max_morphemes: int,
                 max_word_length: int):
        self.trie = trie
        self.types = types
        self.morph_ids = np.zeros((len(types), max_morphemes), dtype=np.int64)
        self.split_targets = np.full((len(types), max_word_length), -1, dtype=np.int8)
        self.version = np.full(len(types), -1, dtype=np.int64)

    def _compute(self, t: int) -> None:
        self.version[t] = self.trie.version
        self.morph_ids[t] = 0
        self.split_targets[t] = -1
        word = self.types[t]
        if not word:
            return
        ids = self.trie.encode(word)[:self.morph_ids.shape[1]]
        self.morph_ids[t, :len(ids)] = ids
        width = self.split_targets.shape[1]
        for s, label in self.trie.boundary_labels(word):
            if s - 1 < width:
                self.split_targets[t, s - 1] = label

    def lookup(self, type_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        unique = np.unique(type_ids)
        for t in unique[self.version[unique] != self.trie.version]:
            self._compute(int(t))
        return self.morph_ids[type_ids], self.split_targets[type_ids]


def prepare_batch(batch: Dict[str, torch.Tensor], indexer: MorphemeIndexer,
                  device: torch.device) -> Dict[str, torch.Tensor]:
    """Batch from CharBatchCollator -> keyword arguments of IncrementalEMGModel.forward."""
    morph, targets = indexer.lookup(batch["input_types"].numpy())
    width = batch["input_chars"].size(2)
    non_blocking = device.type == "cuda"
    inputs = {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()
              if k not in ("input_types", "target_types")}
    inputs["morph_ids"] = torch.from_numpy(morph).to(device, non_blocking=non_blocking)
    inputs["split_targets"] = torch.from_numpy(
        np.ascontiguousarray(targets[:, :, :width])).to(device, non_blocking=non_blocking)
    return inputs


# ============================================================================
# Character encoder
# ============================================================================

class IncrementalMorPieceCNN(nn.Module):
    """
    Character encoder with a morpheme-boundary predictor and a morpheme route.

    1. Characters are embedded with their positions from the start and from
       the end of the word (padding has its own zero position embedding).
    2. Convolutions of several widths give per-character features.
    3. The boundary predictor gives P(boundary after character j), allowed
       for 1 <= j <= length - 2.
    4. Segment-aware pooling and self-attention (padding masked) let
       characters combine mostly within their own soft segment; max-pooling
       over real characters gives the character word vector.
    5. The word's vocabulary pieces (from the ToleranceTrie) are embedded,
       averaged, and added to the word vector through a learned gate.

    Only non-empty words are encoded; empty slots get a zero vector.
    """

    def __init__(self, config: IncrementalMorPieceConfig):
        super().__init__()
        self.config = config
        self.trie = ToleranceTrie(config)
        E, L, H = config.char_embedding_dim, config.max_word_length, config.hidden_dim
        self.split_temperature = 0.5
        self.segment_sharpness = 3.0

        self.char_embedding = nn.Embedding(config.char_vocab_size, E, padding_idx=0)
        self.pos_from_start = nn.Embedding(L + 1, E, padding_idx=L)
        self.pos_from_end = nn.Embedding(L + 1, E, padding_idx=L)
        self.convs = nn.ModuleList([nn.Conv1d(E, f, k) for f, k in
                                    zip(config.char_cnn_filters, config.char_cnn_kernels)])
        self.cnn_output_dim = sum(config.char_cnn_filters)
        self.split_predictor = nn.Sequential(
            nn.Linear(self.cnn_output_dim, H // 2), nn.LayerNorm(H // 2), nn.GELU(), nn.Linear(H // 2, 1)
        )
        self.segment_attention = nn.MultiheadAttention(self.cnn_output_dim, config.attention_heads,
                                                       dropout=config.dropout, batch_first=True)
        self.output_projection = nn.Linear(self.cnn_output_dim, H)
        self.highway = HighwayNetwork(H) if config.use_highway else None
        self.layer_norm = nn.LayerNorm(H)
        self.dropout = nn.Dropout(config.dropout)

        self.morpheme_embedding = nn.Embedding(config.max_vocab_size, config.morpheme_embedding_dim,
                                               padding_idx=0)
        self.morpheme_projection = nn.Linear(config.morpheme_embedding_dim, H)
        self.morpheme_gate = nn.Linear(2 * H, H)
        self.output_dim = H
        self.reset_parameters()

    def reset_parameters(self):
        L = self.config.max_word_length
        for emb in (self.char_embedding, self.pos_from_start, self.pos_from_end, self.morpheme_embedding):
            nn.init.normal_(emb.weight, std=0.02)
        with torch.no_grad():
            self.char_embedding.weight[0].zero_()
            self.pos_from_start.weight[L].zero_()
            self.pos_from_end.weight[L].zero_()
            self.morpheme_embedding.weight[0].zero_()
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(conv.bias)
        for layer in self.split_predictor:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        nn.init.zeros_(self.split_predictor[-1].bias)
        nn.init.xavier_uniform_(self.split_predictor[-1].weight, gain=0.1)
        self.segment_attention._reset_parameters()
        for linear in (self.output_projection, self.morpheme_projection, self.morpheme_gate):
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
        nn.init.constant_(self.morpheme_gate.bias, -1.0)   # character route dominates at the start
        if self.highway is not None:
            self.highway.reset_parameters()
        self.layer_norm.reset_parameters()

    def forward(self, char_ids: torch.Tensor, word_lengths: torch.Tensor, morph_ids: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        char_ids (B, T, W), word_lengths (B, T), morph_ids (B, T, K).
        Returns embeddings (B, T, hidden_dim), boundary logits (B, T, W)
        (sigmoid gives the boundary probability) and the boundary mask (B, T, W).
        """
        batch, seq_len, width_in = char_ids.shape
        total = batch * seq_len
        words, lengths, index = _select_words(char_ids, word_lengths, 1)
        width = words.size(1)
        start, end, inside = _positions(lengths, width, self.config.max_word_length)
        x = (self.char_embedding(words) + self.pos_from_start(start) + self.pos_from_end(end)).transpose(1, 2)
        feats = []
        for conv in self.convs:
            k = conv.kernel_size[0]
            feats.append(F.relu(conv(F.pad(x, ((k - 1) // 2, k // 2)))))
        feats = torch.cat(feats, dim=1).transpose(1, 2)                      # (N, W, C)

        pos = torch.arange(width, device=words.device).unsqueeze(0)
        mask = (pos >= 1) & (pos <= lengths.unsqueeze(1) - 2)
        logits = self.split_predictor(feats).squeeze(-1) / self.split_temperature
        probs = torch.sigmoid(logits) * mask

        segment = torch.cumsum(probs, dim=1) - probs
        attention = torch.exp(-self.segment_sharpness * (segment.unsqueeze(2) - segment.unsqueeze(1)).abs())
        attention = attention * (inside.unsqueeze(1) & inside.unsqueeze(2))
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
        pooled = torch.bmm(attention, feats)
        attended, _ = self.segment_attention(pooled, pooled, pooled, key_padding_mask=~inside,
                                             need_weights=False)
        attended = attended.masked_fill(~inside.unsqueeze(-1), float("-inf"))
        char_vec = self.output_projection(attended.max(dim=1).values)
        if self.highway is not None:
            char_vec = self.highway(char_vec)
        char_vec = self.dropout(self.layer_norm(char_vec))

        morph = morph_ids.reshape(total, -1).index_select(0, index)
        present = (morph != 0).unsqueeze(-1).to(char_vec.dtype)
        morph_mean = (self.morpheme_embedding(morph) * present).sum(1) / present.sum(1).clamp_min(1)
        morph_vec = self.morpheme_projection(morph_mean)
        gate = torch.sigmoid(self.morpheme_gate(torch.cat([char_vec, morph_vec], dim=-1)))
        out = char_vec + gate * morph_vec

        out = _scatter_words(out, index, total).view(batch, seq_len, -1)
        logits = F.pad(_scatter_words(logits, index, total), (0, width_in - width)).view(batch, seq_len, -1)
        mask = F.pad(_scatter_words(mask.to(logits.dtype), index, total), (0, width_in - width))
        return out, logits, mask.view(batch, seq_len, -1) > 0.5

    def update_vocabulary_from_text(self, text: str) -> int:
        """Count the words of ``text`` (already preprocessed) and grow the vocabulary."""
        self.trie.process_text(text)
        return self.trie.update_vocabulary()

    def get_vocab_size(self) -> int:
        return self.trie.vocab_size

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_vocabulary(self, path: str, tries_path: Optional[str] = None) -> None:
        self.trie.save(path)
        if tries_path is not None:
            self.trie.save_tries(tries_path)

    def load_vocabulary(self, path: str, tries_path: Optional[str] = None) -> None:
        self.trie.load(path, tries_path)


# ============================================================================
# Word speller
# ============================================================================

class WordSpeller(nn.Module):
    """
    Next-word distribution spelled character by character:

        log P(w | h) = sum_k log P(c_k | c_<k, h) + log P(<eow> | w, h)

    A GRU initialised from the context vector h, and fed h at every step,
    reads <bow> c_1 ... c_n and predicts c_1 ... c_n <eow>. The distribution
    covers every possible word, so any sentence gets a probability.
    """

    def __init__(self, config: IncrementalMorPieceConfig):
        super().__init__()
        E, Hs, H = config.speller_embedding_dim, config.speller_hidden_dim, config.hidden_dim
        self.char_embedding = nn.Embedding(config.char_vocab_size, E, padding_idx=0)
        self.context_projection = nn.Linear(H, E)
        self.init_state = nn.Linear(H, Hs)
        self.gru = nn.GRU(2 * E, Hs, batch_first=True)
        self.output = nn.Linear(Hs, config.char_vocab_size)

    def forward(self, context: torch.Tensor, target_chars: torch.Tensor, target_lengths: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        context (N, H), target_chars (N, W), target_lengths (N,) of non-empty words.
        Returns (log P(word) (N,), number of predicted symbols per word (N,)).
        """
        n, width = target_chars.shape
        bow = torch.full((n, 1), CharacterVocab.BOW, dtype=target_chars.dtype, device=target_chars.device)
        inputs = torch.cat([bow, target_chars], dim=1)
        targets = F.pad(target_chars, (0, 1)).scatter(1, target_lengths.unsqueeze(1), CharacterVocab.EOW)
        valid = torch.arange(width + 1, device=target_chars.device).unsqueeze(0) <= target_lengths.unsqueeze(1)
        ctx = self.context_projection(context).unsqueeze(1).expand(-1, width + 1, -1)
        h0 = torch.tanh(self.init_state(context)).unsqueeze(0)
        states, _ = self.gru(torch.cat([self.char_embedding(inputs), ctx], dim=-1), h0)
        logp = torch.log_softmax(self.output(states).float(), dim=-1)
        logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1) * valid
        return logp.sum(1), valid.sum(1)


# ============================================================================
# Language model
# ============================================================================

class IncrementalOutput(NamedTuple):
    """Output of :class:`IncrementalEMGModel`; sums and counts are tensors (no host sync)."""
    loss: Optional[torch.Tensor]            # lm_loss + tp_loss_weight * tp_loss
    lm_loss: Optional[torch.Tensor]         # NLL per predicted symbol (characters + <eow>)
    tp_loss: Optional[torch.Tensor]         # boundary supervision (binary cross-entropy)
    nll_sum: Optional[torch.Tensor]         # summed NLL of the target words
    n_symbols: Optional[torch.Tensor]       # predicted symbols
    n_words: Optional[torch.Tensor]         # predicted words
    hidden: list                            # eMG state after the last position
    word_logprobs: Optional[torch.Tensor]   # (B, T) log P(target word), 0 for padding


class IncrementalEMGModel(nn.Module):
    """
    Tokenizer-free eMG-RNN: IncrementalMorPieceCNN -> eMG layers -> WordSpeller.

    The eMG layers are the same recurrent core as model_eMG.EMGModel. The
    vocabulary of the encoder's ToleranceTrie grows during training (see
    train_eMG_incremental.py); the morpheme embedding table has a fixed size
    (max_vocab_size), so parameters never change shape.
    """

    def __init__(self, config: IncrementalMorPieceConfig):
        super().__init__()
        self.config = config
        self.char_vocab = CharacterVocab(config.data_config())
        self.encoder = IncrementalMorPieceCNN(config)
        self.embedding_dropout = LockedDropout(config.dropout)
        self.emg = OptimizedEMG(config.hidden_dim, config.hidden_dim, config.num_layers,
                                dropout=config.dropout, use_layer_norm=config.use_layer_norm,
                                move_gate_input=config.move_gate_input, locked_dropout=True,
                                use_gradient_checkpointing=config.use_gradient_checkpointing,
                                checkpoint_chunk=config.checkpoint_chunk)
        self.output_dropout = LockedDropout(config.dropout)
        self.speller = WordSpeller(config)

    def forward(self, input_chars: torch.Tensor, input_lengths: torch.Tensor, morph_ids: torch.Tensor,
                target_chars: Optional[torch.Tensor] = None, target_lengths: Optional[torch.Tensor] = None,
                split_targets: Optional[torch.Tensor] = None, hidden: Optional[list] = None,
                return_word_logprobs: bool = False) -> IncrementalOutput:
        """
        input_chars (B, T, Wi), input_lengths (B, T), morph_ids (B, T, K): input words.
        target_chars (B, T, Wt), target_lengths (B, T): the word that follows each input word.
        split_targets (B, T, Wi): TP boundary labels for the input words (-1 = ignore).
        """
        x, split_logits, split_mask = self.encoder(input_chars, input_lengths, morph_ids)
        out, new_hidden = self.emg(self.embedding_dropout(x), hidden)
        out = self.output_dropout(out)

        lm_loss = nll_sum = n_symbols = n_words = word_lp = None
        if target_chars is not None:
            batch, seq_len, _ = out.shape
            words, lengths, index = _select_words(target_chars, target_lengths, 1)
            context = out.reshape(batch * seq_len, -1).index_select(0, index)
            logp, symbols = self.speller(context, words, lengths)
            nll_sum = -logp.sum()
            n_symbols = symbols.sum()
            n_words = torch.tensor(logp.numel(), device=out.device)
            lm_loss = nll_sum / n_symbols.clamp_min(1)
            if return_word_logprobs:
                word_lp = _scatter_words(logp, index, batch * seq_len).view(batch, seq_len)

        tp_loss = None
        if split_targets is not None:
            usable = split_mask & (split_targets >= 0)
            bce = F.binary_cross_entropy_with_logits(split_logits.float(), split_targets.clamp_min(0).float(),
                                                     reduction="none")
            tp_loss = (bce * usable).sum() / usable.sum().clamp_min(1)

        loss = None
        if lm_loss is not None:
            loss = lm_loss
            if tp_loss is not None and self.config.tp_loss_weight > 0:
                loss = loss + self.config.tp_loss_weight * tp_loss
        return IncrementalOutput(loss, lm_loss, tp_loss, nll_sum, n_symbols, n_words, new_hidden, word_lp)

    # ------------------------------------------------------------- scoring

    def encode_sequences(self, sequences: List[List[str]], device) -> Dict[str, torch.Tensor]:
        """
        Model inputs for already preprocessed word sequences (sentence markers
        included): word i is the input that predicts word i + 1, so the result
        has one position fewer than the longest sequence. Pass the dict to
        forward(..., return_word_logprobs=True) to get log P(word i + 1).
        """
        T = max(len(s) for s in sequences) - 1
        L, K = self.config.max_word_length, self.config.max_morphemes_per_word
        n = len(sequences)
        chars = np.zeros((n, T + 1, L), dtype=np.int64)
        lengths = np.zeros((n, T + 1), dtype=np.int64)
        morph = np.zeros((n, T + 1, K), dtype=np.int64)
        for i, words in enumerate(sequences):
            for j, word in enumerate(words):
                ids = self.char_vocab.encode_word(word)
                chars[i, j, :len(ids)] = ids
                lengths[i, j] = len(ids)
                pieces = self.encoder.trie.encode(word)[:K]
                morph[i, j, :len(pieces)] = pieces
        t = lambda a: torch.from_numpy(a).to(device)
        return {"input_chars": t(chars[:, :-1]), "input_lengths": t(lengths[:, :-1]),
                "morph_ids": t(morph[:, :-1]), "target_chars": t(chars[:, 1:]),
                "target_lengths": t(lengths[:, 1:])}

    @torch.no_grad()
    def score_sentences(self, sentences: Sequence[str], batch_size: int = 64,
                        include_eos: bool = False) -> List[float]:
        """
        log P(sentence) = sum of log P(w_i | <s> w_1 ... w_{i-1}) over its words,
        with the same preprocessing as the training data. With include_eos the
        probability of the end marker </s> is added.
        """
        device = next(self.parameters()).device
        data_config = self.config.data_config()
        was_training = self.training
        self.eval()
        scores: List[float] = []
        for start in range(0, len(sentences), batch_size):
            sequences = []
            for sentence in sentences[start:start + batch_size]:
                words = [BOS_WORD] + preprocess_line(sentence, data_config)
                sequences.append(words + [EOS_WORD] if include_eos else words)
            if max(len(s) for s in sequences) < 2:
                scores.extend(0.0 for _ in sequences)
                continue
            out = self(**self.encode_sequences(sequences, device), return_word_logprobs=True)
            scores.extend(out.word_logprobs.sum(1).tolist())
        self.train(was_training)
        return scores

    # ------------------------------------------------------------- storage

    def save_pretrained(self, directory: str, extra_config: Optional[dict] = None) -> None:
        """Write weights, vocabulary, tries and configuration to ``directory``."""
        os.makedirs(directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(directory, WEIGHTS_FILE))
        self.encoder.save_vocabulary(os.path.join(directory, VOCAB_FILE),
                                     os.path.join(directory, TRIES_FILE))
        config = {"model_type": "emg_incremental", "model_config": asdict(self.config),
                  "vocab_size": self.encoder.get_vocab_size()}
        if extra_config:
            config.update(extra_config)
        with open(os.path.join(directory, CONFIG_FILE), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, directory: str, map_location="cpu", load_tries: bool = False
                        ) -> "IncrementalEMGModel":
        """Load a model saved by :meth:`save_pretrained` or by the training script."""
        with open(os.path.join(directory, CONFIG_FILE), encoding="utf-8") as f:
            config = IncrementalMorPieceConfig(**json.load(f)["model_config"])
        model = cls(config)
        tries = os.path.join(directory, TRIES_FILE) if load_tries else None
        model.encoder.load_vocabulary(os.path.join(directory, VOCAB_FILE), tries)
        state = torch.load(os.path.join(directory, WEIGHTS_FILE), map_location=map_location)
        model.load_state_dict(state)
        return model

    def get_vocab_size(self) -> int:
        return self.encoder.get_vocab_size()

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def minimal_pair_accuracy(model: IncrementalEMGModel, pairs: Sequence[Tuple[str, str]],
                          batch_size: int = 64, include_eos: bool = False) -> Dict[str, object]:
    """
    BLiMP-style evaluation: the model is right on a pair (good, bad) when it
    gives the grammatical sentence the higher log-probability.
    Returns {"accuracy", "n", "margins"} with margins = log P(good) - log P(bad).
    """
    good = model.score_sentences([g for g, _ in pairs], batch_size, include_eos)
    bad = model.score_sentences([b for _, b in pairs], batch_size, include_eos)
    margins = [g - b for g, b in zip(good, bad)]
    accuracy = sum(m > 0 for m in margins) / max(len(margins), 1)
    return {"accuracy": accuracy, "n": len(margins), "margins": margins}


# ============================================================================
# Self-test
# ============================================================================

if __name__ == "__main__":
    import tempfile

    torch.manual_seed(0)
    config = IncrementalMorPieceConfig(
        char_embedding_dim=16, char_cnn_filters=(16, 32, 32, 48), hidden_dim=64, max_word_length=20,
        tp_cutoff=50, min_morpheme_freq=5, max_vocab_size=500, morpheme_embedding_dim=16,
        speller_hidden_dim=64, speller_embedding_dim=16,
    )
    model = IncrementalEMGModel(config)
    print(f"Parameters: {model.get_num_params():,}; initial vocabulary: {model.get_vocab_size()}")

    text = """
    arrivano arrivavano arrivarono arriviamo arrivate arrivai arrivasti
    mangiamo mangiate mangiano mangiavamo mangiavate mangiavano
    parliamo parlate parlano parlavamo parlavate parlavano
    gatto gatti gatta gatte cane cani casa case
    bellissimo bellissima bellissimi bellissime
    """
    added = model.encoder.update_vocabulary_from_text(text * 100)
    trie = model.encoder.trie
    print(f"Vocabulary after 100 exposures: +{added} tokens, {trie.total_splits} boundary events")
    top = sorted(((c, e) for e, c in trie.ending_counts.items() if trie.ending_stems.get(e) is True),
                 reverse=True)[:8]
    print("Most frequent endings:", ["-" + e for _, e in top])
    for word in ("arrivavano", "mangiavate", "parlavano", "bellissimi"):
        cut = [word[:s] + "|" + word[s:] for s in trie.boundaries(word)]
        print(f"  {word:<12} TP boundaries {cut}  segmentation {trie.segment_word(word)}")

    sentences = ["arrivano domani", "mangiavano spesso insieme", "il gatto dorme"]
    sequences = [[BOS_WORD] + s.split() + [EOS_WORD] for s in sentences]
    types = sorted({w for s in sequences for w in s})
    indexer = MorphemeIndexer(trie, [""] + types, config.max_morphemes_per_word, config.max_word_length)
    batch = model.encode_sequences(sequences, torch.device("cpu"))
    type_index = {w: i + 1 for i, w in enumerate(types)}
    T = batch["input_chars"].size(1)
    input_types = np.zeros((len(sequences), T), dtype=np.int64)
    for i, s in enumerate(sequences):
        for j, w in enumerate(s[:-1]):
            input_types[i, j] = type_index[w]
    _, split_targets = indexer.lookup(input_types)
    batch["split_targets"] = torch.from_numpy(split_targets[:, :, :batch["input_chars"].size(2)].copy())
    out = model(**batch)
    out.loss.backward()
    unused = [n for n, p in model.named_parameters() if p.grad is None]
    assert not unused, f"parameters without gradient: {unused}"
    print(f"LM loss {out.lm_loss.item():.3f} per symbol, TP loss {out.tp_loss.item():.3f}; "
          "every parameter receives a gradient")

    result = minimal_pair_accuracy(model, [("il gatto dorme", "il gatto dormono")])
    print(f"Minimal-pair scoring works (margin {result['margins'][0]:+.2f} nats, untrained model)")

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        reloaded = IncrementalEMGModel.from_pretrained(tmp)
    assert reloaded.get_vocab_size() == model.get_vocab_size()
    a = model.score_sentences(sentences)
    b = reloaded.score_sentences(sentences)
    assert np.allclose(a, b, atol=1e-4), "reloaded model differs"
    print("Save/load round trip gives identical scores.")
    print("All self-tests passed.")
