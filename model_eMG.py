#!/usr/bin/env python3
"""
eMG-RNN: expectation-based Minimalist Grammar Recurrent Neural Network
======================================================================

A recurrent language model whose cell implements two structure-building
operations of expectation-based Minimalist Grammars (Chesi 2021):

- Merge: the current input, gated by the previous output state, is merged
  into a bounded context memory.
- Move: a gate decides how much of the context memory is moved into the
  output, as opposed to passing the current input through.

The architecture was introduced in Chesi et al. (2024), "Different Ways to
Forget: Linguistic Gates in Recurrent Neural Networks", 2nd BabyLM Challenge
(https://aclanthology.org/2024.conll-babylm.9/).

Module contents
---------------
EMGConfig                       hyperparameters
OptimizedEMGCell                one eMG layer (single-step and sequence use)
OptimizedEMG                    stacked eMG layers, layer-major fast loop
CharCNN                         character CNN word encoder
TolerancePrincipleCNN           lexical + compositional dual-route encoder
NeuralMorPiece                  differentiable MorPiece-style segmenter
ToleranceRegularizer            type-level exception-rate regulariser
split_rate_loss                 morpheme-boundary rate regulariser
EMGModel                        the language model
build_char_lookup               token-to-character table for char encoders
EMGHFConfig, EMGLanguageModel   Hugging Face wrapper

Author: Cristiano Chesi (IUSS Pavia, NeTS Lab).
Development, debugging and optimization assisted by Claude (Anthropic).
Released under the MIT License (see LICENSE).
"""

__version__ = "1.0.0"

import logging
import math
from dataclasses import dataclass, fields
from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.checkpoint import checkpoint

import transformers
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger(__name__)

Hidden = List[Tuple[torch.Tensor, torch.Tensor]]


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class EMGConfig:
    """
    Hyperparameters of :class:`EMGModel`.

    Core
        vocab_size, embedding_dim, hidden_dim, num_layers: model size.
        dropout: dropout on embeddings, between layers and on the final
            layer output. It is never applied inside the recurrence.
        locked_dropout: one dropout mask per sequence (variational dropout)
            instead of one per position.
        use_layer_norm: layer-normalise the memory pre-activation
            (c_{t-1} + merge_t) before the tanh.
        move_gate_input: let the current input drive the Move gate together
            with the previous state. False gives the cell of Chesi et al.
            (2024), where the gates depend on the previous state only.
        use_gradient_checkpointing: recompute activations in the backward
            pass, in chunks of ``checkpoint_chunk`` time steps.
        tie_word_embeddings: share the input embedding and output matrix
            (word embeddings only, and only if embedding_dim == hidden_dim).
        debug_checks: check token and target IDs on every forward pass. This
            forces a GPU-CPU synchronisation, so it is off by default.

    Character encoders (at most one; priority NeuralMorPiece >
    TolerancePrincipleCNN > CharCNN)
        use_char_cnn, use_tolerance_cnn, use_neural_morpiece: select one.
        char_vocab_size, char_embedding_dim: character embedding table.
        char_cnn_filters, char_cnn_kernels: filters per kernel width.
        max_word_length: characters kept per token.
        char_cnn_highway: highway layer after the CharCNN.
        morpiece_hidden_dim: GRU size per direction in NeuralMorPiece.
        morpiece_lexical_bypass: add a lexical route to NeuralMorPiece.

    Auxiliary losses
        tolerance_reg_weight: weight of ToleranceRegularizer (tolerance and
            bypass gates) and of split_rate_loss (NeuralMorPiece); 0 disables.
        morpiece_split_rate: target boundary probability per word-internal
            position for NeuralMorPiece.
    """
    vocab_size: int = 50257
    embedding_dim: int = 512
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.1
    locked_dropout: bool = True
    use_layer_norm: bool = False
    move_gate_input: bool = False
    use_gradient_checkpointing: bool = False
    checkpoint_chunk: int = 32
    tie_word_embeddings: bool = True
    debug_checks: bool = False

    use_char_cnn: bool = False
    use_tolerance_cnn: bool = False
    use_neural_morpiece: bool = False
    char_vocab_size: int = 256
    char_embedding_dim: int = 16
    char_cnn_filters: Tuple[int, ...] = (32, 64, 128, 256)
    char_cnn_kernels: Tuple[int, ...] = (1, 2, 3, 4)
    max_word_length: int = 50
    char_cnn_highway: bool = True
    morpiece_hidden_dim: int = 128
    morpiece_lexical_bypass: bool = True

    tolerance_reg_weight: float = 0.01
    morpiece_split_rate: float = 0.25

    def __post_init__(self):
        self.char_cnn_filters = tuple(int(f) for f in self.char_cnn_filters)
        self.char_cnn_kernels = tuple(int(k) for k in self.char_cnn_kernels)

        enabled = [name for name in ("use_neural_morpiece", "use_tolerance_cnn", "use_char_cnn")
                   if getattr(self, name)]
        if len(enabled) > 1:
            logger.warning("Several character encoders enabled; using %s.", enabled[0])
            for name in enabled[1:]:
                setattr(self, name, False)

        if self.uses_characters:
            if len(self.char_cnn_filters) != len(self.char_cnn_kernels):
                raise ValueError("char_cnn_filters and char_cnn_kernels must have the same length")
            if self.tie_word_embeddings:
                logger.info("Weight tying disabled: character encoders have no input "
                            "embedding matrix to tie.")
                self.tie_word_embeddings = False
            if self.use_char_cnn and self.embedding_dim != sum(self.char_cnn_filters):
                logger.info("embedding_dim set to %d (sum of CharCNN filters).",
                            sum(self.char_cnn_filters))
                self.embedding_dim = sum(self.char_cnn_filters)
        elif self.tie_word_embeddings and self.embedding_dim != self.hidden_dim:
            logger.warning("Weight tying disabled: embedding_dim (%d) != hidden_dim (%d).",
                           self.embedding_dim, self.hidden_dim)
            self.tie_word_embeddings = False

    @property
    def uses_characters(self) -> bool:
        return self.use_char_cnn or self.use_tolerance_cnn or self.use_neural_morpiece


# ============================================================================
# Shared helpers
# ============================================================================

class LockedDropout(nn.Module):
    """Variational dropout: one mask per sequence, shared across time steps."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, features)."""
        if not self.training or self.p == 0:
            return x
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p).div_(1 - self.p)
        return x * mask


def _select_words(char_ids: torch.Tensor, word_lengths: torch.Tensor, min_width: int = 1
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flatten (batch, seq_len, L) character IDs, keep only non-empty tokens and
    cut the character axis to the longest of them (at least ``min_width``).

    Returns (words (N, W), lengths (N,), index (N,)) where ``index`` gives the
    position of each kept token in the flattened batch.
    """
    batch, seq_len, width = char_ids.shape
    flat = char_ids.reshape(batch * seq_len, width)
    lengths = word_lengths.reshape(batch * seq_len)
    index = torch.nonzero(lengths > 0, as_tuple=False).squeeze(1)
    words = flat.index_select(0, index)
    lengths = lengths.index_select(0, index).long()
    target = max(int(lengths.max()) if index.numel() else 0, min_width)
    if target <= width:
        words = words[:, :target]
    else:
        words = F.pad(words, (0, target - width))
    return words.long(), lengths, index


def _scatter_words(values: torch.Tensor, index: torch.Tensor, total: int) -> torch.Tensor:
    """Inverse of _select_words: put per-token values back, zeros elsewhere."""
    out = values.new_zeros((total,) + tuple(values.shape[1:]))
    return out.index_copy(0, index, values)


def _positions(lengths: torch.Tensor, width: int, pad_index: int
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Character positions from the start and from the end of each word; padding gets pad_index."""
    pos = torch.arange(width, device=lengths.device).unsqueeze(0).expand(lengths.size(0), -1)
    inside = pos < lengths.unsqueeze(1)
    from_start = torch.where(inside, pos, pad_index)
    from_end = torch.where(inside, lengths.unsqueeze(1) - 1 - pos, pad_index)
    return from_start, from_end, inside


def _masked_max_conv(conv: nn.Conv1d, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """ReLU convolution (no padding) followed by max-pooling over windows that start inside the word."""
    out = F.relu(conv(x))                                            # (N, F, T)
    starts = torch.arange(out.size(2), device=x.device).unsqueeze(0)
    valid = starts < lengths.unsqueeze(1)                            # every word has window 0
    return out.masked_fill(~valid.unsqueeze(1), float("-inf")).max(dim=2).values


class HighwayNetwork(nn.Module):
    """Highway layer, initialised to mostly carry its input through."""

    def __init__(self, dim: int, gate_bias: float = -2.0):
        super().__init__()
        self.gate_bias = gate_bias
        self.transform = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.transform.weight)
        nn.init.zeros_(self.transform.bias)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.constant_(self.gate.bias, self.gate_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(x))
        return g * F.relu(self.transform(x)) + (1 - g) * x


# ============================================================================
# eMG recurrent core
# ============================================================================

class OptimizedEMGCell(nn.Module):
    """
    One eMG layer.

    With input x_t and previous state (m_{t-1}, c_{t-1}):

        x~_t   = W_x x_t + b_x                         input projection
        rho_t  = sigmoid(W_r m_{t-1} + b_r)            retain gate
        mu_t   = sigmoid(W_m m_{t-1} + b_m [+ U x_t])  move gate
        c_t    = tanh(c_{t-1} + rho_t * x~_t)          Merge into the context memory
        m_t    = (1 - mu_t) * x~_t + mu_t * c_t        Move memory into the output

    The U x_t term exists only with ``move_gate_input=True``. With
    ``use_layer_norm=True`` the memory pre-activation is layer-normalised:
    c_t = tanh(LN(c_{t-1} + rho_t * x~_t)). Both gates lie in (0, 1), so m_t
    interpolates between the current input and the memory, and c_t stays in
    (-1, 1). Weights use Xavier initialisation and biases start at zero.

    The input projection does not depend on the recurrent state, so
    :class:`OptimizedEMG` computes it for a whole sequence at once
    (``project_input``) and only runs ``step`` inside the time loop.
    """

    def __init__(self, input_size: int, hidden_size: int,
                 use_layer_norm: bool = False, move_gate_input: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.move_gate_input = move_gate_input
        self.input_transform = nn.Linear(input_size, hidden_size * (2 if move_gate_input else 1))
        self.hidden_transform = nn.Linear(hidden_size, 2 * hidden_size)
        self.cell_norm = nn.LayerNorm(hidden_size) if use_layer_norm else None
        self.reset_parameters()

    def reset_parameters(self):
        for linear in (self.input_transform, self.hidden_transform):
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
        if self.cell_norm is not None:
            self.cell_norm.reset_parameters()

    def project_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """x: (..., input_size) -> (x~, U x or None), each (..., hidden_size)."""
        z = self.input_transform(x)
        if self.move_gate_input:
            x_tilde, move_in = z.chunk(2, dim=-1)
            return x_tilde, move_in
        return z, None

    def step(self, x_tilde: torch.Tensor, move_in: Optional[torch.Tensor],
             m_prev: torch.Tensor, c_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """One time step on projected input; returns (m_t, c_t)."""
        h_retain, h_move = self.hidden_transform(m_prev).chunk(2, dim=-1)
        retain = torch.sigmoid(h_retain)
        move = torch.sigmoid(h_move if move_in is None else h_move + move_in)
        pre = c_prev + retain * x_tilde
        if self.cell_norm is not None:
            pre = self.cell_norm(pre)
        c = torch.tanh(pre)
        m = x_tilde + move * (c - x_tilde)
        return m, c

    def forward(self, input: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-step API: input (batch, input_size), hidden (m, c) -> (m_t, c_t)."""
        x_tilde, move_in = self.project_input(input)
        return self.step(x_tilde, move_in, *hidden)


class OptimizedEMG(nn.Module):
    """
    Stack of eMG layers.

    The sequence is processed layer by layer: each layer projects its whole
    input sequence with one matrix multiplication, then runs the recurrence,
    and its output sequence (after dropout) becomes the next layer's input.
    This gives the same result as stepping all layers together at each time
    step, with far fewer kernel launches, and time steps are split with
    ``unbind`` so the backward pass stays linear in sequence length.

    Dropout is applied to the full output sequence of every layer except the
    last, never to the recurrent state.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 dropout: float = 0.1, use_layer_norm: bool = False,
                 move_gate_input: bool = False, locked_dropout: bool = True,
                 use_gradient_checkpointing: bool = False, checkpoint_chunk: int = 32):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.checkpoint_chunk = max(1, checkpoint_chunk)
        self.cells = nn.ModuleList([
            OptimizedEMGCell(input_size if i == 0 else hidden_size, hidden_size,
                             use_layer_norm, move_gate_input)
            for i in range(num_layers)
        ])
        drop = LockedDropout if locked_dropout else nn.Dropout
        self.layer_dropout = nn.ModuleList([drop(dropout) for _ in range(num_layers - 1)])

    def init_hidden(self, batch_size: int, device=None, dtype=None) -> Hidden:
        return [(torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),
                 torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype))
                for _ in range(self.num_layers)]

    @staticmethod
    def _run(cell: OptimizedEMGCell, x_tilde: torch.Tensor, move_in: Optional[torch.Tensor],
             m: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        steps = x_tilde.unbind(1)
        moves = move_in.unbind(1) if move_in is not None else (None,) * len(steps)
        outputs = []
        for x_t, u_t in zip(steps, moves):
            m, c = cell.step(x_t, u_t, m, c)
            outputs.append(m)
        return torch.stack(outputs, dim=1), m, c

    def _run_checkpointed(self, cell, x_tilde, move_in, m, c):
        chunks = x_tilde.split(self.checkpoint_chunk, dim=1)
        move_chunks = (move_in.split(self.checkpoint_chunk, dim=1) if move_in is not None
                       else (None,) * len(chunks))
        outputs = []
        for x_chunk, u_chunk in zip(chunks, move_chunks):
            out, m, c = checkpoint(self._run, cell, x_chunk, u_chunk, m, c, use_reentrant=False)
            outputs.append(out)
        return torch.cat(outputs, dim=1), m, c

    def forward(self, x: torch.Tensor, hidden: Optional[Hidden] = None) -> Tuple[torch.Tensor, Hidden]:
        """
        x: (batch, seq_len, input_size); hidden: list of (m, c) per layer or None.
        Returns (output (batch, seq_len, hidden_size), new hidden list). The
        hidden list passed in is not modified.
        """
        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device, x.dtype)
        use_ckpt = self.use_gradient_checkpointing and self.training and torch.is_grad_enabled()
        new_hidden: Hidden = []
        layer_input = x
        for i, cell in enumerate(self.cells):
            m, c = hidden[i]
            x_tilde, move_in = cell.project_input(layer_input)
            if use_ckpt:
                out, m, c = self._run_checkpointed(cell, x_tilde, move_in, m, c)
            else:
                out, m, c = self._run(cell, x_tilde, move_in, m, c)
            new_hidden.append((m, c))
            if i < self.num_layers - 1:
                out = self.layer_dropout[i](out)
            layer_input = out
        return layer_input, new_hidden


# ============================================================================
# Character encoders
# ============================================================================

class CharCNN(nn.Module):
    """
    Character CNN word encoder.

    Characters are embedded, convolved with filters of several widths, and
    max-pooled over the windows that start inside the word (padding never
    wins the max). An optional highway layer refines the result. Only
    non-empty tokens are encoded; empty tokens get a zero vector.
    """

    def __init__(self, config: EMGConfig):
        super().__init__()
        self.kernels = config.char_cnn_kernels
        self.char_embedding = nn.Embedding(config.char_vocab_size, config.char_embedding_dim,
                                           padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(config.char_embedding_dim, f, k)
            for f, k in zip(config.char_cnn_filters, config.char_cnn_kernels)
        ])
        self.output_dim = sum(config.char_cnn_filters)
        self.highway = HighwayNetwork(self.output_dim) if config.char_cnn_highway else None
        self.dropout = nn.Dropout(config.dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.char_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.char_embedding.weight[0].zero_()
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(conv.bias)
        if self.highway is not None:
            self.highway.reset_parameters()

    def encode_words(self, words: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """words: (N, W) character IDs, lengths: (N,) -> (N, output_dim)."""
        x = self.char_embedding(words).transpose(1, 2)
        h = torch.cat([_masked_max_conv(conv, x, lengths) for conv in self.convs], dim=-1)
        if self.highway is not None:
            h = self.highway(h)
        return self.dropout(h)

    def forward(self, char_ids: torch.Tensor, word_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """char_ids: (batch, seq_len, L) -> (batch, seq_len, output_dim)."""
        batch, seq_len, _ = char_ids.shape
        if word_lengths is None:
            word_lengths = (char_ids != 0).sum(-1)
        words, lengths, index = _select_words(char_ids, word_lengths, max(self.kernels))
        h = self.encode_words(words, lengths)
        return _scatter_words(h, index, batch * seq_len).view(batch, seq_len, -1)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ToleranceGate(nn.Module):
    """
    Gate between the lexical and the compositional route of a token.

    g = sigmoid(f([char_features; word_features]) + b), with g -> 1 for the
    lexical route (a memorised exception) and g -> 0 for the compositional
    route (a regular form). The learnable bias b starts at -1.5 (g ~ 0.18),
    so the compositional route dominates at the start of training, as most
    items follow the rules; the gate can still reach any value in (0, 1).
    """

    def __init__(self, char_dim: int, word_dim: int, hidden_dim: int = 64,
                 initial_bias: float = -1.5):
        super().__init__()
        self.initial_bias = initial_bias
        self.net = nn.Sequential(
            nn.Linear(char_dim + word_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.regularity_bias = nn.Parameter(torch.tensor(float(initial_bias)))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        with torch.no_grad():
            self.regularity_bias.fill_(self.initial_bias)

    def forward(self, char_features: torch.Tensor, word_features: torch.Tensor) -> torch.Tensor:
        logits = self.net(torch.cat([char_features, word_features], dim=-1))
        return torch.sigmoid(logits + self.regularity_bias)


class TolerancePrincipleCNN(nn.Module):
    """
    Dual-route token encoder informed by Yang's Tolerance Principle.

    - Lexical route: a word embedding stores each token whole (exceptions,
      irregular forms such as "sono", "ho", "vado").
    - Compositional route: a character CNN builds the token from its
      characters (regular forms such as "arrivano", "mangiavano"). Each
      character is embedded together with its position from the start and
      from the end of the word, so filters can anchor on prefixes and on
      suffixes, whose positions from the end line up across words of
      different lengths. Padding has its own zero position embedding, and
      max-pooling only uses windows that start inside the word.
    - Tolerance gate: blends the two routes per token. ToleranceRegularizer
      pushes the proportion of word types routed lexically toward the TP
      prediction 1/ln(V).

    Returns (embeddings (batch, seq_len, hidden_dim), gate (batch, seq_len, 1)).
    """

    def __init__(self, config: EMGConfig):
        super().__init__()
        self.max_word_length = config.max_word_length
        self.kernels = config.char_cnn_kernels
        pad = config.max_word_length
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.char_embedding = nn.Embedding(config.char_vocab_size, config.char_embedding_dim,
                                           padding_idx=0)
        self.pos_from_start = nn.Embedding(pad + 1, config.char_embedding_dim, padding_idx=pad)
        self.pos_from_end = nn.Embedding(pad + 1, config.char_embedding_dim, padding_idx=pad)
        self.convs = nn.ModuleList([
            nn.Conv1d(config.char_embedding_dim, f, k)
            for f, k in zip(config.char_cnn_filters, config.char_cnn_kernels)
        ])
        self.cnn_output_dim = sum(config.char_cnn_filters)
        self.highway = HighwayNetwork(self.cnn_output_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.tolerance_gate = ToleranceGate(self.cnn_output_dim, config.hidden_dim)
        self.char_projection = nn.Linear(self.cnn_output_dim, config.hidden_dim)
        self.output_dim = config.hidden_dim
        self.reset_parameters()

    def reset_parameters(self):
        L = self.max_word_length
        nn.init.normal_(self.word_embedding.weight, std=0.02)
        nn.init.normal_(self.char_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_from_start.weight, std=0.02)
        with torch.no_grad():
            self.char_embedding.weight[0].zero_()
            self.pos_from_start.weight[L].zero_()
            # Suffix bias: positions closer to the end of the word start with larger weights.
            for i in range(L):
                scale = 1.0 + 0.3 * (L - i) / L
                self.pos_from_end.weight[i].normal_(0.0, 0.02 * scale)
            self.pos_from_end.weight[L].zero_()
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(conv.bias)
        self.highway.reset_parameters()
        self.tolerance_gate.reset_parameters()
        nn.init.xavier_uniform_(self.char_projection.weight)
        nn.init.zeros_(self.char_projection.bias)

    def encode_characters(self, words: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """words: (N, W), lengths: (N,) -> (N, cnn_output_dim)."""
        start, end, _ = _positions(lengths, words.size(1), self.max_word_length)
        x = self.char_embedding(words) + self.pos_from_start(start) + self.pos_from_end(end)
        x = x.transpose(1, 2)
        h = torch.cat([_masked_max_conv(conv, x, lengths) for conv in self.convs], dim=-1)
        return self.dropout(self.highway(h))

    def forward(self, token_ids: torch.Tensor, char_ids: torch.Tensor,
                word_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = char_ids.shape
        if word_lengths is None:
            word_lengths = (char_ids != 0).sum(-1)
        words, lengths, index = _select_words(char_ids, word_lengths, max(self.kernels))
        char_features = _scatter_words(self.encode_characters(words, lengths), index,
                                       batch * seq_len).view(batch, seq_len, -1)
        word_features = self.word_embedding(token_ids)
        gate = self.tolerance_gate(char_features, word_features)
        output = gate * word_features + (1 - gate) * self.char_projection(char_features)
        return output, gate

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MorPieceOutput(NamedTuple):
    """Output of :class:`NeuralMorPiece`."""
    embeddings: torch.Tensor          # (batch, seq_len, hidden_dim)
    split_probs: torch.Tensor         # (batch, seq_len, L): P(boundary after character j)
    split_mask: torch.Tensor          # (batch, seq_len, L): positions where a boundary is allowed
    gate: Optional[torch.Tensor]      # (batch, seq_len, 1) lexical bypass gate, or None


class NeuralMorPiece(nn.Module):
    """
    Differentiable adaptation of MorPiece (https://github.com/cristianochesi/morpiece).

    MorPiece reads each word through a root trie (left to right) and an
    inflection trie (right to left) and places a morpheme boundary where the
    Tolerance Principle holds on both sides. Here:

    - A bidirectional character GRU plays the role of the two tries. Padding
      is packed away, so the right-to-left direction starts at the last real
      character of every word.
    - A boundary predictor gives P(boundary after character j) from both
      directions. Boundaries are allowed for 1 <= j <= length - 2: the first
      piece has at least two characters, as in MorPiece, and the last at
      least one.
    - Segment-aware pooling lets each character attend mostly to characters
      of its own soft segment (segment index of character i = expected number
      of boundaries before i).
    - A CNN with max-pooling over real characters turns the segment-aware
      features into a word vector.
    - An optional lexical bypass (word embedding + ToleranceGate) stores
      irregular forms whole.

    For "arrivavano" the forward direction sees the stem "arriv-", the
    backward direction the ending "-avano", and the boundary predictor learns
    where they meet.
    """

    def __init__(self, config: EMGConfig):
        super().__init__()
        H = config.morpiece_hidden_dim
        E = config.char_embedding_dim
        self.max_word_length = config.max_word_length
        pad = config.max_word_length
        self.split_temperature = 0.5
        self.segment_sharpness = 3.0

        self.char_embedding = nn.Embedding(config.char_vocab_size, E, padding_idx=0)
        self.pos_from_start = nn.Embedding(pad + 1, E, padding_idx=pad)
        self.pos_from_end = nn.Embedding(pad + 1, E, padding_idx=pad)
        self.encoder = nn.GRU(E, H, batch_first=True, bidirectional=True)
        self.split_predictor = nn.Sequential(
            nn.Linear(2 * H, H), nn.LayerNorm(H), nn.GELU(), nn.Linear(H, 1)
        )
        self.segment_convs = nn.ModuleList([
            nn.Conv1d(2 * H, f, k, padding=k // 2)
            for f, k in zip(config.char_cnn_filters, config.char_cnn_kernels)
        ])
        self.cnn_output_dim = sum(config.char_cnn_filters)
        self.output_projection = nn.Linear(self.cnn_output_dim, config.hidden_dim)
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.output_dim = config.hidden_dim

        if config.morpiece_lexical_bypass:
            self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
            self.bypass_gate = ToleranceGate(self.cnn_output_dim, config.hidden_dim)
        else:
            self.word_embedding = None
            self.bypass_gate = None
        self.reset_parameters()

    def reset_parameters(self):
        L = self.max_word_length
        nn.init.normal_(self.char_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_from_start.weight, std=0.02)
        nn.init.normal_(self.pos_from_end.weight, std=0.02)
        with torch.no_grad():
            self.char_embedding.weight[0].zero_()
            self.pos_from_start.weight[L].zero_()
            self.pos_from_end.weight[L].zero_()
        self.encoder.reset_parameters()
        for layer in self.split_predictor:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        # Small output weights: boundary probabilities start close to 0.5.
        nn.init.zeros_(self.split_predictor[-1].bias)
        nn.init.xavier_uniform_(self.split_predictor[-1].weight, gain=0.1)
        for conv in self.segment_convs:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(conv.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.layer_norm.reset_parameters()
        if self.word_embedding is not None:
            nn.init.normal_(self.word_embedding.weight, std=0.02)
            self.bypass_gate.reset_parameters()

    @staticmethod
    def split_mask(lengths: torch.Tensor, width: int) -> torch.Tensor:
        """Positions j with 1 <= j <= length - 2 (boundary after character j)."""
        pos = torch.arange(width, device=lengths.device).unsqueeze(0)
        return (pos >= 1) & (pos <= lengths.unsqueeze(1) - 2)

    def encode_words(self, words: torch.Tensor, lengths: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        words: (N, W) character IDs of non-empty words, lengths: (N,).
        Returns (features (N, cnn_output_dim), split_probs (N, W), split_mask (N, W)).
        """
        n, width = words.shape
        start, end, inside = _positions(lengths, width, self.max_word_length)
        x = self.char_embedding(words) + self.pos_from_start(start) + self.pos_from_end(end)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        states, _ = self.encoder(packed)
        states, _ = pad_packed_sequence(states, batch_first=True, total_length=width)

        mask = self.split_mask(lengths, width)
        logits = self.split_predictor(states).squeeze(-1)
        split_probs = torch.sigmoid(logits / self.split_temperature) * mask

        # Segment index of character i = expected number of boundaries before i.
        segment = torch.cumsum(split_probs, dim=1) - split_probs
        distance = (segment.unsqueeze(2) - segment.unsqueeze(1)).abs()
        attention = torch.exp(-self.segment_sharpness * distance)
        attention = attention * (inside.unsqueeze(1) & inside.unsqueeze(2))
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
        pooled = torch.bmm(attention, states).transpose(1, 2)        # (N, 2H, W)

        features = []
        for conv in self.segment_convs:
            y = F.relu(conv(pooled))[:, :, :width]
            y = y.masked_fill(~inside.unsqueeze(1), float("-inf"))
            features.append(y.max(dim=2).values)
        return torch.cat(features, dim=-1), split_probs, mask

    def forward(self, token_ids: torch.Tensor, char_ids: torch.Tensor,
                word_lengths: Optional[torch.Tensor] = None) -> MorPieceOutput:
        batch, seq_len, L = char_ids.shape
        total = batch * seq_len
        if word_lengths is None:
            word_lengths = (char_ids != 0).sum(-1)
        words, lengths, index = _select_words(char_ids, word_lengths, 1)
        width = words.size(1)
        features, probs, mask = self.encode_words(words, lengths)

        output = self.dropout(self.layer_norm(self.output_projection(features)))
        output = _scatter_words(output, index, total).view(batch, seq_len, -1)
        features = _scatter_words(features, index, total).view(batch, seq_len, -1)
        probs = F.pad(_scatter_words(probs, index, total), (0, L - width)).view(batch, seq_len, L)
        mask = F.pad(_scatter_words(mask.to(probs.dtype), index, total), (0, L - width))
        mask = mask.view(batch, seq_len, L) > 0.5

        gate = None
        if self.word_embedding is not None:
            lexical = self.word_embedding(token_ids)
            gate = self.bypass_gate(features, lexical)
            output = gate * lexical + (1 - gate) * output
        return MorPieceOutput(output, probs, mask, gate)

    @torch.no_grad()
    def analyze_segmentation(self, char_ids: torch.Tensor,
                             word_lengths: Optional[torch.Tensor] = None,
                             chars_to_str: Optional[Callable[[int], str]] = None,
                             threshold: float = 0.5) -> List[List[str]]:
        """
        Segment words with the current boundary predictor.

        char_ids: (N, L) character IDs of N words. ``chars_to_str`` maps a
        character ID to a string (for instance CharacterVocab.decode_char);
        by default printable ASCII IDs are shown as characters.
        Returns one list of segments per word, e.g. ['arriv', 'ava', 'no'].
        """
        was_training = self.training
        self.eval()
        if word_lengths is None:
            word_lengths = (char_ids != 0).sum(-1)
        words, lengths, index = _select_words(char_ids.unsqueeze(1), word_lengths.unsqueeze(1), 1)
        _, probs, _ = self.encode_words(words, lengths)
        self.train(was_training)

        def default(c: int) -> str:
            return chr(c) if 32 <= c < 127 else "?"
        to_str = chars_to_str or default
        results: List[List[str]] = [[] for _ in range(char_ids.size(0))]
        for row, word_idx in enumerate(index.tolist()):
            n = int(lengths[row])
            chars = [to_str(int(c)) for c in words[row, :n]]
            segments, current = [], []
            for j, ch in enumerate(chars):
                current.append(ch)
                if j < n - 1 and probs[row, j] > threshold:
                    segments.append("".join(current))
                    current = []
            segments.append("".join(current))
            results[word_idx] = segments
        return results

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# Auxiliary losses
# ============================================================================

class ToleranceRegularizer:
    """
    Pushes gate behaviour toward Yang's Tolerance Principle.

    A rule over N items is productive iff its exceptions e <= N / ln N, so
    the expected proportion of exceptions is 1 / ln N. The regulariser takes
    the mean gate value of each word type in the batch (gate -> 1 means
    "stored as an exception"), and pushes the mean over types toward
    1 / ln V, with V the vocabulary size. Counting types rather than tokens
    follows the TP, whose N is a number of types. A second term rewards clear
    decisions (gate near 0 or 1).
    """

    def __init__(self, vocab_size: int, target_exception_rate: Optional[float] = None,
                 bimodal_weight: float = 0.1):
        self.vocab_size = vocab_size
        self.bimodal_weight = bimodal_weight
        if target_exception_rate is None:
            target_exception_rate = 1.0 / math.log(vocab_size)
        self.target_exception_rate = target_exception_rate

    def __call__(self, gate_values: torch.Tensor, token_ids: torch.Tensor,
                 valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """gate_values: (batch, seq_len, 1); token_ids, valid: (batch, seq_len)."""
        gates = gate_values.squeeze(-1)
        if valid is None:
            valid = torch.ones_like(token_ids, dtype=torch.bool)
        gates = gates[valid].float()
        ids = token_ids[valid]
        if gates.numel() == 0:
            return gate_values.sum() * 0.0
        types, inverse = torch.unique(ids, return_inverse=True)
        totals = gates.new_zeros(types.numel()).index_add(0, inverse, gates)
        counts = gates.new_zeros(types.numel()).index_add(0, inverse, torch.ones_like(gates))
        type_gates = totals / counts
        rate_loss = (type_gates.mean() - self.target_exception_rate) ** 2
        bimodal_loss = (type_gates * (1 - type_gates)).mean()
        return rate_loss + self.bimodal_weight * bimodal_loss


def split_rate_loss(split_probs: torch.Tensor, split_mask: torch.Tensor,
                    target_rate: float, bimodal_weight: float = 0.1) -> torch.Tensor:
    """
    Morpheme-boundary regulariser: the mean boundary probability over the
    positions where a boundary is allowed (``split_mask``) is pushed toward
    ``target_rate``, and probabilities are pushed toward 0 or 1.
    """
    probs = split_probs[split_mask].float()
    if probs.numel() == 0:
        return split_probs.sum() * 0.0
    rate_loss = (probs.mean() - target_rate) ** 2
    bimodal_loss = (probs * (1 - probs)).mean()
    return rate_loss + bimodal_weight * bimodal_loss


# ============================================================================
# Token-to-character table
# ============================================================================

def build_char_lookup(token_strings: Sequence[str], encode_word: Callable[[str], Sequence[int]],
                      max_word_length: int) -> torch.Tensor:
    """
    Build the (vocab_size, max_word_length) table that maps every token ID to
    its character IDs, for models with a character encoder.

    token_strings: surface string of every token ID, in ID order (for a
        Hugging Face tokenizer: ``[tok.decode([i]).strip() for i in range(len(tok))]``).
    encode_word: function from a string to character IDs, for instance
        ``CharacterVocab(...).encode_word`` from data_preparation_charwise.

    Pass the result to :meth:`EMGModel.set_char_lookup`. Tokens with an empty
    string (padding, special tokens) get no characters and a zero vector.
    """
    table = torch.zeros(len(token_strings), max_word_length, dtype=torch.int16)
    for i, token in enumerate(token_strings):
        ids = list(encode_word(token))[:max_word_length]
        if ids:
            table[i, :len(ids)] = torch.tensor(ids, dtype=torch.int16)
    return table


# ============================================================================
# Language model
# ============================================================================

class EMGModel(nn.Module):
    """
    eMG-RNN language model: input encoder -> eMG layers -> output projection.

    Input encoders, chosen by :class:`EMGConfig`:
    - word embeddings (default), tied to the output matrix when possible;
    - CharCNN (``use_char_cnn``);
    - TolerancePrincipleCNN (``use_tolerance_cnn``): lexical + compositional;
    - NeuralMorPiece (``use_neural_morpiece``): learned morpheme boundaries.

    Character encoders need character IDs for every token: pass ``char_ids``
    (batch, seq_len, max_word_length), or register a token-to-character table
    once with :meth:`set_char_lookup` (see :func:`build_char_lookup`). The
    table is saved with the model and is also used by :meth:`generate` and
    by the Hugging Face wrapper.

    ``forward(idx, targets)`` does not shift targets: targets[:, t] is the
    token that should follow idx[:, t].
    """

    def __init__(self, config: EMGConfig):
        super().__init__()
        self.config = config
        self.word_embedding = None
        self.char_cnn = None
        self.tolerance_cnn = None
        self.neural_morpiece = None

        if config.use_neural_morpiece:
            self.neural_morpiece = NeuralMorPiece(config)
            input_size = config.hidden_dim
        elif config.use_tolerance_cnn:
            self.tolerance_cnn = TolerancePrincipleCNN(config)
            input_size = config.hidden_dim
        elif config.use_char_cnn:
            self.char_cnn = CharCNN(config)
            input_size = self.char_cnn.output_dim
        else:
            self.word_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
            input_size = config.embedding_dim

        drop = LockedDropout if config.locked_dropout else nn.Dropout
        self.embedding_dropout = drop(config.dropout)
        self.input_projection = (nn.Linear(input_size, config.hidden_dim)
                                 if input_size != config.hidden_dim else None)
        self.emg = OptimizedEMG(
            config.hidden_dim, config.hidden_dim, config.num_layers,
            dropout=config.dropout, use_layer_norm=config.use_layer_norm,
            move_gate_input=config.move_gate_input, locked_dropout=config.locked_dropout,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
            checkpoint_chunk=config.checkpoint_chunk,
        )
        self.output_dropout = drop(config.dropout)
        self.output_projection = nn.Linear(config.hidden_dim, config.vocab_size)

        self.tolerance_regularizer = None
        has_gate = config.use_tolerance_cnn or (config.use_neural_morpiece
                                                and config.morpiece_lexical_bypass)
        if has_gate and config.tolerance_reg_weight > 0:
            self.tolerance_regularizer = ToleranceRegularizer(config.vocab_size)

        if config.uses_characters:
            self.register_buffer("char_lookup", torch.zeros(
                config.vocab_size, config.max_word_length, dtype=torch.int16))
        else:
            self.char_lookup = None
        self._char_lookup_ready = False

        if config.tie_word_embeddings and self.word_embedding is not None:
            self.output_projection.weight = self.word_embedding.weight
        self.reset_parameters()
        logger.info("EMG model initialised with %.2fM parameters (%.2fM without embeddings).",
                    self.get_num_params(False) / 1e6, self.get_num_params(True) / 1e6)

    # ------------------------------------------------------------------ setup

    def reset_parameters(self):
        """Initialise the parameters owned directly by the model; encoders and cells initialise themselves."""
        tied = self.word_embedding is not None and \
            self.output_projection.weight is self.word_embedding.weight
        if not tied:
            nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        if self.word_embedding is not None:
            nn.init.normal_(self.word_embedding.weight, mean=0.0, std=0.02)
        if self.input_projection is not None:
            nn.init.xavier_uniform_(self.input_projection.weight)
            nn.init.zeros_(self.input_projection.bias)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Parameter count; with non_embedding=True the input encoder is excluded."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            encoder = self.neural_morpiece or self.tolerance_cnn or self.char_cnn
            if encoder is not None:
                n_params -= sum(p.numel() for p in encoder.parameters())
            elif self.output_projection.weight is not self.word_embedding.weight:
                n_params -= self.word_embedding.weight.numel()
        return n_params

    def set_char_lookup(self, table: torch.Tensor) -> None:
        """Register the (vocab_size, max_word_length) token-to-character table."""
        if self.char_lookup is None:
            raise ValueError("This model has no character encoder.")
        if tuple(table.shape) != tuple(self.char_lookup.shape):
            raise ValueError(f"Expected a table of shape {tuple(self.char_lookup.shape)}, "
                             f"got {tuple(table.shape)}.")
        with torch.no_grad():
            self.char_lookup.copy_(table.to(self.char_lookup.dtype))
        self._char_lookup_ready = True

    def _chars_for(self, idx: torch.Tensor) -> torch.Tensor:
        if not self._char_lookup_ready:
            if not bool(self.char_lookup.any()):
                raise ValueError("This model uses a character encoder: pass char_ids, or register a "
                                 "token-to-character table with set_char_lookup() "
                                 "(see build_char_lookup).")
            self._char_lookup_ready = True
        return self.char_lookup[idx].long()

    def init_hidden(self, batch_size: int, device=None, dtype=None) -> Hidden:
        """Zero state for every layer: list of (m_0, c_0)."""
        return self.emg.init_hidden(batch_size, device, dtype)

    @staticmethod
    def detach_hidden(hidden: Hidden) -> Hidden:
        """Cut the graph between batches for truncated backpropagation through time."""
        return [(m.detach(), c.detach()) for m, c in hidden]

    # ---------------------------------------------------------------- forward

    def _check_ids(self, idx: torch.Tensor, targets: Optional[torch.Tensor]) -> None:
        if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= self.config.vocab_size):
            raise ValueError(f"Token IDs out of range [0, {self.config.vocab_size}).")
        if targets is not None:
            real = targets[targets != -100]
            if real.numel() and (int(real.min()) < 0 or int(real.max()) >= self.config.vocab_size):
                raise ValueError(f"Target IDs out of range [0, {self.config.vocab_size}).")

    def _encode(self, idx, char_ids, word_lengths):
        """Returns (inputs, gate, morpiece_output, valid_tokens)."""
        if self.word_embedding is not None:
            return self.word_embedding(idx), None, None, None
        if char_ids is None:
            char_ids = self._chars_for(idx)
        if word_lengths is None:
            word_lengths = (char_ids != 0).sum(-1)
        valid = word_lengths > 0
        if self.neural_morpiece is not None:
            out = self.neural_morpiece(idx, char_ids, word_lengths)
            return out.embeddings, out.gate, out, valid
        if self.tolerance_cnn is not None:
            x, gate = self.tolerance_cnn(idx, char_ids, word_lengths)
            return x, gate, None, valid
        return self.char_cnn(char_ids, word_lengths), None, None, valid

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        hidden: Optional[Hidden] = None,
        char_ids: Optional[torch.Tensor] = None,
        word_lengths: Optional[torch.Tensor] = None,
        return_gate_values: bool = False,
        return_split_probs: bool = False,
        **kwargs,
    ):
        """
        idx: (batch, seq_len) token IDs.
        targets: (batch, seq_len) next-token IDs aligned with idx, -100 to ignore.
        hidden: state from a previous call (truncated BPTT, incremental decoding);
            a state with a different batch size is replaced by a zero state.
        char_ids, word_lengths: character input for character encoders.

        Returns (logits, loss, new_hidden), followed by the tolerance or bypass
        gate values (batch, seq_len, 1) if ``return_gate_values`` and by the
        NeuralMorPiece boundary probabilities (batch, seq_len, L) if
        ``return_split_probs``; an extra is None when the encoder has none.
        ``loss`` is the mean cross-entropy plus the weighted auxiliary losses.
        """
        batch, _ = idx.shape
        if self.config.debug_checks:
            self._check_ids(idx, targets)
        if hidden is not None and hidden[0][0].size(0) != batch:
            hidden = None

        x, gate, morpiece, valid = self._encode(idx, char_ids, word_lengths)
        x = self.embedding_dropout(x)
        if self.input_projection is not None:
            x = self.input_projection(x)
        output, new_hidden = self.emg(x, hidden)
        logits = self.output_projection(self.output_dropout(output))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                   targets.reshape(-1), ignore_index=-100)
            weight = self.config.tolerance_reg_weight
            if weight > 0 and gate is not None and self.tolerance_regularizer is not None:
                loss = loss + weight * self.tolerance_regularizer(gate, idx, valid)
            if weight > 0 and morpiece is not None:
                loss = loss + weight * split_rate_loss(morpiece.split_probs, morpiece.split_mask,
                                                       self.config.morpiece_split_rate)

        result = (logits, loss, new_hidden)
        if return_gate_values:
            result += (gate,)
        if return_split_probs:
            result += (morpiece.split_probs if morpiece is not None else None,)
        return result

    @torch.no_grad()
    def sentence_logprobs(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                          char_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Summed log-probability of input_ids[:, 1:] given the preceding tokens,
        one value per sequence. Sequences must be right-padded, with padding
        marked by ``attention_mask`` (1 = real token). Call ``model.eval()`` first.
        """
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]
        chars = char_ids[:, :-1] if char_ids is not None else None
        logits, _, _ = self(inputs, char_ids=chars)
        logp = torch.log_softmax(logits.float(), dim=-1)
        token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        if attention_mask is not None:
            token_logp = token_logp * attention_mask[:, 1:].to(token_logp.dtype)
        return token_logp.sum(-1)

    # --------------------------------------------------------------- training

    def configure_optimizers(self, weight_decay: float, learning_rate: float,
                             betas: Tuple[float, float], device_type: str) -> torch.optim.Optimizer:
        """
        AdamW with weight decay on matrices and embeddings only (not on biases,
        LayerNorm weights or scalar gate biases). Uses the fused CUDA
        implementation when ``device_type == 'cuda'`` and the parameters are on the GPU.
        """
        decay, no_decay = [], []
        for _, param in self.named_parameters():
            if param.requires_grad:
                (decay if param.dim() >= 2 else no_decay).append(param)
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        on_cuda = device_type == "cuda" and all(p.is_cuda for p in decay + no_decay)
        optimizer = torch.optim.AdamW(groups, lr=learning_rate, betas=betas,
                                      **({"fused": True} if on_cuda else {}))
        logger.info("AdamW: %d tensors with weight decay, %d without.", len(decay), len(no_decay))
        return optimizer

    # ------------------------------------------------------------- generation

    @staticmethod
    def _filter_logits(logits: torch.Tensor, top_k: Optional[int], top_p: Optional[float]) -> torch.Tensor:
        if top_k is not None and top_k > 0:
            kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            remove = probs.cumsum(-1) - probs > top_p          # always keeps the top token
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
        return logits

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: Optional[int] = None, top_p: Optional[float] = None,
                 char_ids: Optional[torch.Tensor] = None, do_sample: bool = True) -> torch.Tensor:
        """
        Continue ``idx`` (batch, seq_len) by ``max_new_tokens`` tokens. The
        prompt is read once; each new token is then fed alone with the carried
        state. Character models need a lookup table (set_char_lookup) for the
        generated tokens; ``char_ids`` may still be given for the prompt.
        """
        was_training = self.training
        self.eval()
        logits, _, hidden = self(idx, char_ids=char_ids)
        out = idx
        for step in range(max_new_tokens):
            next_logits = logits[:, -1, :].float()
            if do_sample:
                next_logits = self._filter_logits(next_logits / max(temperature, 1e-6), top_k, top_p)
                next_token = torch.multinomial(F.softmax(next_logits, dim=-1), num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            out = torch.cat((out, next_token), dim=1)
            if step + 1 < max_new_tokens:
                logits, _, hidden = self(next_token, hidden=hidden)
        self.train(was_training)
        return out

    @torch.no_grad()
    def analyze_gate_values(self, idx: torch.Tensor, char_ids: Optional[torch.Tensor] = None,
                            tokenizer=None) -> dict:
        """
        Statistics of the lexical/compositional gate (TolerancePrincipleCNN or
        the NeuralMorPiece bypass). With a tokenizer, also lists the ten most
        compositional and most lexical tokens.
        """
        if self.tolerance_cnn is None and (self.neural_morpiece is None
                                           or self.neural_morpiece.bypass_gate is None):
            raise ValueError("analyze_gate_values needs use_tolerance_cnn=True or "
                             "use_neural_morpiece=True with morpiece_lexical_bypass=True.")
        was_training = self.training
        self.eval()
        _, _, _, gate_values = self(idx, char_ids=char_ids, return_gate_values=True)
        self.train(was_training)
        gates = gate_values.squeeze(-1).float().cpu()
        tokens = idx.cpu()
        results = {
            "mean_gate": float(gates.mean()),
            "lexical_rate": float((gates > 0.5).float().mean()),
            "compositional_rate": float((gates <= 0.5).float().mean()),
            "gate_std": float(gates.std()) if gates.numel() > 1 else 0.0,
            "target_exception_rate": 1.0 / math.log(self.config.vocab_size),
        }
        if tokenizer is not None:
            flat_gates, flat_tokens = gates.flatten(), tokens.flatten()
            order = flat_gates.argsort()
            results["most_compositional"] = [
                (tokenizer.decode([int(flat_tokens[i])]), float(flat_gates[i])) for i in order[:10]]
            results["most_lexical"] = [
                (tokenizer.decode([int(flat_tokens[i])]), float(flat_gates[i]))
                for i in order.flip(0)[:10]]
        return results


# ============================================================================
# Hugging Face wrapper
# ============================================================================

_TRANSFORMERS_MAJOR = int(transformers.__version__.split(".")[0])


class EMGHFConfig(PretrainedConfig):
    """Hugging Face configuration for :class:`EMGLanguageModel`; fields mirror :class:`EMGConfig`."""
    model_type = "emg"

    def __init__(
        self,
        vocab_size: int = 50257,
        embedding_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        locked_dropout: bool = True,
        use_layer_norm: bool = False,
        move_gate_input: bool = False,
        use_gradient_checkpointing: bool = False,
        checkpoint_chunk: int = 32,
        debug_checks: bool = False,
        use_char_cnn: bool = False,
        use_tolerance_cnn: bool = False,
        use_neural_morpiece: bool = False,
        char_vocab_size: int = 256,
        char_embedding_dim: int = 16,
        char_cnn_filters: Sequence[int] = (32, 64, 128, 256),
        char_cnn_kernels: Sequence[int] = (1, 2, 3, 4),
        max_word_length: int = 50,
        char_cnn_highway: bool = True,
        morpiece_hidden_dim: int = 128,
        morpiece_lexical_bypass: bool = True,
        tolerance_reg_weight: float = 0.01,
        morpiece_split_rate: float = 0.25,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.locked_dropout = locked_dropout
        self.use_layer_norm = use_layer_norm
        self.move_gate_input = move_gate_input
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.checkpoint_chunk = checkpoint_chunk
        self.debug_checks = debug_checks
        self.use_char_cnn = use_char_cnn
        self.use_tolerance_cnn = use_tolerance_cnn
        self.use_neural_morpiece = use_neural_morpiece
        self.char_vocab_size = char_vocab_size
        self.char_embedding_dim = char_embedding_dim
        self.char_cnn_filters = list(char_cnn_filters)
        self.char_cnn_kernels = list(char_cnn_kernels)
        self.max_word_length = max_word_length
        self.char_cnn_highway = char_cnn_highway
        self.morpiece_hidden_dim = morpiece_hidden_dim
        self.morpiece_lexical_bypass = morpiece_lexical_bypass
        self.tolerance_reg_weight = tolerance_reg_weight
        self.morpiece_split_rate = morpiece_split_rate
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def to_emg_config(self) -> EMGConfig:
        values = {f.name: getattr(self, f.name) for f in fields(EMGConfig)}
        return EMGConfig(**values)

    @classmethod
    def from_emg_config(cls, config: EMGConfig, **kwargs) -> "EMGHFConfig":
        values = {f.name: getattr(config, f.name) for f in fields(EMGConfig)}
        values.update(kwargs)
        return cls(**values)


class EMGLanguageModel(PreTrainedModel):
    """
    Hugging Face wrapper around :class:`EMGModel`.

    - ``save_pretrained`` / ``from_pretrained`` (weights, configuration and,
      for character models, the token-to-character table);
    - ``AutoModelForCausalLM`` once this module is imported; a saved model
      also loads with ``trust_remote_code=True`` without the import;
    - Hugging Face conventions: ``labels`` are the input IDs (shifted inside
      ``forward``), ``attention_mask`` marks right padding, and outputs carry
      ``.loss`` and ``.logits``, so the model works with
      lm-evaluation-harness (``HFLM``) for log-likelihood tasks such as BLiMP.
    """
    config_class = EMGHFConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _no_split_modules = ["OptimizedEMG"]
    if _TRANSFORMERS_MAJOR >= 5:
        _tied_weights_keys = {"model.output_projection.weight": "model.word_embedding.weight"}
    else:
        _tied_weights_keys = ["model.output_projection.weight"]

    def __init__(self, config: EMGHFConfig):
        super().__init__(config)
        emg_config = config.to_emg_config()
        config.tie_word_embeddings = emg_config.tie_word_embeddings
        self.model = EMGModel(emg_config)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        # Every eMG module initialises its own subtree in reset_parameters();
        # Hugging Face visits children before parents, so custom inits win.
        if module is not self and hasattr(module, "reset_parameters"):
            module.reset_parameters()

    def get_input_embeddings(self):
        return self.model.word_embedding

    def set_input_embeddings(self, value):
        self.model.word_embedding = value

    def get_output_embeddings(self):
        return self.model.output_projection

    def set_output_embeddings(self, value):
        self.model.output_projection = value

    def set_char_lookup(self, table: torch.Tensor) -> None:
        self.model.set_char_lookup(table)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, past_key_values: Optional[Hidden] = None,
                use_cache: Optional[bool] = None, char_ids: Optional[torch.Tensor] = None,
                return_dict: Optional[bool] = None, **kwargs) -> CausalLMOutputWithPast:
        targets = None
        if labels is not None:
            targets = torch.full_like(labels, -100)
            targets[:, :-1] = labels[:, 1:]
            if attention_mask is not None:
                targets[:, :-1] = targets[:, :-1].masked_fill(attention_mask[:, 1:] == 0, -100)
        logits, loss, hidden = self.model(input_ids, targets=targets, hidden=past_key_values,
                                          char_ids=char_ids)
        return CausalLMOutputWithPast(loss=loss, logits=logits,
                                      past_key_values=hidden if use_cache else None)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: Optional[int] = None,
                 max_length: Optional[int] = None, do_sample: bool = False,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None, char_ids: Optional[torch.Tensor] = None,
                 **kwargs) -> torch.Tensor:
        """
        Greedy or sampled continuation (see :meth:`EMGModel.generate`).
        ``max_length`` counts the prompt, ``max_new_tokens`` does not; other
        Hugging Face generation arguments are ignored.
        """
        if max_new_tokens is None:
            max_new_tokens = max(0, (max_length or input_ids.size(1) + 20) - input_ids.size(1))
        return self.model.generate(input_ids, max_new_tokens, temperature=temperature,
                                   top_k=top_k, top_p=top_p, char_ids=char_ids, do_sample=do_sample)


try:
    from transformers import AutoConfig, AutoModelForCausalLM
    AutoConfig.register("emg", EMGHFConfig)
    AutoModelForCausalLM.register(EMGHFConfig, EMGLanguageModel)
except ValueError:
    pass  # already registered (module imported twice)
EMGHFConfig.register_for_auto_class()
EMGLanguageModel.register_for_auto_class("AutoModelForCausalLM")


# ============================================================================
# Convenience constructors
# ============================================================================

def create_emg_small(**kwargs) -> EMGModel:
    """Small model (256 hidden units, 2 layers), for tests."""
    return EMGModel(EMGConfig(hidden_dim=256, embedding_dim=256, num_layers=2, **kwargs))


def create_emg_base(**kwargs) -> EMGModel:
    """Base model (512 hidden units, 2 layers)."""
    return EMGModel(EMGConfig(hidden_dim=512, embedding_dim=512, num_layers=2, **kwargs))


def create_emg_large(**kwargs) -> EMGModel:
    """Large model (1024 hidden units, 3 layers)."""
    return EMGModel(EMGConfig(hidden_dim=1024, embedding_dim=1024, num_layers=3, **kwargs))


# ============================================================================
# Self-test
# ============================================================================

if __name__ == "__main__":
    import tempfile

    torch.manual_seed(0)
    batch, seq_len, vocab = 2, 12, 400
    chars = 30

    def random_char_table(n_tokens: int, width: int) -> torch.Tensor:
        lengths = torch.randint(1, 9, (n_tokens,))
        table = torch.zeros(n_tokens, width, dtype=torch.int16)
        for i, n in enumerate(lengths.tolist()):
            table[i, :n] = torch.randint(97, 123, (n,), dtype=torch.int16)
        table[0] = 0                                    # token 0 = padding, no characters
        return table

    idx = torch.randint(1, vocab, (batch, seq_len))
    targets = torch.randint(1, vocab, (batch, seq_len))
    table = random_char_table(vocab, chars)

    print("1. Word-embedding model")
    model = EMGModel(EMGConfig(vocab_size=vocab, embedding_dim=64, hidden_dim=64))
    logits, loss, hidden = model(idx, targets)
    loss.backward()
    model.eval()
    with torch.no_grad():
        full, _, _ = model(idx)
        part, _, h = model(idx[:, :6])
        rest, _, _ = model(idx[:, 6:], hidden=h)
    assert torch.allclose(full[:, 6:], rest, atol=1e-5), "incremental state mismatch"
    cell = model.emg.cells[0]
    x = model.word_embedding(idx)
    m = c = torch.zeros(batch, 64)
    for t in range(seq_len):
        m, c = cell(x[:, t], (m, c))
    first_layer, _, _ = OptimizedEMG._run(cell, *cell.project_input(x), torch.zeros(batch, 64),
                                          torch.zeros(batch, 64))
    assert torch.allclose(first_layer[:, -1], m, atol=1e-5), "fast loop mismatch"
    out = model.generate(idx[:, :4], max_new_tokens=5, do_sample=False)
    assert out.shape == (batch, 9)
    print(f"   loss {loss.item():.3f}; incremental decoding and fast loop consistent")

    print("2. CharCNN with a token-to-character table")
    model = EMGModel(EMGConfig(vocab_size=vocab, hidden_dim=64, use_char_cnn=True,
                               char_cnn_filters=(16, 16, 16, 16), max_word_length=chars))
    model.set_char_lookup(table)
    logits, loss, _ = model(idx, targets)
    loss.backward()
    print(f"   loss {loss.item():.3f}; generated {model.generate(idx[:, :3], 4).shape[1]} tokens")

    print("3. Tolerance Principle CNN")
    model = EMGModel(EMGConfig(vocab_size=vocab, hidden_dim=64, use_tolerance_cnn=True,
                               char_cnn_filters=(16, 16, 16, 16), max_word_length=chars))
    model.set_char_lookup(table)
    logits, loss, _, gates = model(idx, targets, return_gate_values=True)
    loss.backward()
    stats = model.analyze_gate_values(idx)
    print(f"   loss {loss.item():.3f}; mean gate {stats['mean_gate']:.3f} "
          f"(TP target {stats['target_exception_rate']:.3f})")

    print("4. Neural MorPiece")
    model = EMGModel(EMGConfig(vocab_size=vocab, hidden_dim=64, use_neural_morpiece=True,
                               char_cnn_filters=(16, 16, 16, 16), max_word_length=chars,
                               morpiece_hidden_dim=32))
    model.set_char_lookup(table)
    logits, loss, _, gates, splits = model(idx, targets, return_gate_values=True,
                                           return_split_probs=True)
    loss.backward()
    unused = [n for n, p in model.named_parameters() if p.grad is None]
    assert not unused, f"parameters without gradient: {unused}"
    word = torch.tensor([[ord(ch) for ch in "arrivavano"] + [0] * (chars - 10)])
    segments = model.neural_morpiece.analyze_segmentation(word)
    print(f"   loss {loss.item():.3f}; split probs {tuple(splits.shape)}; "
          f"untrained segmentation {segments[0]}")

    print("5. Hugging Face wrapper: save and reload")
    import importlib                              # saved code must come from the module, not __main__
    module = importlib.import_module("model_eMG")
    transformers.logging.set_verbosity_error()
    hf = module.EMGLanguageModel(module.EMGHFConfig(vocab_size=vocab, embedding_dim=64,
                                                    hidden_dim=64)).eval()
    with tempfile.TemporaryDirectory() as tmp:
        hf.save_pretrained(tmp)
        reloaded = module.EMGLanguageModel.from_pretrained(tmp).eval()
    with torch.no_grad():
        a, b = hf(idx).logits, reloaded(idx).logits
    assert torch.allclose(a, b, atol=1e-6), "reloaded model differs"
    lp = hf.model.sentence_logprobs(idx)
    print(f"   reloaded logits identical; sentence log-probs {lp.tolist()}")

    print("All self-tests passed.")
