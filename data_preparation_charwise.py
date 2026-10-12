#!/usr/bin/env python3
"""
Character-level data preparation for the tokenizer-free eMG-RNN
(expectation-based Minimalist Grammar Recurrent Neural Network).

Raw text is split into words; every distinct word (type) gets an ID and a
row of character IDs. Examples are stored as arrays of type IDs, so the
on-disk cache takes 4 bytes per word, and a batch becomes a character tensor
with one indexing operation. Batches are padded only to the longest example
and the longest word they contain.

Contents
--------
CharLevelConfig       preprocessing settings
CharacterVocab        the character inventory (single source of truth)
preprocess_line       text normalisation shared by training and scoring
CharLevelDataset      cached, type-indexed dataset
train_val_split       deterministic train/validation split
CharBatchCollator     dynamic padding into character tensors
BucketBatchSampler    length-bucketed, DDP-aware batch sampler
create_char_dataloader convenience constructor

Author: Cristiano Chesi (IUSS Pavia, NeTS Lab).
Development, debugging and optimization assisted by Claude (Anthropic).
Released under the MIT License (see LICENSE).
"""

import hashlib
import json
import logging
import math
import os
import re
import shutil
import tempfile
import unicodedata
from array import array
from dataclasses import asdict, dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

logger = logging.getLogger(__name__)

BOS_WORD = "<s>"
EOS_WORD = "</s>"
SPECIAL_WORDS = (BOS_WORD, EOS_WORD)
CACHE_FORMAT = 2


# ============================================================================
# Configuration and character inventory
# ============================================================================

@dataclass
class CharLevelConfig:
    """
    Preprocessing settings.

    max_word_length: characters kept per word (longer words are truncated).
    max_seq_length: words per example, sentence markers included; longer
        lines are split into several examples.
    char_vocab_size: size of the character embedding table (must exceed the
        largest character ID, see CharacterVocab).
    lowercase: lowercase the text.
    split_punctuation: separate . , ; : ! ? " ( ) [ ] « » ... from words.
    add_sentence_markers: wrap every example in <s> ... </s>, so that the
        first word of a sentence is predicted too.
    use_cache, cache_dir: on-disk cache (default <corpus>/.cache_char, or
        ~/.cache/emg_rnn when the corpus directory is read-only).
    """
    max_word_length: int = 50
    max_seq_length: int = 256
    char_vocab_size: int = 256
    lowercase: bool = True
    split_punctuation: bool = True
    add_sentence_markers: bool = True
    use_cache: bool = True
    cache_dir: Optional[str] = None


class CharacterVocab:
    """
    Character inventory shared by the dataset, the models and the scorers.

    IDs 0-5 are reserved: 0 padding, 1 unknown character, 2 end of word and
    3 beginning of word (used by the word speller), 4 and 5 the single
    characters of the sentence markers <s> and </s>. Printable ASCII keeps
    its code point (32-126); accented letters and typographic punctuation
    follow from 128. Any other character maps to 1.
    """
    PAD, UNK, EOW, BOW, BOS, EOS = 0, 1, 2, 3, 4, 5

    LETTERS = "àèéìíîòóùúáâäëïöüçñß"
    TYPOGRAPHY = (
        "‘",  # left single quotation mark
        "’",  # right single quotation mark / typographic apostrophe
        "“",  # left double quotation mark
        "”",  # right double quotation mark
        "«",  # left guillemet
        "»",  # right guillemet
        "–",  # en dash
        "—",  # em dash
        "…",  # ellipsis
    )

    def __init__(self, config: Optional[CharLevelConfig] = None):
        self.config = config or CharLevelConfig()
        self.char_to_id: Dict[str, int] = {}
        for code in range(32, 127):
            self.char_to_id[chr(code)] = code
        extended = list(self.LETTERS)
        extended += [c.upper() for c in self.LETTERS if len(c.upper()) == 1 and c.upper() != c]
        extended += list(self.TYPOGRAPHY)
        next_id = 128
        for char in extended:
            if char not in self.char_to_id:
                self.char_to_id[char] = next_id
                next_id += 1
        self.id_to_char: Dict[int, str] = {v: k for k, v in self.char_to_id.items()}
        self.size = next_id
        if self.config.char_vocab_size < self.size:
            raise ValueError(f"char_vocab_size must be at least {self.size}")

    def encode_char(self, char: str) -> int:
        if self.config.lowercase:
            char = char.lower()
        return self.char_to_id.get(char, self.UNK)

    def decode_char(self, idx: int) -> str:
        special = {self.PAD: "", self.UNK: "�", self.EOW: "", self.BOW: "",
                   self.BOS: BOS_WORD, self.EOS: EOS_WORD}
        return special.get(idx, self.id_to_char.get(idx, "�"))

    def encode_word(self, word: str) -> List[int]:
        """Character IDs of a word, truncated to max_word_length."""
        if word == BOS_WORD:
            return [self.BOS]
        if word == EOS_WORD:
            return [self.EOS]
        return [self.encode_char(c) for c in word[:self.config.max_word_length]]

    def decode_word(self, ids: Sequence[int]) -> str:
        return "".join(self.decode_char(int(i)) for i in ids if int(i) != self.PAD)

    def signature(self) -> str:
        """Hash of the inventory, used in cache keys."""
        items = sorted(self.char_to_id.items(), key=lambda kv: kv[1])
        return hashlib.md5(json.dumps(items, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]


_PUNCTUATION = re.compile(r"([.,;:!?\"()\[\]«»“”…])")


def preprocess_line(line: str, config: CharLevelConfig) -> List[str]:
    """Unicode NFC normalisation, optional lowercasing and punctuation splitting, whitespace split."""
    line = unicodedata.normalize("NFC", line)
    if config.lowercase:
        line = line.lower()
    if config.split_punctuation:
        line = _PUNCTUATION.sub(r" \1 ", line)
    return line.split()


# ============================================================================
# Dataset
# ============================================================================

class CharLevelDataset(Dataset):
    """
    Dataset of word sequences read from a directory of UTF-8 text files.

    Regimens
        naturalistic    each line is an example (long lines are split)
        conversational  consecutive lines are paired into one example
        sliding_window  the corpus is concatenated and cut into windows of
                        max_seq_length words with 50% overlap

    Each example is a sequence of word-type IDs; type 0 is padding, and the
    type table holds the strings (``types``), their character IDs
    (``type_chars``, shape (n_types, max_word_length)), their lengths and
    their corpus counts. ``__getitem__`` returns a LongTensor of type IDs.
    """

    def __init__(self, file_path: str, config: CharLevelConfig,
                 regimen: str = "naturalistic", cache_dir: Optional[str] = None):
        if regimen not in ("naturalistic", "conversational", "sliding_window"):
            raise ValueError(f"Unknown regimen {regimen!r}")
        self.file_path = file_path
        self.config = config
        self.regimen = regimen
        self.char_vocab = CharacterVocab(config)

        cache_root = self._cache_root(cache_dir or config.cache_dir)
        self.cache_path = os.path.join(cache_root, f"chardata_{self._cache_key()}")
        if config.use_cache and os.path.isdir(self.cache_path):
            self._load(self.cache_path)
            logger.info("Loaded cached dataset from %s", self.cache_path)
        else:
            self._build()
            if config.use_cache:
                self._save_atomic(cache_root, self.cache_path)
                logger.info("Saved dataset cache to %s", self.cache_path)
        self.lengths = np.diff(self.offsets).astype(np.int64)
        self.num_tokens = int(self.lengths.sum() - len(self.lengths))
        logger.info("%d examples, %d predicted words, %d word types",
                    len(self), self.num_tokens, len(self.types) - 1)

    # ------------------------------------------------------------- caching

    def _cache_root(self, cache_dir: Optional[str]) -> str:
        candidates = [cache_dir] if cache_dir else [os.path.join(self.file_path, ".cache_char")]
        candidates.append(os.path.join(os.path.expanduser("~"), ".cache", "emg_rnn"))
        for candidate in candidates:
            try:
                os.makedirs(candidate, exist_ok=True)
                if os.access(candidate, os.W_OK):
                    return candidate
            except OSError:
                continue
        raise OSError("No writable cache directory; pass cache_dir or set use_cache=False.")

    def _corpus_files(self) -> List[str]:
        return [name for name in sorted(os.listdir(self.file_path))
                if not name.startswith(".") and os.path.isfile(os.path.join(self.file_path, name))]

    def _cache_key(self) -> str:
        info = []
        for name in self._corpus_files():
            stat = os.stat(os.path.join(self.file_path, name))
            info.append((name, stat.st_mtime_ns, stat.st_size))
        settings = {k: v for k, v in asdict(self.config).items() if k not in ("use_cache", "cache_dir")}
        key = json.dumps([CACHE_FORMAT, self.regimen, settings, self.char_vocab.signature(), info])
        return hashlib.md5(key.encode("utf-8")).hexdigest()[:16]

    def _save_atomic(self, cache_root: str, final_path: str) -> None:
        tmp = tempfile.mkdtemp(prefix=".tmp_", dir=cache_root)
        try:
            np.save(os.path.join(tmp, "tokens.npy"), self.tokens)
            np.save(os.path.join(tmp, "offsets.npy"), self.offsets)
            np.save(os.path.join(tmp, "type_chars.npy"), self.type_chars)
            np.save(os.path.join(tmp, "type_lengths.npy"), self.type_lengths)
            np.save(os.path.join(tmp, "type_counts.npy"), self.type_counts)
            with open(os.path.join(tmp, "types.json"), "w", encoding="utf-8") as f:
                json.dump(self.types, f, ensure_ascii=False)
            os.replace(tmp, final_path)
        except OSError:
            # Another process finished the same cache first; keep that one.
            shutil.rmtree(tmp, ignore_errors=True)

    def _load(self, path: str) -> None:
        self.tokens = np.load(os.path.join(path, "tokens.npy"), mmap_mode="r")
        self.offsets = np.load(os.path.join(path, "offsets.npy"))
        self.type_chars = np.load(os.path.join(path, "type_chars.npy"))
        self.type_lengths = np.load(os.path.join(path, "type_lengths.npy"))
        self.type_counts = np.load(os.path.join(path, "type_counts.npy"))
        with open(os.path.join(path, "types.json"), encoding="utf-8") as f:
            self.types = json.load(f)

    # ---------------------------------------------------------- processing

    def _read_lines(self) -> Iterator[List[str]]:
        for name in self._corpus_files():
            try:
                with open(os.path.join(self.file_path, name), encoding="utf-8") as f:
                    for line in f:
                        words = preprocess_line(line, self.config)
                        if words:
                            yield words
            except (OSError, UnicodeDecodeError) as error:
                logger.warning("Skipping %s: %s", name, error)

    def _examples(self) -> Iterator[List[str]]:
        cfg = self.config
        marks = 2 if cfg.add_sentence_markers else 0
        body = max(1, cfg.max_seq_length - marks)

        def wrap(words: List[str]) -> List[str]:
            return [BOS_WORD] + words + [EOS_WORD] if cfg.add_sentence_markers else words

        if self.regimen == "naturalistic":
            for words in self._read_lines():
                for start in range(0, len(words), body):
                    yield wrap(words[start:start + body])
        elif self.regimen == "conversational":
            pending = None
            for words in self._read_lines():
                if pending is None:
                    pending = words
                    continue
                combined = pending + words
                pending = None
                for start in range(0, len(combined), body):
                    yield wrap(combined[start:start + body])
            if pending is not None:
                yield wrap(pending[:body])
        else:
            stream: List[str] = []
            for words in self._read_lines():
                stream.extend(wrap(words))
            window, stride = cfg.max_seq_length, max(1, cfg.max_seq_length // 2)
            if len(stream) <= window:
                if stream:
                    yield stream
            else:
                starts = list(range(0, len(stream) - window + 1, stride))
                if starts[-1] + window < len(stream):
                    starts.append(len(stream) - window)
                for start in starts:
                    yield stream[start:start + window]

    def _build(self) -> None:
        logger.info("Processing %s (%s regimen)", self.file_path, self.regimen)
        type_ids: Dict[str, int] = {"": 0}
        types: List[str] = [""]
        counts = array("q", [0])
        tokens = array("i")
        offsets = array("q", [0])
        for words in self._examples():
            if len(words) < 2:
                continue
            for word in words:
                tid = type_ids.get(word)
                if tid is None:
                    tid = type_ids[word] = len(types)
                    types.append(word)
                    counts.append(0)
                counts[tid] += 1
                tokens.append(tid)
            offsets.append(len(tokens))

        L = self.config.max_word_length
        type_chars = np.zeros((len(types), L), dtype=np.int16)
        type_lengths = np.zeros(len(types), dtype=np.int16)
        for tid, word in enumerate(types[1:], start=1):
            ids = self.char_vocab.encode_word(word)
            type_chars[tid, :len(ids)] = ids
            type_lengths[tid] = len(ids)

        self.types = types
        self.tokens = np.frombuffer(tokens, dtype=np.int32).copy()
        self.offsets = np.frombuffer(offsets, dtype=np.int64).copy()
        self.type_chars = type_chars
        self.type_lengths = type_lengths
        self.type_counts = np.frombuffer(counts, dtype=np.int64).copy()

    # ------------------------------------------------------------- access

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, idx: int) -> torch.Tensor:
        start, end = int(self.offsets[idx]), int(self.offsets[idx + 1])
        return torch.from_numpy(np.asarray(self.tokens[start:end], dtype=np.int64))

    def get_words(self, idx: int) -> List[str]:
        """Words of example ``idx`` (sentence markers included)."""
        start, end = int(self.offsets[idx]), int(self.offsets[idx + 1])
        return [self.types[t] for t in self.tokens[start:end]]

    def get_raw_text(self, idx: int) -> str:
        return " ".join(self.get_words(idx))


def train_val_split(dataset: CharLevelDataset, val_fraction: float, seed: int = 0
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic split of example indices into (train, validation)."""
    order = np.random.default_rng(seed).permutation(len(dataset))
    n_val = int(round(len(dataset) * val_fraction)) if val_fraction > 0 else 0
    if 0 < val_fraction and n_val == 0 and len(dataset) > 1:
        n_val = 1
    return np.sort(order[n_val:]), np.sort(order[:n_val])


# ============================================================================
# Batching
# ============================================================================

class CharBatchCollator:
    """
    Turn a list of type-ID sequences into a padded batch.

    Returns a dict of tensors, with T = longest example - 1:
        input_types, target_types   (B, T)  type IDs, 0 = padding
        input_chars                 (B, T, Wi) character IDs of the input words
        input_lengths               (B, T)
        target_chars                (B, T, Wt) character IDs of the next words
        target_lengths              (B, T)
    Wi and Wt are the longest input and target words of the batch.
    """

    def __init__(self, dataset: CharLevelDataset):
        self.type_chars = dataset.type_chars
        self.type_lengths = dataset.type_lengths.astype(np.int64)

    def __call__(self, sequences: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        T = max(len(s) for s in sequences) - 1
        types = np.zeros((len(sequences), T + 1), dtype=np.int64)
        for row, seq in enumerate(sequences):
            types[row, :len(seq)] = seq.numpy()
        inputs, targets = types[:, :-1], types[:, 1:]
        batch = {"input_types": inputs, "target_types": targets}
        for side, ids in (("input", inputs), ("target", targets)):
            lengths = self.type_lengths[ids]
            width = max(1, int(lengths.max()))
            batch[f"{side}_chars"] = self.type_chars[ids][:, :, :width].astype(np.int64)
            batch[f"{side}_lengths"] = lengths
        return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in batch.items()}


class BucketBatchSampler(Sampler):
    """
    Batch sampler that groups examples of similar length.

    Each epoch the indices are shuffled (seed + epoch), split evenly across
    DDP ranks, cut into pools of ``batch_size * pool_factor`` examples,
    sorted by length inside each pool, cut into batches, and the batches are
    shuffled. Every rank gets the same number of batches. With
    ``shuffle=False`` the order is deterministic and sorted by length.
    """

    def __init__(self, lengths: np.ndarray, batch_size: int, indices: Optional[np.ndarray] = None,
                 shuffle: bool = True, seed: int = 0, rank: int = 0, world_size: int = 1,
                 drop_last: bool = True, pool_factor: int = 50):
        self.lengths = np.asarray(lengths)
        self.indices = np.arange(len(lengths)) if indices is None else np.asarray(indices)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.drop_last = drop_last
        self.pool_factor = pool_factor
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _shard(self) -> np.ndarray:
        idx = self.indices
        if self.shuffle:
            idx = np.random.default_rng(self.seed + self.epoch).permutation(idx)
        per_rank = len(idx) // self.world_size
        return idx[self.rank * per_rank:(self.rank + 1) * per_rank]

    def _batches(self) -> List[np.ndarray]:
        shard = self._shard()
        if not self.shuffle:
            shard = shard[np.argsort(self.lengths[shard], kind="stable")]
            batches = [shard[i:i + self.batch_size] for i in range(0, len(shard), self.batch_size)]
        else:
            rng = np.random.default_rng(self.seed + 7919 * (self.epoch + 1))
            pool = self.batch_size * self.pool_factor
            batches = []
            for start in range(0, len(shard), pool):
                chunk = shard[start:start + pool]
                chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
                batches.extend(chunk[i:i + self.batch_size] for i in range(0, len(chunk), self.batch_size))
            rng.shuffle(batches)
        if self.drop_last:
            batches = [b for b in batches if len(b) == self.batch_size]
        return batches

    def __iter__(self):
        for batch in self._batches():
            yield batch.tolist()

    def __len__(self) -> int:
        per_rank = len(self.indices) // self.world_size
        if self.drop_last:
            if self.shuffle:
                pool = self.batch_size * self.pool_factor
                full, rest = divmod(per_rank, pool)
                return full * self.pool_factor + rest // self.batch_size
            return per_rank // self.batch_size
        return math.ceil(per_rank / self.batch_size)


def create_char_dataloader(file_path: str, config: CharLevelConfig, batch_size: int = 32,
                           shuffle: bool = True, num_workers: int = 4,
                           regimen: str = "naturalistic") -> DataLoader:
    """DataLoader over a whole corpus directory, with length bucketing and dynamic padding."""
    dataset = CharLevelDataset(file_path, config, regimen=regimen)
    sampler = BucketBatchSampler(dataset.lengths, batch_size, shuffle=shuffle, drop_last=shuffle)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=CharBatchCollator(dataset),
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())


# ============================================================================
# Self-test
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_dir = tempfile.mkdtemp(prefix="emg_char_data_")
    text = (
        "Il gatto nero dorme sulla sedia.\n"
        "I gatti neri dormono sulle sedie.\n"
        "La casa è grande e bella.\n"
        "Le case sono grandi e belle.\n"
        "Arriviamo domani mattina presto.\n"
        "Arrivavamo sempre in ritardo.\n"
        "Mangiamo la pasta al pomodoro.\n"
        "Mangiavano spesso insieme all’aperto.\n"
    )
    with open(os.path.join(test_dir, "test.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    try:
        config = CharLevelConfig(max_word_length=30, max_seq_length=32)
        vocab = CharacterVocab(config)
        assert vocab.encode_word("all’aperto")[3] != CharacterVocab.UNK, "typographic apostrophe lost"
        assert vocab.decode_word(vocab.encode_word("è")) == "è"

        dataset = CharLevelDataset(test_dir, config)
        again = CharLevelDataset(test_dir, config)            # second time: from the cache
        assert np.array_equal(np.asarray(dataset.tokens), np.asarray(again.tokens))
        print("Example 0:", dataset.get_words(0))

        train_idx, val_idx = train_val_split(dataset, 0.25, seed=0)
        print(f"Train/validation examples: {len(train_idx)}/{len(val_idx)}")

        loader = DataLoader(dataset, batch_sampler=BucketBatchSampler(dataset.lengths, 2, train_idx),
                            collate_fn=CharBatchCollator(dataset))
        for batch in loader:
            print({k: tuple(v.shape) for k, v in batch.items()})
            break
        print("All self-tests passed.")
    finally:
        shutil.rmtree(test_dir)
