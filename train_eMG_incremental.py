#!/usr/bin/env python3
"""
Train the tokenizer-free eMG-RNN (expectation-based Minimalist Grammar
Recurrent Neural Network), model_eMG_incremental.IncrementalEMGModel.

Usage
    python train_eMG_incremental.py --corpus_path ./data
    torchrun --nproc_per_node=4 train_eMG_incremental.py --corpus_path ./data --amp bf16

Pipeline
    1. The corpus is preprocessed and cached (data_preparation_charwise); with
       several processes, rank 0 builds the cache and the others load it.
    2. A held-out fraction of the examples (--val_fraction) is used for validation.
    3. Before training, and every --vocab_update_interval optimizer steps, the
       next --vocab_update_lines training examples (in a fixed random order,
       identical on every process) are fed to the ToleranceTrie, and the
       morpheme vocabulary grows. Updates stop after one pass over the
       training data. Vocabulary growth is logged to vocab_growth.tsv.
    4. Batches are grouped by length and padded only as much as they need.
    5. The learning rate warms up linearly, then follows a cosine decay to
       --min_lr over the optimizer steps that the token budget requires.
    6. The checkpoint with the lowest validation loss is kept in best_model/,
       the last one in final_model/.

Author: Cristiano Chesi (IUSS Pavia, NeTS Lab).
Development, debugging and optimization assisted by Claude (Anthropic).
Released under the MIT License (see LICENSE).
"""

import os


def setup_environment():
    """Allocator and threading defaults; existing environment values are kept."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    threads = str(min(os.cpu_count() or 4, 32))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, threads)


setup_environment()

import argparse
import logging
import math
import sys
import time
from contextlib import nullcontext
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from data_preparation_charwise import (SPECIAL_WORDS, BucketBatchSampler, CharBatchCollator,
                                       CharLevelConfig, CharLevelDataset, train_val_split)
from model_eMG_incremental import (IncrementalEMGModel, IncrementalMorPieceConfig, MorphemeIndexer,
                                   prepare_batch)

logger = logging.getLogger("train_eMG_incremental")


# ============================================================================
# Arguments
# ============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the tokenizer-free eMG-RNN.")

    g = p.add_argument_group("data")
    g.add_argument("--corpus_path", required=True, help="directory of UTF-8 text files")
    g.add_argument("--regimen", default="naturalistic",
                   choices=["naturalistic", "conversational", "sliding_window"])
    g.add_argument("--seq_length", type=int, default=256, help="maximum words per example")
    g.add_argument("--max_word_length", type=int, default=50, help="characters kept per word")
    g.add_argument("--char_vocab_size", type=int, default=256)
    g.add_argument("--keep_case", action="store_true", help="do not lowercase the text")
    g.add_argument("--no_split_punctuation", action="store_true",
                   help="keep punctuation attached to words")
    g.add_argument("--val_fraction", type=float, default=0.01, help="examples held out for validation")
    g.add_argument("--cache_dir", default=None, help="dataset cache (default <corpus_path>/.cache_char)")

    g = p.add_argument_group("model")
    g.add_argument("--hidden_dim", type=int, default=512)
    g.add_argument("--num_layers", type=int, default=2)
    g.add_argument("--dropout", type=float, default=0.1)
    g.add_argument("--use_layer_norm", action="store_true", help="layer-normalise the eMG memory")
    g.add_argument("--move_gate_input", action="store_true", help="let the input drive the Move gate")
    g.add_argument("--gradient_checkpointing", action="store_true")
    g.add_argument("--char_embedding_dim", type=int, default=32)
    g.add_argument("--char_cnn_filters", default="64,128,128,192", help="filters per kernel width")
    g.add_argument("--char_cnn_kernels", default="1,2,3,4", help="kernel widths")
    g.add_argument("--attention_heads", type=int, default=4)
    g.add_argument("--speller_hidden_dim", type=int, default=256)
    g.add_argument("--speller_embedding_dim", type=int, default=64)

    g = p.add_argument_group("Tolerance Principle and vocabulary")
    g.add_argument("--tp_cutoff", type=int, default=100, help="minimum mother-node count for the TP")
    g.add_argument("--min_morpheme_freq", type=int, default=10)
    g.add_argument("--min_suffix_stems", type=int, default=3)
    g.add_argument("--max_vocab_size", type=int, default=50000)
    g.add_argument("--morpheme_embedding_dim", type=int, default=128)
    g.add_argument("--max_morphemes_per_word", type=int, default=8)
    g.add_argument("--tp_loss_weight", type=float, default=0.1, help="weight of the TP boundary loss")
    g.add_argument("--vocab_update_interval", type=int, default=500,
                   help="optimizer steps between vocabulary updates")
    g.add_argument("--vocab_update_lines", type=int, default=2000,
                   help="training examples read by each vocabulary update")

    g = p.add_argument_group("optimisation")
    g.add_argument("--batch_size", type=int, default=32, help="examples per process")
    g.add_argument("--accumulation_steps", type=int, default=4)
    g.add_argument("--epochs", type=int, default=10, help="maximum number of epochs")
    g.add_argument("--max_tokens_exposure", type=float, default=100,
                   help="training budget in millions of predicted words")
    g.add_argument("--learning_rate", type=float, default=2e-3)
    g.add_argument("--min_lr", type=float, default=2e-4)
    g.add_argument("--warmup_iters", type=int, default=0, help="warmup steps (0: 10%% of the schedule)")
    g.add_argument("--weight_decay", type=float, default=1e-6)
    g.add_argument("--beta1", type=float, default=0.9)
    g.add_argument("--beta2", type=float, default=0.999)
    g.add_argument("--grad_clip", type=float, default=0.25)
    g.add_argument("--amp", default="none", choices=["none", "fp16", "bf16"], help="mixed precision")

    g = p.add_argument_group("system")
    g.add_argument("--num_workers", type=int, default=4)
    g.add_argument("--output_dir", default="./checkpoints_incremental")
    g.add_argument("--log_interval", type=int, default=10, help="optimizer steps between log lines")
    g.add_argument("--eval_interval", type=int, default=0,
                   help="optimizer steps between evaluations (0: end of each epoch)")
    g.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


# ============================================================================
# Helpers
# ============================================================================

def setup_distributed():
    """(rank, world_size, local_rank); initialises torch.distributed under torchrun."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        rank, world, local = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        return rank, world, local
    return 0, 1, 0


def barrier():
    if dist.is_initialized():
        dist.barrier()


def get_lr(step: int, warmup: int, learning_rate: float, total: int, min_lr: float) -> float:
    """Linear warmup, then cosine decay from learning_rate to min_lr at step ``total``."""
    if step < warmup:
        return learning_rate * (step + 1) / warmup
    if step >= total:
        return min_lr
    ratio = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (learning_rate - min_lr)


def build_optimizer(model: torch.nn.Module, args, device: torch.device) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for param in model.parameters():
        if param.requires_grad:
            (decay if param.dim() >= 2 else no_decay).append(param)
    groups = [{"params": decay, "weight_decay": args.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    extra = {"fused": True} if device.type == "cuda" else {}
    return torch.optim.AdamW(groups, lr=args.learning_rate, betas=(args.beta1, args.beta2), **extra)


class VocabularySchedule:
    """
    Feeds training examples to the ToleranceTrie in a fixed random order,
    ``lines`` examples per update, until every training example has been
    read once. The order depends only on the seed, so every process builds
    the same vocabulary.
    """

    def __init__(self, dataset: CharLevelDataset, indices: np.ndarray, lines: int, seed: int):
        self.dataset = dataset
        self.order = np.random.default_rng(seed + 1).permutation(indices)
        self.lines = max(1, lines)
        self.position = 0

    @property
    def done(self) -> bool:
        return self.position >= len(self.order)

    def next_words(self) -> Iterator[str]:
        chunk = self.order[self.position:self.position + self.lines]
        self.position += len(chunk)
        for idx in chunk:
            for word in self.dataset.get_words(int(idx)):
                if word not in SPECIAL_WORDS:
                    yield word


def update_vocabulary(model: IncrementalEMGModel, schedule: VocabularySchedule) -> int:
    trie = model.encoder.trie
    for word in schedule.next_words():
        trie.process_word(word)
    return trie.update_vocabulary()


@torch.no_grad()
def evaluate(model: IncrementalEMGModel, loader: Optional[DataLoader], indexer: MorphemeIndexer,
             device: torch.device, autocast) -> Optional[Dict[str, float]]:
    """Validation loss per symbol, word perplexity and TP loss, summed over all processes."""
    if loader is None:
        return None
    was_training = model.training
    model.eval()
    sums = torch.zeros(5, device=device, dtype=torch.float64)
    for batch in loader:
        inputs = prepare_batch(batch, indexer, device)
        with autocast():
            out = model(**inputs)
        sums += torch.stack([out.nll_sum.double(), out.n_symbols.double(), out.n_words.double(),
                             out.tp_loss.double(), torch.ones((), device=device, dtype=torch.float64)])
    if dist.is_initialized():
        dist.all_reduce(sums)
    model.train(was_training)
    nll, symbols, words, tp, batches = sums.tolist()
    if words == 0:
        return None
    return {"loss": nll / symbols, "word_perplexity": math.exp(min(nll / words, 50.0)),
            "tp_loss": tp / max(batches, 1)}


def save_checkpoint(model: IncrementalEMGModel, optimizer, args, directory: str, step: int,
                    epoch: int, tokens: int, metrics: Optional[Dict[str, float]]) -> None:
    summary = {"step": step, "epoch": epoch, "words_seen": tokens, "metrics": metrics}
    model.save_pretrained(directory, extra_config={"training": summary, "args": vars(args)})
    torch.save({"optimizer": optimizer.state_dict(), "step": step, "epoch": epoch, "args": vars(args)},
               os.path.join(directory, "training_state.pt"))
    logger.info("Checkpoint saved to %s", directory)


# ============================================================================
# Training
# ============================================================================

def train(args: argparse.Namespace) -> None:
    rank, world, local = setup_distributed()
    main = rank == 0
    logging.basicConfig(level=logging.INFO if main else logging.WARNING,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    amp = args.amp
    if amp != "none" and device.type != "cuda":
        logger.warning("Mixed precision needs a GPU; training in float32.")
        amp = "none"
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp)
    autocast = (lambda: torch.autocast(device.type, dtype=amp_dtype)) if amp_dtype else nullcontext
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == "fp16"))

    # ------------------------------------------------------------- data
    data_config = CharLevelConfig(max_word_length=args.max_word_length, max_seq_length=args.seq_length,
                                  char_vocab_size=args.char_vocab_size, lowercase=not args.keep_case,
                                  split_punctuation=not args.no_split_punctuation,
                                  cache_dir=args.cache_dir)
    if main:
        dataset = CharLevelDataset(args.corpus_path, data_config, regimen=args.regimen)
    barrier()
    if not main:
        dataset = CharLevelDataset(args.corpus_path, data_config, regimen=args.regimen)
    train_idx, val_idx = train_val_split(dataset, args.val_fraction, args.seed)
    collator = CharBatchCollator(dataset)
    loader_kwargs = dict(collate_fn=collator, num_workers=args.num_workers,
                         pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    train_sampler = BucketBatchSampler(dataset.lengths, args.batch_size, train_idx, shuffle=True,
                                       seed=args.seed, rank=rank, world_size=world, drop_last=True)
    if len(train_sampler) == 0:
        raise ValueError("Not enough training examples for one batch per process; lower --batch_size.")
    train_loader = DataLoader(dataset, batch_sampler=train_sampler, **loader_kwargs)
    val_loader = None
    if len(val_idx) >= world:
        val_sampler = BucketBatchSampler(dataset.lengths, args.batch_size, val_idx, shuffle=False,
                                         rank=rank, world_size=world, drop_last=False)
        val_loader = DataLoader(dataset, batch_sampler=val_sampler, **loader_kwargs)
    else:
        logger.warning("No validation data: the best checkpoint is chosen on training loss.")

    # ------------------------------------------------------------ model
    model_config = IncrementalMorPieceConfig(
        char_vocab_size=args.char_vocab_size, char_embedding_dim=args.char_embedding_dim,
        max_word_length=args.max_word_length, lowercase=not args.keep_case,
        split_punctuation=not args.no_split_punctuation,
        char_cnn_filters=tuple(int(x) for x in args.char_cnn_filters.split(",")),
        char_cnn_kernels=tuple(int(x) for x in args.char_cnn_kernels.split(",")),
        attention_heads=args.attention_heads,
        tp_cutoff=args.tp_cutoff, min_morpheme_freq=args.min_morpheme_freq,
        min_suffix_stems=args.min_suffix_stems, max_vocab_size=args.max_vocab_size,
        morpheme_embedding_dim=args.morpheme_embedding_dim,
        max_morphemes_per_word=args.max_morphemes_per_word,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout,
        use_layer_norm=args.use_layer_norm, move_gate_input=args.move_gate_input,
        use_gradient_checkpointing=args.gradient_checkpointing,
        speller_hidden_dim=args.speller_hidden_dim, speller_embedding_dim=args.speller_embedding_dim,
        tp_loss_weight=args.tp_loss_weight,
    )
    raw_model = IncrementalEMGModel(model_config).to(device)
    indexer = MorphemeIndexer(raw_model.encoder.trie, dataset.types, args.max_morphemes_per_word,
                              args.max_word_length)
    schedule = VocabularySchedule(dataset, train_idx, args.vocab_update_lines, args.seed)
    model = DDP(raw_model, device_ids=[local] if device.type == "cuda" else None) if world > 1 else raw_model
    optimizer = build_optimizer(raw_model, args, device)

    # --------------------------------------------------------- schedule
    batches_per_epoch = len(train_sampler)
    epoch_fraction = batches_per_epoch * args.batch_size * world / max(len(train_idx), 1)
    words_per_epoch = (dataset.lengths[train_idx] - 1).sum() * epoch_fraction
    epochs_needed = min(args.epochs, args.max_tokens_exposure * 1e6 / max(words_per_epoch, 1))
    total_steps = max(1, math.ceil(batches_per_epoch * epochs_needed / args.accumulation_steps))
    warmup = args.warmup_iters if args.warmup_iters > 0 else max(1, total_steps // 10)

    os.makedirs(args.output_dir, exist_ok=True)
    growth_log = os.path.join(args.output_dir, "vocab_growth.tsv")
    added = update_vocabulary(raw_model, schedule)
    if main:
        with open(growth_log, "w", encoding="utf-8") as f:
            f.write("step\twords_read\tvocab_size\tadded\n")
            f.write(f"0\t{raw_model.encoder.trie.total_words}\t{raw_model.get_vocab_size()}\t{added}\n")
        logger.info("eMG-RNN (tokenizer-free): %s parameters, %d process(es), device %s",
                    f"{raw_model.get_num_params():,}", world, device)
        logger.info("%d training / %d validation examples; %.2fM predicted words per epoch",
                    len(train_idx), len(val_idx), words_per_epoch / 1e6)
        logger.info("%d optimizer steps (%.2f epochs), warmup %d, effective batch %d examples",
                    total_steps, epochs_needed, warmup, args.batch_size * world * args.accumulation_steps)
        logger.info("Initial vocabulary: %d tokens", raw_model.get_vocab_size())

    # ------------------------------------------------------------- loop
    step, micro, epoch = 0, 0, 0
    best = float("inf")
    stats = torch.zeros(5, device=device, dtype=torch.float64)   # nll, symbols, words, tp, batches
    words_seen = 0
    train_loss = float("inf")
    last_time = time.time()
    metrics = None
    done = False
    model.train()
    max_epochs = max(1, math.ceil(epochs_needed))

    def checkpoint_if_best(current: Optional[Dict[str, float]], fallback: float) -> None:
        nonlocal best
        value = current["loss"] if current else fallback
        if main and value < best:
            best = value
            save_checkpoint(raw_model, optimizer, args, os.path.join(args.output_dir, "best_model"),
                            step, epoch, words_seen, current)

    def optimizer_step() -> None:
        nonlocal step
        lr = get_lr(step, warmup, args.learning_rate, total_steps, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1

    for epoch in range(max_epochs):
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            inputs = prepare_batch(batch, indexer, device)
            last_micro = (micro + 1) % args.accumulation_steps == 0
            sync = model.no_sync() if (world > 1 and not last_micro) else nullcontext()
            with sync:
                with autocast():
                    out = model(**inputs)
                scaler.scale(out.loss / args.accumulation_steps).backward()
            stats += torch.stack([out.nll_sum.detach().double(), out.n_symbols.double(),
                                  out.n_words.double(), out.tp_loss.detach().double(),
                                  torch.ones((), device=device, dtype=torch.float64)])
            micro += 1
            if not last_micro:
                continue
            optimizer_step()

            if step % args.vocab_update_interval == 0 and not schedule.done:
                added = update_vocabulary(raw_model, schedule)
                if main:
                    with open(growth_log, "a", encoding="utf-8") as f:
                        f.write(f"{step}\t{raw_model.encoder.trie.total_words}\t"
                                f"{raw_model.get_vocab_size()}\t{added}\n")
                    if added or schedule.done:
                        logger.info("Vocabulary update: +%d tokens, %d total%s", added,
                                    raw_model.get_vocab_size(), " (last update)" if schedule.done else "")

            if step % args.log_interval == 0 or step == total_steps:
                if world > 1:
                    dist.all_reduce(stats)
                nll, symbols, words, tp, batches = stats.tolist()
                words_seen += int(words)
                elapsed = time.time() - last_time
                if main:
                    logger.info("step %d/%d | epoch %d | loss %.4f | TP loss %.4f | lr %.2e | "
                                "%.2fM words | vocab %d | %.0f words/s",
                                step, total_steps, epoch + 1, nll / max(symbols, 1), tp / max(batches, 1),
                                get_lr(step, warmup, args.learning_rate, total_steps, args.min_lr),
                                words_seen / 1e6, raw_model.get_vocab_size(), words / max(elapsed, 1e-9))
                train_loss = nll / max(symbols, 1)
                stats.zero_()
                last_time = time.time()

            if args.eval_interval and step % args.eval_interval == 0:
                metrics = evaluate(raw_model, val_loader, indexer, device, autocast)
                if main and metrics:
                    logger.info("validation | loss %.4f | word perplexity %.2f | TP loss %.4f",
                                metrics["loss"], metrics["word_perplexity"], metrics["tp_loss"])
                checkpoint_if_best(metrics, train_loss)

            if step >= total_steps:
                done = True
                break
        if done:
            break
        if not args.eval_interval:
            metrics = evaluate(raw_model, val_loader, indexer, device, autocast)
            if main and metrics:
                logger.info("epoch %d | validation loss %.4f | word perplexity %.2f | TP loss %.4f",
                            epoch + 1, metrics["loss"], metrics["word_perplexity"], metrics["tp_loss"])
            checkpoint_if_best(metrics, train_loss)

    if micro % args.accumulation_steps:
        optimizer_step()      # apply the gradients of a final partial accumulation
    if world > 1:
        dist.all_reduce(stats)
    words_seen += int(stats[2].item())

    metrics = evaluate(raw_model, val_loader, indexer, device, autocast)
    if main:
        if metrics:
            logger.info("final | validation loss %.4f | word perplexity %.2f",
                        metrics["loss"], metrics["word_perplexity"])
            checkpoint_if_best(metrics, float("inf"))
        save_checkpoint(raw_model, optimizer, args, os.path.join(args.output_dir, "final_model"),
                        step, epoch, words_seen, metrics)
        logger.info("Training complete: %d steps, %.2fM words, vocabulary %d tokens",
                    step, words_seen / 1e6, raw_model.get_vocab_size())
    if dist.is_initialized():
        dist.destroy_process_group()


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    try:
        train(args)
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        if dist.is_initialized():
            dist.destroy_process_group()
        sys.exit(1)


if __name__ == "__main__":
    main()
