# eMG-RNN

**expectation-based Minimalist Grammar Recurrent Neural Network**

eMG-RNN is a recurrent language model whose cell implements two structure-building operations of expectation-based Minimalist Grammars ([Chesi 2021](https://arxiv.org/abs/2109.13871)). **Merge** adds the current input, gated by the previous state, to a bounded context memory. **Move** decides how much of that memory reaches the output, as opposed to passing the input through. The architecture was introduced in [Chesi et al. (2024), *Different Ways to Forget: Linguistic Gates in Recurrent Neural Networks*](https://aclanthology.org/2024.conll-babylm.9/), at the 2nd BabyLM Challenge.

This repository provides two models built on the same eMG core:

- **`EMGModel`**, a token-level language model for any tokenizer. It offers optional character-level input encoders informed by Yang's Tolerance Principle, and a Hugging Face wrapper that works with `save_pretrained`, `from_pretrained` and lm-evaluation-harness (BLiMP).
- **`IncrementalEMGModel`**, a tokenizer-free model that reads words as characters and spells out the next word character by character. It assigns a probability to any sentence, and grows a morpheme vocabulary during training following [MorPiece](https://github.com/cristianochesi/morpiece).

## Contents

| File | What it contains |
|---|---|
| `model_eMG.py` | The eMG cell and layers, `EMGModel` with its input encoders (word embeddings, CharCNN, TolerancePrincipleCNN, NeuralMorPiece), the auxiliary losses, and the Hugging Face wrapper `EMGLanguageModel`. |
| `model_eMG_incremental.py` | `IncrementalEMGModel`: the Tolerance Principle tries (`ToleranceTrie`), the character encoder with morpheme route, the word speller, scoring and minimal-pair evaluation. |
| `data_preparation_charwise.py` | The character inventory, text preprocessing, a cached type-indexed dataset, a length-bucketed sampler and a dynamic-padding collator. |
| `train_eMG_incremental.py` | The training script for `IncrementalEMGModel`: single GPU or multi-GPU, mixed precision, validation and checkpointing. |
| `stat_mor_learning.py` | A standalone statistical morpheme learner with an Italian demo. It uses only the Python standard library. |
| `lm_eval_emg.py` | lm-evaluation-harness models: `emg_incremental` for the tokenizer-free model and `emg_lm` for checkpoints of the 2024 BabyLM release. |
| `LICENSE`, `CITATION.cff` | MIT License and citation metadata. |

## The eMG cell

For a layer input $x_t$ and previous state $(m_{t-1}, c_{t-1})$:

```math
\begin{aligned}
\tilde{x}_t &= W_x x_t + b_x \\
\rho_t &= \sigma(W_\rho\, m_{t-1} + b_\rho) && \text{retain gate} \\
\mu_t &= \sigma(W_\mu\, m_{t-1} + b_\mu + U x_t) && \text{move gate} \\
c_t &= \tanh(c_{t-1} + \rho_t \odot \tilde{x}_t) && \text{Merge into the context memory} \\
m_t &= (1 - \mu_t) \odot \tilde{x}_t + \mu_t \odot c_t && \text{Move memory into the output}
\end{aligned}
```

$m_t$ goes to the next layer and to the output, and $c_t$ is the context memory. Both gates lie in (0, 1), so $m_t$ interpolates between the projected input and the memory, and $c_t$ stays in (−1, 1). Weights use Xavier initialisation and biases start at zero.

- The term $U x_t$ exists only with `move_gate_input=True`. Without it (the default), the cell is the one of Chesi et al. (2024), where both gates depend on the previous state only.
- With `use_layer_norm=True`, the memory pre-activation is layer-normalised: $c_t = \tanh(\mathrm{LN}(c_{t-1} + \rho_t \odot \tilde{x}_t))$.
- A layer has one input block and two state blocks, against four of each in an LSTM of the same width.

`OptimizedEMG` runs the layers one at a time over the whole sequence. Each layer projects its entire input with a single matrix multiplication and then runs the recurrence, so only the state-dependent part stays inside the time loop. Dropout is applied to the embeddings, between layers and to the final output, one mask per sequence; it is never applied inside the recurrence. With `use_gradient_checkpointing=True`, activations are recomputed in chunks of `checkpoint_chunk` time steps.

## Installation

Requirements: Python 3.9 or later and PyTorch 2.3 or later. The code was tested on Linux with Python 3.13, PyTorch 2.14.1, transformers 4.46.3 and 5.19.0, and lm-evaluation-harness 0.4.13.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU training, first install the PyTorch build that matches your CUDA driver ([pytorch.org/get-started](https://pytorch.org/get-started/locally/)), then the requirements. For evaluation, also install lm-evaluation-harness with `pip install "lm_eval[hf]"` (the `hf` extra is needed only for its built-in `hf` model).

## Quick checks

Each module has a self-test:

```bash
python model_eMG.py                  # every encoder, generation, Hugging Face save and reload
python model_eMG_incremental.py      # tolerance tries, vocabulary growth, scoring, save and reload
python data_preparation_charwise.py  # dataset cache, train/validation split, batching
python stat_mor_learning.py          # statistical morpheme learner demo
```

## Token-level model: `EMGModel`

### Training with your own loop

`EMGModel.forward(idx, targets)` does not shift the targets: `targets[:, t]` is the token that follows `idx[:, t]`, and `-100` marks positions to ignore.

```python
import torch
from model_eMG import EMGConfig, EMGModel

config = EMGConfig(vocab_size=32000, embedding_dim=512, hidden_dim=512, num_layers=2)
model = EMGModel(config)
optimizer = model.configure_optimizers(weight_decay=1e-6, learning_rate=2e-3,
                                       betas=(0.9, 0.999), device_type="cuda")

tokens = torch.randint(0, config.vocab_size, (8, 65))   # (batch, seq_len + 1)
logits, loss, hidden = model(tokens[:, :-1], targets=tokens[:, 1:])
loss.backward()
optimizer.step()
```

For truncated backpropagation through time, pass `EMGModel.detach_hidden(hidden)` to the next call. `model.generate(idx, max_new_tokens)` reads the prompt once and then feeds each new token with the carried state.

### Main options

| `EMGConfig` field | Default | Effect |
|---|---|---|
| `hidden_dim`, `embedding_dim`, `num_layers` | 512, 512, 2 | model size |
| `dropout`, `locked_dropout` | 0.1, True | dropout on embeddings, between layers and on the output |
| `move_gate_input` | False | add the input term $U x_t$ to the Move gate |
| `use_layer_norm` | False | layer-normalise the memory pre-activation |
| `use_gradient_checkpointing`, `checkpoint_chunk` | False, 32 | recompute activations in chunks of time steps |
| `tie_word_embeddings` | True | share input and output matrices (word embeddings only) |
| `use_char_cnn` | False | character CNN encoder |
| `use_tolerance_cnn` | False | lexical + compositional encoder with a tolerance gate |
| `use_neural_morpiece` | False | encoder with learned morpheme boundaries |
| `tolerance_reg_weight` | 0.01 | weight of the auxiliary losses of the character encoders |
| `debug_checks` | False | check token IDs on every step (slower) |

### Character-level encoders

- **CharCNN** builds each token from its characters with convolutions of several widths, max-pooled over the windows that start inside the word.
- **TolerancePrincipleCNN** has two routes. A word embedding stores tokens whole (exceptions such as *sono*, *ho*, *vado*), and a character CNN composes them (regular forms such as *arrivano*). Characters carry their position from the start and from the end of the word, so word endings line up across words of different lengths. A tolerance gate blends the two routes. A regulariser pushes the proportion of word types routed lexically toward the Tolerance Principle prediction 1/ln V (exceptions e ≤ N/ln N), and rewards clear decisions.
- **NeuralMorPiece** is a differentiable version of MorPiece's two tries. A bidirectional character GRU reads the stem side and the ending side, and a predictor gives the probability of a morpheme boundary after each character. Characters attend mostly within their own soft segment, and an optional lexical bypass stores irregular forms whole. A regulariser pushes the boundary probability over word-internal positions toward `morpiece_split_rate`.

At most one encoder is active; if several are set, NeuralMorPiece takes priority over TolerancePrincipleCNN, which takes priority over CharCNN. Character encoders need the characters of every token. Register a token-to-character table once, and it is saved with the model and used for training, generation and evaluation:

```python
from transformers import AutoTokenizer
from data_preparation_charwise import CharacterVocab, CharLevelConfig
from model_eMG import EMGConfig, EMGModel, build_char_lookup

tokenizer = AutoTokenizer.from_pretrained("your-tokenizer")
config = EMGConfig(vocab_size=len(tokenizer), hidden_dim=512, use_tolerance_cnn=True)
model = EMGModel(config)

chars = CharacterVocab(CharLevelConfig(max_word_length=config.max_word_length))
strings = [tokenizer.decode([i]).strip() for i in range(len(tokenizer))]
model.set_char_lookup(build_char_lookup(strings, chars.encode_word, config.max_word_length))
```

`model.analyze_gate_values(idx, tokenizer=tokenizer)` reports how often tokens take the lexical route and lists the most lexical and most compositional ones. `model.neural_morpiece.analyze_segmentation(char_ids, chars_to_str=chars.decode_char)` shows the boundaries that NeuralMorPiece has learned.

### Hugging Face format and BLiMP

`EMGLanguageModel` wraps `EMGModel` for the Hugging Face ecosystem. `labels` follow the Hugging Face convention (they are the input IDs, shifted inside the model), `attention_mask` marks right padding, and the output has `.loss` and `.logits`. A saved model includes its source file, so `AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)` loads it anywhere; pass the same flag to `AutoTokenizer` when the tokenizer is saved in the same folder. After `import model_eMG`, the flag is not needed.

```python
from model_eMG import EMGHFConfig, EMGLanguageModel

hf_model = EMGLanguageModel(EMGHFConfig.from_emg_config(config))
hf_model.model.load_state_dict(model.state_dict())     # weights of a trained EMGModel
hf_model.save_pretrained("emg-base")
tokenizer.save_pretrained("emg-base")
```

Evaluate on BLiMP with lm-evaluation-harness:

```bash
lm_eval --model hf \
  --model_args pretrained=./emg-base,trust_remote_code=True \
  --tasks blimp --batch_size 32
```

The tokenizer must define a beginning- or end-of-sentence token: lm-evaluation-harness uses it as the context of BLiMP sentences, which have no prefix. Without lm-evaluation-harness, `model.sentence_logprobs(input_ids, attention_mask)` returns the summed log-probability of each right-padded sequence; the model prefers the grammatical member of a minimal pair when it gets the higher score.

## Tokenizer-free model: `IncrementalEMGModel`

### How it works

1. **Character encoder.** Every word is read as characters with their positions from both ends. Convolutions give per-character features, a predictor gives the probability of a morpheme boundary after each character, and segment-aware attention and max-pooling produce a word vector.
2. **Morpheme route.** The word is segmented into pieces of the growing vocabulary (word, stem, `##`ending). Their embeddings are averaged and added to the word vector through a learned gate.
3. **eMG layers.** The same recurrent core as `EMGModel` reads the sequence of word vectors.
4. **Word speller.** The next word is predicted character by character, ending with an end-of-word symbol: $\log P(w \mid h) = \sum_k \log P(c_k \mid c_{<k}, h) + \log P(\langle\mathrm{eow}\rangle \mid w, h)$. Every possible word gets a probability, so whole sentences can be scored.
5. **Tolerance Principle supervision.** The boundary predictor is trained to reproduce the boundaries that the Tolerance Principle places in each word, wherever the counts are large enough to decide.

The vocabulary comes from `ToleranceTrie`, which follows MorPiece. Every word is counted in a root trie (left to right) and an inflection trie (right to left). With $f(x)$ the count of the node reached by string $x$, a boundary separates the stem $w_{<s}$ from the ending $w_{\geq s}$ when the Tolerance Principle holds on both sides:

```math
\mathrm{TP}(m, d) \iff m > \text{cutoff},\; d \neq m,\; d > \frac{m}{\ln m}
\qquad
\text{root: } m = f(w_{<s-1}),\ d = f(w_{<s})
\qquad
\text{inflection: } m = f(w_{> s}),\ d = f(w_{\geq s})
```

Stems and endings that recur at least `min_morpheme_freq` times enter the vocabulary, endings only if they attach to at least `min_suffix_stems` different stems; frequent words and single characters are added too. Token IDs never change once given, and the morpheme embedding table has a fixed size (`max_vocab_size`).

### Data format

`--corpus_path` points to a directory of UTF-8 plain-text files:

- One sentence or utterance per line. Files whose names start with `.` are skipped.
- Text is normalised (Unicode NFC) and lowercased, punctuation is separated from words (`--keep_case` and `--no_split_punctuation` turn this off), and words are split on whitespace.
- Every example is wrapped in `<s> … </s>`, so the first word of a sentence is predicted too.
- `--regimen naturalistic` (default) makes each line one example, `conversational` pairs consecutive lines, and `sliding_window` concatenates the corpus and cuts windows of `--seq_length` words with 50% overlap. Lines longer than `--seq_length` words are split into several examples.

The processed corpus is cached in `<corpus_path>/.cache_char/`, or in `~/.cache/emg_rnn/` when the corpus directory is read-only (`--cache_dir` sets another place). The cache key covers file names, sizes and modification times plus the processing settings, so a changed corpus is processed again.

### Training

```bash
python train_eMG_incremental.py --corpus_path ./data --output_dir ./checkpoints_incremental
```

On several GPUs, with bfloat16 mixed precision:

```bash
torchrun --nproc_per_node=4 train_eMG_incremental.py --corpus_path ./data --amp bf16
```

On a small corpus (a few thousand lines), a CPU run to check the setup takes under a minute:

```bash
python train_eMG_incremental.py --corpus_path ./data --output_dir ./smoke_test \
  --hidden_dim 64 --char_embedding_dim 16 --char_cnn_filters 16,32,32,48 --max_word_length 20 \
  --speller_hidden_dim 64 --speller_embedding_dim 16 --morpheme_embedding_dim 16 \
  --batch_size 16 --epochs 1 --num_workers 0 --max_tokens_exposure 0.05 \
  --tp_cutoff 50 --min_morpheme_freq 5 --vocab_update_interval 20 --vocab_update_lines 500
```

The learning-rate schedule (linear warmup, cosine decay to `--min_lr`) spans the optimizer steps that the word budget `--max_tokens_exposure` actually requires, measured on the real, unpadded data. The vocabulary is updated before training and every `--vocab_update_interval` steps, each time from the next `--vocab_update_lines` training examples, in a fixed random order shared by all processes. Updates stop after one pass over the training data.

### Main arguments

| Group | Argument | Default | Meaning |
|---|---|---|---|
| Data | `--regimen` | `naturalistic` | how lines become examples |
| | `--seq_length` | 256 | maximum words per example |
| | `--max_word_length` | 50 | characters kept per word |
| | `--val_fraction` | 0.01 | examples held out for validation |
| Model | `--hidden_dim`, `--num_layers` | 512, 2 | eMG size |
| | `--dropout` | 0.1 | dropout rate |
| | `--move_gate_input`, `--use_layer_norm` | off | eMG cell options (see above) |
| | `--gradient_checkpointing` | off | recompute activations to save memory |
| | `--char_embedding_dim` | 32 | character embedding size |
| | `--char_cnn_filters`, `--char_cnn_kernels` | `64,128,128,192`, `1,2,3,4` | convolution filters per kernel width |
| | `--speller_hidden_dim`, `--speller_embedding_dim` | 256, 64 | word speller size |
| Tolerance Principle | `--tp_cutoff` | 100 | minimum mother-node count for the TP |
| | `--min_morpheme_freq` | 10 | minimum count for the vocabulary |
| | `--min_suffix_stems` | 3 | distinct stems an ending needs |
| | `--max_vocab_size`, `--morpheme_embedding_dim` | 50000, 128 | morpheme embedding table |
| | `--tp_loss_weight` | 0.1 | weight of the boundary supervision |
| | `--vocab_update_interval` | 500 | optimizer steps between vocabulary updates |
| | `--vocab_update_lines` | 2000 | examples read by each update |
| Optimisation | `--batch_size`, `--accumulation_steps` | 32, 4 | examples per process, gradient accumulation |
| | `--epochs` | 10 | maximum epochs |
| | `--max_tokens_exposure` | 100 | budget, in millions of predicted words |
| | `--learning_rate`, `--min_lr` | 2e-3, 2e-4 | peak and final learning rate |
| | `--warmup_iters` | 0 | warmup steps (0 means 10% of the schedule) |
| | `--weight_decay`, `--grad_clip` | 1e-6, 0.25 | AdamW weight decay, gradient-norm clipping |
| | `--amp` | `none` | mixed precision: `fp16`, `bf16` or `none` |
| System | `--num_workers` | 4 | data-loading workers |
| | `--log_interval`, `--eval_interval` | 10, 0 | steps between log lines and evaluations (0: each epoch) |
| | `--cache_dir`, `--output_dir`, `--seed` | | cache location, checkpoints, random seed |

Run `python train_eMG_incremental.py --help` for the full list.

### Outputs

```
checkpoints_incremental/
├── best_model/              # lowest validation loss
│   ├── pytorch_model.pt     # model weights
│   ├── vocabulary.json      # morpheme vocabulary and counts
│   ├── tries.json.gz        # root and inflection tries
│   ├── config.json          # model configuration, training arguments and metrics
│   └── training_state.pt    # optimizer state
├── final_model/             # same layout, at the end of training
└── vocab_growth.tsv         # vocabulary size after every update
```

The training log reports the loss per predicted character (natural log), the boundary-supervision loss, the vocabulary size and the throughput. Each evaluation also reports the word perplexity.

### Scoring sentences and minimal pairs

```python
import json
from model_eMG_incremental import IncrementalEMGModel, minimal_pair_accuracy

model = IncrementalEMGModel.from_pretrained("checkpoints_incremental/best_model")

model.score_sentences(["I ragazzi mangiano la pasta.", "I ragazzi mangia la pasta."])

# a BLiMP-style file with one {"sentence_good": ..., "sentence_bad": ...} object per line
with open("blimp/anaphor_gender_agreement.jsonl") as f:
    pairs = [(d["sentence_good"], d["sentence_bad"]) for d in map(json.loads, f)]
print(minimal_pair_accuracy(model, pairs)["accuracy"])
```

Sentences go through the same preprocessing as the training data. A sentence's score is $\sum_i \log P(w_i \mid \langle s\rangle\, w_1 \ldots w_{i-1})$; with `include_eos=True` the probability of `</s>` is added.

## Evaluation with lm-evaluation-harness

Every eMG model gives lm-evaluation-harness the summed log-probability of a continuation given its context, computed by next-token (or next-word) prediction as in training. On BLiMP a model is right on a pair when the grammatical sentence gets the higher sum.

| Model | lm-eval model | Model arguments |
|---|---|---|
| `EMGModel` saved with `EMGLanguageModel.save_pretrained` | `hf` (built in) | `pretrained=PATH,trust_remote_code=True` |
| `IncrementalEMGModel` checkpoint | `emg_incremental` | `pretrained=PATH`; optional `include_eos=True` |
| Checkpoint of the 2024 BabyLM release | `emg_lm` | `pretrained=PATH`; optional `checkpoint`, `normalize`, `prefix_token` |

`lm_eval_emg.py` registers the two `emg_*` models and runs the lm-evaluation-harness command line:

```bash
python lm_eval_emg.py --model emg_incremental \
  --model_args pretrained=checkpoints_incremental/best_model --tasks blimp --batch_size 32

python lm_eval_emg.py --model emg_lm --model_args pretrained=path/to/2024_model --tasks blimp
```

The same models are available from Python after importing the module:

```python
import lm_eval
import lm_eval_emg  # registers emg_lm and emg_incremental

results = lm_eval.simple_evaluate(model="emg_incremental",
                                  model_args="pretrained=checkpoints_incremental/best_model",
                                  tasks=["blimp"])
```

About `emg_lm`:

- The folder holds `model.pt` and the tokenizer files. `model.pt` may be a plain state dict, one saved from DataParallel or DistributedDataParallel, or the 2024 training-script dict with a `state_dict` key. Vocabulary size, dimensions, number of layers and cell variant (`model_eMG_RNN.py` or `model_eMG_RNN_base.py`) are read from it.
- `normalize=True` (default) applies the 2024 preprocessing: lowercasing, spaces around `. , ! ?`, and removal of the speaker labels `a:`, `b:`, `david:`, `sarah:`.
- The 2024 models were trained on lines without a start symbol, so they never learned to predict the first token of a line. With an empty context, as in BLiMP, the first token therefore serves as context and the following tokens are scored; `prefix_token=TOKEN` prepends a token of the vocabulary instead.
- `generate_until` tasks use greedy decoding with the recurrent state carried over. `emg_incremental` supports log-likelihood tasks only.

## Statistical morpheme learner

`stat_mor_learning.py` models early morpheme discovery from word frequencies alone. For every prefix it counts the characters that follow, and for every suffix the characters that precede, with the word edges as an extra symbol. A position is a boundary when both sides are open choices by the Tolerance Principle: the rule "this prefix continues with its usual next character" has more than N/ln N exceptions, and likewise for the characters before the ending. Words are then split recursively at their best-scoring boundaries. The demo learns from Italian verb paradigms and segments, for example, *bellissimo* as *bell-issim-o* and *parlandolo* as *parla-ndo-l-o*.

```bash
python stat_mor_learning.py
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{chesi-etal-2024-different,
    title = "Different Ways to Forget: Linguistic Gates in Recurrent Neural Networks",
    author = "Chesi, Cristiano  and
      Bressan, Veronica  and
      Barbini, Matilde  and
      Fusco, Achille  and
      Bianchessi, Maria Letizia Piccini  and
      Neri, Sofia  and
      Rossi, Sarah  and
      Sgrizzi, Tommaso",
    booktitle = "The 2nd BabyLM Challenge at the 28th Conference on Computational Natural Language Learning",
    month = nov,
    year = "2024",
    address = "Miami, FL, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.conll-babylm.9/",
    pages = "106--117"
}
```

## License

eMG-RNN is released under the [MIT License](LICENSE): anyone may use, copy, modify and redistribute it, also commercially, provided that the copyright and license notice is kept. If you use it in research, please cite the paper above; GitHub's "Cite this repository" button reads it from `CITATION.cff`.

## Related work

- Chesi, C. (2021). Expectation-based Minimalist Grammars. [arXiv:2109.13871](https://arxiv.org/abs/2109.13871).
- Yang, C. (2016). *The Price of Linguistic Productivity*. MIT Press.
- MorPiece tokenizer: [github.com/cristianochesi/morpiece](https://github.com/cristianochesi/morpiece)
- BabyLM 2024 code for the original paper: [github.com/cristianochesi/babylm-2024](https://github.com/cristianochesi/babylm-2024)

## Acknowledgements

Code by Cristiano Chesi (IUSS Pavia, NeTS Lab). Development, debugging and optimization were assisted by Claude (Anthropic).
