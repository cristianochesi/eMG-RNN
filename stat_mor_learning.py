#!/usr/bin/env python3
"""
Statistical morpheme learner.

A purely statistical model of early morpheme discovery, in the spirit of
morphological bootstrapping in child language acquisition: no rules and no
language-specific knowledge, only counts over the words heard.

The learner tracks, for every prefix of every word, the characters that can
follow it, and for every suffix the characters that can precede it; the
word edges count as a boundary symbol "#". A position inside a word is a
candidate morpheme boundary when what follows the stem and what precedes
the ending are both open choices, by the Tolerance Principle (Yang 2016):

    A rule over N items is productive iff its exceptions e <= N / ln N.

On the stem side, take the rule "this prefix continues with its most
frequent next character": N is the count of the prefix and e the count of
all other continuations. If e > N / ln N the continuation is not
rule-governed, so the prefix is a free-standing unit. The ending side is
tested in the same way on the characters that precede it. Each side must
also be seen at least ``min_frequency`` times with at least
``min_branching`` different neighbours.

For "cercarlo": "cerca-" combines with -re, -ndo, -to, -rlo...; "-lo" combines
with cercar-, veder-, sentir-... Each candidate boundary is scored from the
entropy of the continuations on both sides and from its frequency, always
with statistics anchored at the word edges. Segmentation then splits the
word recursively at the highest-scoring boundaries, keeping every piece at
least ``min_piece_len`` characters long.

Counts are token counts, as MorPiece recommends for Italian. (The
ToleranceTrie of model_eMG_incremental.py applies the TP in MorPiece's
form instead, to node frequencies along the two tries.)

References
    Yang, C. (2016). The Price of Linguistic Productivity. MIT Press.
    Saffran, J. R., Aslin, R. N., & Newport, E. L. (1996). Statistical
        learning by 8-month-old infants. Science, 274(5294), 1926-1928.
    MorPiece: https://github.com/cristianochesi/morpiece

Author: Cristiano Chesi (IUSS Pavia, NeTS Lab).
Development, debugging and optimization assisted by Claude (Anthropic).
Released under the MIT License (see LICENSE).
"""

from collections import defaultdict
from math import log, log2
from typing import Dict, List, Optional, Set, Tuple

EDGE = "#"


class ChildMorphemeLearner:
    """
    Learns morpheme boundaries from word frequencies alone.

    min_frequency: minimum count N of a piece before the Tolerance Principle
        is applied to it.
    min_branching: minimum number of different continuations (stem side) or
        predecessors (ending side).
    min_piece_len: minimum length of a morpheme.
    min_confidence: minimum boundary score used by segment(),
        segment_detailed() and show_analysis().
    """

    def __init__(self, min_frequency: int = 3, min_branching: int = 2, min_piece_len: int = 1,
                 min_confidence: float = 0.01):
        self.min_frequency = min_frequency
        self.min_branching = min_branching
        self.min_piece_len = min_piece_len
        self.min_confidence = min_confidence

        # prefix_continuations["cerc"] = {"a": 50, "h": 5}; "#" marks the end of the word
        self.prefix_continuations: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # suffix_predecessors["lo"] = {"r": 100, "o": 30}; "#" marks the start of the word
        self.suffix_predecessors: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.prefix_freq: Dict[str, int] = defaultdict(int)
        self.suffix_freq: Dict[str, int] = defaultdict(int)
        self.word_freq: Dict[str, int] = defaultdict(int)
        self.vocabulary: Set[str] = set()
        self.total_tokens = 0

    # ------------------------------------------------------------ learning

    def observe(self, word: str) -> None:
        """Count one occurrence of a word (all prefixes and suffixes, whole word included)."""
        word = word.lower().strip()
        if len(word) < 2:
            return
        n = len(word)
        self.total_tokens += 1
        self.vocabulary.add(word)
        self.word_freq[word] += 1
        for i in range(1, n + 1):
            prefix = word[:i]
            self.prefix_continuations[prefix][word[i] if i < n else EDGE] += 1
            self.prefix_freq[prefix] += 1
        for i in range(n):
            suffix = word[i:]
            self.suffix_predecessors[suffix][word[i - 1] if i > 0 else EDGE] += 1
            self.suffix_freq[suffix] += 1

    def observe_text(self, text: str) -> int:
        """Observe every word of a text (letters only); returns the number of words."""
        count = 0
        for token in text.split():
            word = "".join(c for c in token.lower() if c.isalpha())
            if word:
                self.observe(word)
                count += 1
        return count

    # ------------------------------------------------------------- scoring

    @staticmethod
    def _entropy(distribution: Dict[str, int]) -> float:
        """Entropy in bits (higher = less predictable continuation)."""
        total = sum(distribution.values())
        if total == 0:
            return 0.0
        return -sum((c / total) * log2(c / total) for c in distribution.values() if c > 0)

    def _is_productive(self, freq: int, branching: int, exceptions: int) -> bool:
        """
        True when the piece is a free-standing unit: seen at least
        min_frequency times, with at least min_branching different
        neighbours, and no productive continuation rule: exceptions > N / ln N.
        """
        if freq < max(self.min_frequency, 2) or branching < self.min_branching:
            return False
        return exceptions > freq / log(freq)

    def _sides(self, word: str, pos: int) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """(N, branching, exceptions) of the stem side and of the ending side at ``pos``."""
        after = self.prefix_continuations.get(word[:pos], {})
        before = self.suffix_predecessors.get(word[pos:], {})
        stem_n, ending_n = sum(after.values()), sum(before.values())
        stem = (stem_n, len(after), stem_n - max(after.values(), default=0))
        ending = (ending_n, len(before), ending_n - max(before.values(), default=0))
        return stem, ending

    def _split_confidence(self, word: str, pos: int) -> float:
        """
        Score of a boundary between word[:pos] and word[pos:] (0 if either
        side is not productive): frequency of the rarer side relative to all
        tokens, plus the mean entropy of the continuations on the two sides.
        """
        if pos < self.min_piece_len or pos > len(word) - self.min_piece_len:
            return 0.0
        stem, ending = self._sides(word, pos)
        if not (self._is_productive(*stem) and self._is_productive(*ending)):
            return 0.0
        prefix_entropy = self._entropy(self.prefix_continuations.get(word[:pos], {}))
        suffix_entropy = self._entropy(self.suffix_predecessors.get(word[pos:], {}))
        freq_score = min(stem[0], ending[0]) / max(self.total_tokens, 1)
        return min(1.0, freq_score * 10 + (prefix_entropy + suffix_entropy) / 2 * 0.1)

    def boundary_scores(self, word: str) -> Dict[int, float]:
        """Score of every inner position of the word, with word-anchored statistics."""
        word = word.lower().strip()
        return {pos: self._split_confidence(word, pos) for pos in range(1, len(word))}

    def find_best_split(self, word: str) -> Tuple[int, float]:
        """The single best boundary of a word: (position, score), or (-1, 0.0)."""
        scores = self.boundary_scores(word)
        valid = {p: s for p, s in scores.items()
                 if self.min_piece_len <= p <= len(word) - self.min_piece_len and s > 0}
        if not valid:
            return -1, 0.0
        best = max(valid, key=lambda p: (valid[p], -p))
        return best, valid[best]

    # --------------------------------------------------------- segmentation

    def boundaries(self, word: str, min_confidence: Optional[float] = None) -> List[int]:
        """
        Boundary positions chosen by recursive binary splitting: split the
        word at its best-scoring boundary, then each half at its own best
        boundary, and so on. Every boundary keeps the score it has in the
        whole word, so pieces inside the word are never judged with
        word-edge statistics that do not apply to them.
        """
        word = word.lower().strip()
        threshold = self.min_confidence if min_confidence is None else min_confidence
        scores = {p: s for p, s in self.boundary_scores(word).items() if s >= threshold and s > 0}
        chosen: List[int] = []

        def split(lo: int, hi: int) -> None:
            candidates = [p for p in scores if lo + self.min_piece_len <= p <= hi - self.min_piece_len]
            if not candidates:
                return
            best = max(candidates, key=lambda p: (scores[p], -p))
            chosen.append(best)
            split(lo, best)
            split(best, hi)

        split(0, len(word))
        return sorted(chosen)

    def segment(self, word: str, min_confidence: Optional[float] = None) -> List[str]:
        """Segment a word into morphemes."""
        word = word.lower().strip()
        if not word:
            return []
        cuts = [0] + self.boundaries(word, min_confidence) + [len(word)]
        return [word[a:b] for a, b in zip(cuts, cuts[1:])]

    def segment_detailed(self, word: str, min_confidence: Optional[float] = None) -> List[Tuple[str, float]]:
        """Same segmentation as segment(); each piece comes with the score of the boundary that ends it (0.0 for the last)."""
        word = word.lower().strip()
        if not word:
            return []
        scores = self.boundary_scores(word)
        cuts = self.boundaries(word, min_confidence)
        edges = [0] + cuts + [len(word)]
        return [(word[a:b], scores.get(b, 0.0) if b < len(word) else 0.0)
                for a, b in zip(edges, edges[1:])]

    def show_analysis(self, word: str, min_confidence: Optional[float] = None) -> None:
        """Print the statistics of every position; marks the boundaries chosen by segment()."""
        word = word.lower().strip()
        chosen = set(self.boundaries(word, min_confidence))
        print(f"\nAnalysis of '{word}':")
        print("-" * 96)
        for pos in range(1, len(word)):
            stem, ending = self._sides(word, pos)
            conf = self._split_confidence(word, pos)
            p_mark = "+" if self._is_productive(*stem) else "-"
            s_mark = "+" if self._is_productive(*ending) else "-"
            marker = "  <<< SPLIT" if pos in chosen else ""
            print(f"  {word[:pos]:>12} | {word[pos:]:<12} "
                  f"stem: N={stem[0]:>5} b={stem[1]:>2} e={stem[2]:>5} {p_mark}  "
                  f"ending: N={ending[0]:>5} b={ending[1]:>2} e={ending[2]:>5} {s_mark}  "
                  f"score={conf:.3f}{marker}")


def run_experiment() -> ChildMorphemeLearner:
    """Learn from Italian verb paradigms and a few nouns and adjectives, then segment test words."""
    print("=" * 70)
    print("STATISTICAL MORPHEME LEARNER")
    print("Morpheme boundaries from frequency statistics and the Tolerance Principle")
    print("=" * 70)

    learner = ChildMorphemeLearner(min_frequency=3, min_branching=2, min_piece_len=1)

    corpus = """
    cercare cerca cerchi cerca cerchiamo cercate cercano
    cercavo cercavi cercava cercavamo cercavate cercavano
    cercando cercato cercata cercati cercate
    cercarlo cercarla cercarli cercarle cercandolo cercandola

    parlare parlo parli parla parliamo parlate parlano
    parlavo parlavi parlava parlavamo parlavate parlavano
    parlando parlato parlata parlati parlate
    parlarlo parlarla parlarli parlarle parlandolo parlandola

    mangiare mangio mangi mangia mangiamo mangiate mangiano
    mangiavo mangiavi mangiava mangiavamo mangiavate mangiavano
    mangiando mangiato mangiata mangiati mangiate
    mangiarlo mangiarla mangiarli mangiarle mangiandolo mangiandola

    arrivare arrivo arrivi arriva arriviamo arrivate arrivano
    arrivavo arrivavi arrivava arrivavamo arrivavate arrivavano
    arrivando arrivato arrivata arrivati arrivate
    arrivarlo arrivarla arrivarli arrivarle arrivandolo arrivandola

    vedere vedo vedi vede vediamo vedete vedono
    vedevo vedevi vedeva vedevamo vedevate vedevano
    vedendo veduto veduta veduti vedute
    vederlo vederla vederli vederle vedendolo vedendola

    dormire dormo dormi dorme dormiamo dormite dormono
    dormivo dormivi dormiva dormivamo dormivate dormivano
    dormendo dormito dormita dormiti dormite
    dormirlo dormirla dormirli dormirle dormendolo dormendola

    finire finisco finisci finisce finiamo finite finiscono
    finivo finivi finiva finivamo finivate finivano
    finendo finito finita finiti finite
    finirlo finirla finirli finirle finendolo finendola

    amare amo ami ama amiamo amate amano
    amavo amavi amava amavamo amavate amavano
    amando amato amata amati amate
    amarlo amarla amarli amarle amandolo amandola

    portare porto porti porta portiamo portate portano
    portavo portavi portava portavamo portavate portavano
    portando portato portata portati portate
    portarlo portarla portarli portarle portandolo portandola

    trovare trovo trovi trova troviamo trovate trovano
    trovavo trovavi trovava trovavamo trovavate trovavano
    trovando trovato trovata trovati trovate
    trovarlo trovarla trovarli trovarle trovandolo trovandola

    gatto gatti gatta gatte
    cane cani
    casa case
    libro libri
    bella bello belle belli bellissimo bellissima bellissimi bellissime
    grande grandi grandissimo grandissima grandissimi grandissime
    """

    print("\nTraining: repeated exposure to Italian word forms...")
    for _ in range(30):
        learner.observe_text(corpus)
    print(f"  Total tokens: {learner.total_tokens:,}")
    print(f"  Vocabulary size: {len(learner.vocabulary)}")

    test_words = [
        "cercarlo",     # cerca-r-lo or cerc-a-r-lo
        "parlandolo",   # parla-ndo-lo or parl-a-ndo-lo
        "arrivavano",   # arriva-va-no or arriv-a-va-no
        "vedendolo",    # ved-e-ndo-lo
        "dormivano",    # dorm-i-va-no
        "mangiando",    # mangi-a-ndo
        "bellissimo",   # bell-issim-o
        "amato",        # am-a-to or ama-to
        "cercate",      # cerc-a-te
        "dormirlo",     # dorm-i-r-lo
    ]

    print("\n" + "=" * 70)
    print("SEGMENTATION RESULTS")
    print("=" * 70)
    print(f"\n{'Word':<15} {'Segmentation':<30}")
    print("-" * 50)
    for word in test_words:
        print(f"{word:<15} {'-'.join(learner.segment(word)):<30}")

    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS: why does 'cercarlo' split where it does?")
    print("=" * 70)
    learner.show_analysis("cercarlo")

    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS: 'arrivavano'")
    print("=" * 70)
    learner.show_analysis("arrivavano")

    print("\n" + "=" * 70)
    print("DISCOVERED PRODUCTIVE PATTERNS")
    print("=" * 70)
    print("\nMost productive endings (by number of different predecessors):")
    endings = [(s, learner.suffix_freq[s], len(p)) for s, p in learner.suffix_predecessors.items()
               if learner.suffix_freq[s] >= 10 and len(p) >= 2]
    endings.sort(key=lambda x: (-x[2], -x[1]))
    for suffix, freq, branching in endings[:15]:
        print(f"  -{suffix:<10} frequency={freq:>5}, branching={branching:>2}")

    print("\nMost productive stems (by number of different continuations):")
    stems = [(p, learner.prefix_freq[p], len(c)) for p, c in learner.prefix_continuations.items()
             if learner.prefix_freq[p] >= 10 and len(c) >= 2 and len(p) >= 2]
    stems.sort(key=lambda x: (-x[2], -x[1]))
    for prefix, freq, branching in stems[:15]:
        continuations = list(learner.prefix_continuations[prefix].keys())[:6]
        print(f"  {prefix:<10}- frequency={freq:>5}, branching={branching:>2}, continues: {continuations}")
    return learner


if __name__ == "__main__":
    run_experiment()
