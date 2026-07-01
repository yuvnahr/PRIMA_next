"""Legacy-compatible emotion classifier implementation."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from affect.emotion_cache import LRUCache
from affect.emotion_profile import EmotionProfile
from affect.interfaces import SimilarityBackend, choose_similarity_backend

TOKEN_RE = re.compile(r"[a-zA-Z']+")

SCORE_LABELS = {
    "negative": 0,
    "positive": 0,
    "neutral": 0,
    "uncertainty": 0,
    "litigious": 0,
    "model_strong": 0,
    "model_weak": 0,
    "anticipation": 0,
    "anger": 0,
    "fear": 0,
    "sadness": 0,
    "trust": 0,
    "senerity": 0,
    "joy_ecstasy": 0,
    "joy": 0,
    "sad": 0,
    "admire": 0,
    "acceptance": 0,
    "amazement_surprise": 0,
    "surprise": 0,
    "distraction": 0,
    "boredom": 0,
    "disgust_loathing": 0,
    "disgust": 0,
    "interest_vigilance": 0,
}

NEGATIONS = {
    "never",
    "not",
    "no",
    "didn",
    "didn't",
    "didnt",
    "doesnt",
    "doesn't",
    "won't",
    "wont",
    "isn't",
    "isnt",
    "aren't",
    "arent",
    "don't",
    "dont",
    "haven't",
    "havent",
    "weren't",
    "werent",
    "wasn't",
    "wasnt",
    "wouldn't",
    "wouldnt",
    "can't",
    "cant",
    "cannot",
    "couldn't",
    "couldnt",
    "shouldnt",
    "shouldn't",
    "neither",
    "impossible",
    "nothing",
}

INTENSITY_MODIFIERS = {
    "absolutely": ("B_INCR", 0.9),
    "almost": ("B_DECR", 0.2),
    "always": ("B_INCR", 0.9),
    "amazingly": ("B_INCR", 0.8),
    "awfully": ("B_INCR", 0.8),
    "completely": ("B_INCR", 0.9),
    "deeply": ("B_INCR", 0.9),
    "entirely": ("B_INCR", 0.9),
    "especially": ("B_INCR", 0.8),
    "extremely": ("B_INCR", 0.9),
    "fully": ("B_INCR", 0.9),
    "greatly": ("B_INCR", 0.8),
    "highly": ("B_INCR", 0.9),
    "hugely": ("B_INCR", 0.9),
    "incredible": ("B_INCR", 0.9),
    "incredibly": ("B_INCR", 0.9),
    "intensely": ("B_INCR", 0.9),
    "kind of": ("B_DECR", 0.3),
    "kinda": ("B_DECR", 0.3),
    "less": ("B_DECR", 0.3),
    "little": ("B_DECR", 0.3),
    "more": ("B_INCR", 0.6),
    "most": ("B_INCR", 0.8),
    "not much": ("B_DECR", 0.2),
    "particularly": ("B_INCR", 0.7),
    "partly": ("B_DECR", 0.3),
    "partially": ("B_DECR", 0.5),
    "quite": ("B_INCR", 0.6),
    "really": ("B_INCR", 0.8),
    "slight": ("B_DECR", 0.3),
    "slightly": ("B_DECR", 0.6),
    "barely": ("B_DECR", 0.7),
    "so": ("B_INCR", 0.8),
    "somewhat": ("B_DECR", 0.4),
    "sort of": ("B_DECR", 0.4),
    "substantially": ("B_INCR", 0.7),
    "super": ("B_INCR", 0.8),
    "thoroughly": ("B_INCR", 0.8),
    "total": ("B_INCR", 0.8),
    "totally": ("B_INCR", 0.9),
    "truly": ("B_INCR", 0.9),
    "unbelievably": ("B_INCR", 0.9),
    "utterly": ("B_INCR", 0.8),
    "very": ("B_INCR", 0.8),
    "not very": ("B_DECR", 0.7),
}

CANONICAL_EMOTION_ALIASES = {
    "sad": "sadness",
    "joy_ecstasy": "joy",
    "amazement_surprise": "surprise",
    "interest_vigilance": "surprise",
    "disgust_loathing": "disgust",
    "senerity": "trust",
    "serenity": "trust",
    "admire": "trust",
    "acceptance": "trust",
    "positive": "joy",
    "negative": "sadness",
}


OPPOSITE_EMOTIONS = {
    "anticipation": "surprise",
    "anger": "joy",
    "fear": "trust",
    "sadness": "joy_ecstasy",
    "trust": "fear",
    "joy": "sad",
    "surprise": "anticipation",
    "disgust": "joy",
    "senerity": "distraction",
    "joy_ecstasy": "sadness",
    "sad": "joy",
    "admire": "disgust",
    "acceptance": "disgust",
    "amazement_surprise": "anticipation",
    "distraction": "senerity",
    "boredom": "interest_vigilance",
    "disgust_loathing": "joy_ecstasy",
    "interest_vigilance": "boredom",
    "negative": "positive",
    "positive": "negative",
    "model_strong": "model_weak",
    "model_weak": "model_strong",
    "uncertainty": "uncertainty",
    "litigious": "litigious",
}

FALLBACK_LEXICON = {
    "anticipation": [
        "waiting", "tomorrow", "soon", "expect", "expecting", "upcoming", "hopeful", "looking forward",
        "counting down", "planned", "preparing", "eager", "deadline", "awaiting",
    ],
    "anger": [
        "angry", "mad", "furious", "rage", "outraged", "irritated", "frustrated", "annoying", "canceled",
        "delayed", "insulted", "betrayed", "unfair", "yelled", "argument", "resent", "hostile", "livid",
    ],
    "fear": [
        "afraid", "scared", "nervous", "worried", "terrified", "anxious", "panic", "panicked", "unsafe",
        "threat", "danger", "dread", "horror", "frightened", "uneasy", "brakes", "tests", "alarm",
    ],
    "sadness": [
        "sad", "grief", "grieving", "heartbroken", "miserable", "depressed", "failed", "passed away", "died",
        "loss", "crying", "tears", "rejected", "lonely", "miss", "empty", "mourning", "devastated", "hurt",
    ],
    "trust": [
        "trust", "trusted", "count on", "reliable", "friend", "support", "supported", "safe", "secure", "honest",
        "loyal", "dependable", "comforted", "backed me up", "protected", "faith", "confidence", "reassured",
    ],
    "senerity": ["peace", "peaceful", "content", "okay", "calm", "fine", "relieved", "let go", "settled"],
    "joy": [
        "joy", "happy", "happiness", "pleasure", "delighted", "delight", "glad", "smiling", "celebrate",
        "celebrated", "promoted", "promotion", "baby", "steps", "proposal", "accepted", "love", "proud",
        "wonderful", "great news", "won", "success", "excited", "cheerful",
    ],
    "joy_ecstasy": ["amazing", "dream", "thrilled", "ecstatic", "finished", "fantastic", "overjoyed", "elated"],
    "admire": ["talented", "dedication", "incredible", "overcame", "look up", "respect", "inspiring"],
    "acceptance": ["terms", "made peace", "it is what it is"],
    "surprise": [
        "surprise", "surprised", "wonder", "wonderstruck", "awe", "amazed", "astonished", "unexpected", "suddenly",
        "startled", "shocked", "raffle", "found", "package", "unplanned", "out of nowhere", "couldn't believe",
    ],
    "amazement_surprise": ["unbelievable", "shocking", "astonishing", "stunned"],
    "distraction": ["wandering", "replaying", "focus", "thinking", "staring"],
    "boredom": ["bored", "nothing", "routine", "counting", "meeting", "tedious"],
    "disgust": [
        "disgust", "disgusted", "mold", "hair", "dirty", "unclean", "gross", "nauseated", "rotten", "filthy",
        "repulsed", "sickened", "stench", "slimy", "spoiled",
    ],
    "disgust_loathing": ["cockroaches", "loathing", "revolting", "vile"],
    "interest_vigilance": ["interesting", "attention", "add up", "eye on"],
}



@dataclass(frozen=True, slots=True)
class LexiconEntry:
    token: str
    label: str
    embedding: np.ndarray


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def calculate_scores(nearest_neighbs_words: Sequence[str], nearest_neighbs_labels: Sequence[str]) -> dict[str, float]:
    """Replicate the legacy rank-weighted scoring behavior."""
    score_dict = SCORE_LABELS.copy()
    for index, label in enumerate(nearest_neighbs_labels):
        score_dict[label] = score_dict.get(label, 0) + 50 - index

    score_max = (len(nearest_neighbs_words) * (len(nearest_neighbs_words) - 1)) / 2
    normalized: dict[str, float] = {}
    for key, value in score_dict.items():
        if value and score_max:
            normalized[key] = round(value / score_max, 3)
    return normalized


def normalize_scores(scores: Mapping[str, float]) -> dict[str, float]:
    total = sum(value for value in scores.values() if value > 0)
    if total <= 0:
        return {}
    return {key: round(value / total, 3) for key, value in scores.items() if value > 0}


def canonicalize_scores(scores: Mapping[str, float]) -> dict[str, float]:
    """Collapse legacy fine-grained labels into publication-facing emotion labels."""
    canonical: dict[str, float] = {}
    for label, score in scores.items():
        mapped = CANONICAL_EMOTION_ALIASES.get(label, label)
        canonical[mapped] = canonical.get(mapped, 0.0) + float(score)
    return normalize_scores(canonical)




def fix_score(current_score: float, direction: str, strength: float) -> float:
    if direction == "B_INCR":
        return current_score + current_score * strength
    if direction == "B_DECR":
        return current_score - current_score * strength
    return current_score


def map_opposite_emotions(emotion_scores: Mapping[str, float]) -> dict[str, float]:
    opposed: dict[str, float] = {}
    for emotion, score in emotion_scores.items():
        opposite = OPPOSITE_EMOTIONS.get(emotion, emotion)
        opposed[opposite] = opposed.get(opposite, 0.0) + score
    return opposed


def resolve_modifiers_and_negations(tokens: Sequence[str], scores: Mapping[str, float]) -> dict[str, float]:
    """Replicate legacy modifier behavior on the dominant emotion."""
    if not scores:
        return {}
    normalized = dict(sorted(scores.items(), key=lambda item: item[1]))
    context = " ".join(tokens[max(0, len(tokens) - 5) :]).lower()
    dominant = list(normalized.keys())[-1]

    for modifier, (direction, strength) in INTENSITY_MODIFIERS.items():
        if modifier in context:
            normalized[dominant] = fix_score(normalized[dominant], direction, strength)

    if any(token in NEGATIONS for token in tokens):
        normalized = map_opposite_emotions(normalized)

    return normalize_scores(normalized)


def _stable_embedding(text: str, dimensions: int = 64) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    while len(values) < dimensions:
        for byte in digest:
            values.append((byte / 127.5) - 1.0)
            if len(values) == dimensions:
                break
        digest = hashlib.sha256(digest).digest()
    vector = np.array(values, dtype="float32")
    norm = np.linalg.norm(vector)
    return vector / (norm if norm else 1.0)


def _lexicon_from_fallback() -> list[LexiconEntry]:
    entries: list[LexiconEntry] = []
    for label, phrases in FALLBACK_LEXICON.items():
        for phrase in phrases:
            entries.append(LexiconEntry(phrase, label, _stable_embedding(phrase)))
    return entries


class LegacyAffectClassifier:
    """Legacy-compatible retrieval plus modifier classifier.

    The class can operate with injected model/tokenizer/lexicon objects to
    preserve benchmark behavior, but has a deterministic local fallback so the
    PRIMA state layer and tests do not require network downloads.
    """

    def __init__(
        self,
        entries: Iterable[LexiconEntry] | None = None,
        backend: SimilarityBackend | None = None,
        cache_size: int = 512,
        prefer_faiss: bool = False,
    ) -> None:
        self.entries = list(entries or _lexicon_from_fallback())
        self.matrix = np.vstack([entry.embedding for entry in self.entries]).astype("float32")
        self.backend = backend or choose_similarity_backend(self.matrix, prefer_faiss=prefer_faiss)
        self.cache: LRUCache[str, EmotionProfile] = LRUCache(cache_size)

    @classmethod
    def from_legacy_dataframe(
        cls,
        df: Any,
        cache_size: int = 512,
        prefer_faiss: bool = False,
    ) -> LegacyAffectClassifier:
        """Create a classifier from the legacy CSV dataframe shape.

        Expected columns match the previous implementation:
        `token`, `fourteen_label`, and `embedding`.
        """
        entries: list[LexiconEntry] = []
        for _, row in cast(Any, df).iterrows():
            entries.append(
                LexiconEntry(
                    token=str(row["token"]),
                    label=str(row["fourteen_label"]),
                        embedding=np.array(row["embedding"], dtype="float32"),
                )
            )
        return cls(entries=entries, cache_size=cache_size, prefer_faiss=prefer_faiss)

    def classify(self, text: str) -> EmotionProfile:
        cached = self.cache.get(text)
        if cached is not None:
            return cached

        tokens = tokenize(text)
        scores, keywords = self._classify_by_lexical_match(text, tokens)
        if not scores:
            scores, keywords = self._classify_by_similarity(text)

        resolved_scores = canonicalize_scores(resolve_modifiers_and_negations(tokens, scores) or scores)
        profile = EmotionProfile.from_scores(resolved_scores, emotional_keywords=tuple(keywords))
        self.cache.set(text, profile)
        return profile

    def legacy_build_profile(self, text: str) -> list[object]:
        profile = self.classify(text)
        return [dict(profile.emotions), list(profile.emotional_keywords)]

    def classify_with_embedding(
        self,
        text: str,
        embedding: Sequence[float],
        top_k: int = 50,
        modifier_detection: bool = True,
    ) -> EmotionProfile:
        """Classify with an externally supplied sentence embedding."""
        scores, keywords = self._classify_embedding(embedding, top_k=top_k)
        resolved = resolve_modifiers_and_negations(tokenize(text), scores) if modifier_detection else scores
        return EmotionProfile.from_scores(canonicalize_scores(resolved or scores), tuple(keywords))

    def _classify_embedding(self, embedding: Sequence[float] | np.ndarray, top_k: int = 50) -> tuple[dict[str, float], list[str]]:
        query = np.array(embedding, dtype="float32")
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        _, indices = self.backend.search(query, min(top_k, len(self.entries)))
        words = [self.entries[index].token for index in indices]
        labels = [self.entries[index].label for index in indices]
        return normalize_scores(calculate_scores(words, labels)), words[:5]

    def _classify_by_similarity(self, text: str, top_k: int = 50) -> tuple[dict[str, float], list[str]]:
        query = _stable_embedding(text, dimensions=self.matrix.shape[1])
        return self._classify_embedding(query, top_k=top_k)

    def _classify_by_lexical_match(self, text: str, tokens: Sequence[str]) -> tuple[dict[str, float], list[str]]:
        lowered = text.lower()
        raw_scores: dict[str, float] = {}
        keywords: list[str] = []
        for entry in self.entries:
            phrase_tokens = entry.token.split()
            match = entry.token in lowered if len(phrase_tokens) > 1 else entry.token in tokens
            if match:
                weight = 1.0 + min(0.5, len(phrase_tokens) * 0.1)
                raw_scores[entry.label] = raw_scores.get(entry.label, 0.0) + weight
                keywords.append(entry.token)
        return normalize_scores(raw_scores), keywords[:8]


_DEFAULT_CLASSIFIER = LegacyAffectClassifier()


def get_emotion_profile(text: str) -> EmotionProfile:
    """Backward-compatible public profile helper."""
    return _DEFAULT_CLASSIFIER.classify(text)


def legacy_build_profile(text: str) -> list[object]:
    """Adapter for older experiments expecting [score_dict, keywords]."""
    return _DEFAULT_CLASSIFIER.legacy_build_profile(text)


def build_vocab_matrix(df: Any) -> np.ndarray:
    """Legacy-compatible normalized vocabulary matrix builder."""
    matrix: np.ndarray = np.asarray(
        df["embedding"].tolist(),
        dtype=np.float32,
    )

    norms: np.ndarray = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True,
    )

    norms = np.where(norms == 0, 1e-9, norms)

    result: np.ndarray = matrix / norms
    return result


def mean_pooling(model_output: Any, attention_mask: Any) -> np.ndarray:
    """Legacy mean-pooling helper for transformer outputs.

    Returns a numpy ndarray regardless of whether the model returns a torch
    tensor or a numpy array.
    """
    token_embeddings = model_output[0]
    # attempt to operate on torch tensors if available, otherwise fall back to numpy
    try:
        # torch path
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1).clamp(min=1e-9)
        # convert to numpy and ensure the expression is typed as ndarray
        pooled_arr: np.ndarray

        if hasattr(pooled, "cpu"):
            pooled_arr = np.asarray(pooled.cpu().numpy(), dtype=np.float32)
        else:
            pooled_arr = np.asarray(pooled, dtype=np.float32)

        return pooled_arr
    except Exception:
        # numpy-path (or other array-likes)
        input_mask_expanded = np.expand_dims(attention_mask, -1) * np.ones_like(token_embeddings)
        pooled = (token_embeddings * input_mask_expanded).sum(1) / np.maximum(input_mask_expanded.sum(1), 1e-9)
        return cast(np.ndarray, np.asarray(pooled))


def get_mean_pooling_emb(sentences: list[str], tokenizer: Any, model: Any) -> list[list[float]]:
    """Encode sentences using an injected tokenizer/model pair."""
    device = next(model.parameters()).device
    encoded_input = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)
    import torch

    with torch.no_grad():
        model_output = model(**encoded_input)
    return cast(list[list[float]], mean_pooling(model_output, encoded_input["attention_mask"]).tolist())


def build_profile(
    sentence: str,
    window_size: int,
    df: object,
    vocab_matrix: np.ndarray,
    tokenizer: object,
    model: object,
    keyword_extraction: bool = True,
    modifier_detection: bool = True,
) -> list[object]:
    """Legacy benchmark adapter with the old `build_profile` call shape."""
    del window_size, keyword_extraction
    classifier = LegacyAffectClassifier.from_legacy_dataframe(df)
    classifier.matrix = vocab_matrix
    classifier.backend = choose_similarity_backend(vocab_matrix)
    embedding = get_mean_pooling_emb([sentence], tokenizer, model)[0]
    profile = classifier.classify_with_embedding(
        sentence,
        embedding,
        top_k=50,
        modifier_detection=modifier_detection,
    )
    return [dict(sorted(profile.emotions.items(), key=lambda item: item[1])), list(profile.emotional_keywords)]
