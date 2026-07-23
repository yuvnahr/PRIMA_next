"""Official GoEmotions labels plus explicit PRIMA engineering mappings."""

from __future__ import annotations

GOEMOTIONS_VERSION = "google-research-goemotions-2020"
LABELS = (
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
    "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral",
)
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
DEFINITIONS = {
    "admiration": "respect or approval", "amusement": "finding something funny", "anger": "hostility or rage", "annoyance": "mild irritation", "approval": "positive evaluation", "caring": "concern for welfare", "confusion": "lack of understanding", "curiosity": "desire to know", "desire": "wanting an outcome", "disappointment": "sadness after an unmet expectation", "disapproval": "negative evaluation", "disgust": "revulsion", "embarrassment": "self-conscious discomfort", "excitement": "energized positive anticipation", "fear": "perceived threat", "gratitude": "thankfulness", "grief": "deep sorrow over loss", "joy": "happiness", "love": "affection", "nervousness": "anxious unease", "optimism": "positive expectation", "pride": "satisfaction in achievement", "realization": "coming to understand", "relief": "ease after concern", "remorse": "regret for wrongdoing", "sadness": "unhappiness or sorrow", "surprise": "response to the unexpected", "neutral": "no annotated emotion",
}
EKMAN_GROUPS = {"anger": ("anger", "annoyance", "disapproval"), "disgust": ("disgust",), "fear": ("fear", "nervousness"), "joy": ("joy", "amusement", "approval", "excitement", "gratitude", "love", "optimism", "relief", "pride", "admiration", "desire", "caring"), "sadness": ("sadness", "disappointment", "embarrassment", "grief", "remorse"), "surprise": ("surprise", "realization", "confusion", "curiosity")}
SENTIMENT_GROUPS = {"positive": ("amusement", "excitement", "joy", "love", "desire", "optimism", "caring", "pride", "admiration", "gratitude", "relief", "approval"), "negative": ("fear", "nervousness", "remorse", "embarrassment", "disappointment", "sadness", "grief", "disgust", "anger", "annoyance", "disapproval"), "ambiguous": ("realization", "surprise", "curiosity", "confusion")}
# This is a PRIMA engineering mapping into existing core PAD categories, not an official GoEmotions claim.
PRIMA_CORE_MAP = {"admiration": "trust", "amusement": "joy", "anger": "anger", "annoyance": "anger", "approval": "trust", "caring": "trust", "confusion": "surprise", "curiosity": "surprise", "desire": "anticipation", "disappointment": "sadness", "disapproval": "disgust", "disgust": "disgust", "embarrassment": "sadness", "excitement": "joy", "fear": "fear", "gratitude": "trust", "grief": "sadness", "joy": "joy", "love": "joy", "nervousness": "fear", "optimism": "anticipation", "pride": "joy", "realization": "surprise", "relief": "joy", "remorse": "sadness", "sadness": "sadness", "surprise": "surprise", "neutral": "neutral"}
ALIASES = {"serenity": "neutral"}
CONTRASTS = {"anger": ("annoyance", "disapproval"), "fear": ("nervousness",), "joy": ("amusement", "excitement", "relief"), "sadness": ("grief", "disappointment", "remorse"), "surprise": ("realization", "confusion", "curiosity")}


def validate_taxonomy() -> dict[str, object]:
    missing = [label for label in LABELS if label not in PRIMA_CORE_MAP]
    invalid = sorted(set(PRIMA_CORE_MAP.values()) - {"joy", "trust", "anticipation", "surprise", "fear", "sadness", "anger", "disgust", "neutral"})
    if missing or invalid:
        raise ValueError(f"Invalid GoEmotions PRIMA mapping: missing={missing}, invalid={invalid}")
    return {"version": GOEMOTIONS_VERSION, "label_count": len(LABELS), "labels": list(LABELS), "prima_mapping_complete": not missing, "invalid_core_labels": invalid}
