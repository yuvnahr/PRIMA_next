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
    "admiration": "Finding something impressive or worthy of respect.", "amusement": "Finding something funny or being entertained.", "anger": "A strong feeling of displeasure or antagonism.", "annoyance": "Mild anger or irritation.", "approval": "Having or expressing a favorable opinion.", "caring": "Displaying kindness and concern for others.", "confusion": "Lack of understanding or uncertainty.", "curiosity": "A strong desire to know or learn something.", "desire": "A strong feeling of wanting something or wishing for something to happen.", "disappointment": "Sadness or displeasure caused by the nonfulfillment of hopes or expectations.", "disapproval": "Having or expressing an unfavorable opinion.", "disgust": "Revulsion or strong disapproval aroused by something unpleasant or offensive.", "embarrassment": "Self-consciousness, shame, or awkwardness.", "excitement": "A feeling of great enthusiasm and eagerness.", "fear": "Being afraid or worried.", "gratitude": "A feeling of thankfulness and appreciation.", "grief": "Intense sorrow, especially caused by someone's death.", "joy": "A feeling of pleasure and happiness.", "love": "A strong positive emotion of regard and affection.", "nervousness": "Apprehension, worry, or anxiety.", "optimism": "Hopefulness and confidence about the future or the success of something.", "pride": "Pleasure or satisfaction due to one's own achievements or those of someone closely associated.", "realization": "Becoming aware of something.", "relief": "Reassurance and relaxation following release from anxiety or distress.", "remorse": "Regret or a guilty feeling.", "sadness": "Emotional pain or sorrow.", "surprise": "Feeling astonished or startled by something unexpected.", "neutral": "No particular emotion is expressed.",
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
