from affect.classifiers.goemotions_encoder import GoEmotionsEncoder
from affect.taxonomies.goemotions import LABELS


def test_lazy_encoder_initializes_all_private_state_without_loading_models():
    encoder = GoEmotionsEncoder("fixture-model")
    assert encoder._model is None
    assert encoder._tokenizer is None
    assert encoder._device is None
    assert encoder._temperature == 1.0
    assert encoder._thresholds == dict.fromkeys(LABELS, 0.5)
    encoder._thresholds["joy"] = 0.8
    assert GoEmotionsEncoder("another-model")._thresholds["joy"] == 0.5
