"""Single source of truth. Serialized outputs preserve this order."""

from types import MappingProxyType

LABELS = ("neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust")
LABEL_TO_ID = MappingProxyType({name: i for i, name in enumerate(LABELS)})
MARKERS = ("<SELF>", "<OTHER>", "<CURRENT>")
SAMPLE_RATE = 16000
