from .FixColorOfLoopMovie import FixColor, Int2String
from .FixColorOfLoopMovie2 import FixColorHighPrecision
from .SliceImageBatch import SliceImageBatch
NODE_CLASS_MAPPINGS = {
    "Fix color of Loop images": FixColor,
    "Fix color of Loop images HighPrecision": FixColorHighPrecision,
    "Int to String": Int2String,
    "SliceImageBatch": SliceImageBatch,
}