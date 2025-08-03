import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.layers import (
    Linear,
    Embeding,
    RMSNorm,
    SiLU,
    SwiGLU,
    Softmax,
    scaled_dot_product_attention,
    RoPE,
    MultiheadAttention,
    TranformerBlock,
    TransformerLM,
)
from cs336_basics.loss import CELosss
from cs336_basics.optim import AdamW, get_lr_cosine_sheduler
from cs336_basics.train import (
    clip_gradients,
    save_checkpoint,
    load_checkpoint,
)
from cs336_basics.dataset import SequenceDataset
