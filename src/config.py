from dataclasses import dataclass, field, asdict, replace
from typing import List

@dataclass
class ModelConfig:
  img_size:int = 224
  patch_size:int = 16
  in_chans:int = 3
  embed_dim:int = 768
  num_blocks:int = 12,
  num_heads:int = 12
  mlp_ratio:float = 4.
  mlp_layers:List[float] = None
  dropout:float = 0.1
  attn_dropout:float = 0.1
  output_dim:int = 1
  relative_positional_bias:bool = False