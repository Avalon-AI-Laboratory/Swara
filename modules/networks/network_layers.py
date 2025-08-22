import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, einsum
from math import floor, log, pi
from typing import Any, List, Optional, Sequence, Tuple, Union

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many