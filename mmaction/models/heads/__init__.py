from .base import BaseHead
from .i3d_head import I3DHead
from .poseslowfast_head import PoseSlowFastHead
from .simple_head import SimpleHead
from .slowfast_head import SlowFastHead
from .ssn_head import SSNHead
from .tpn_head import TPNHead
from .tsm_head import TSMHead
from .tsn_head import TSNHead

__all__ = [
    'TSNHead', 'I3DHead', 'BaseHead', 'TSMHead', 'SlowFastHead', 'SSNHead',
    'TPNHead', 'SimpleHead', 'PoseSlowFastHead'
]
