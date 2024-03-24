__version__ = '0.0.9'

from .io import BrainvisionInfo
from .proc import EEGRef
from .proc import FilterParams
from .proc import EpochProps
from .proc import ICAParams
from .proc import fNIRSreadParams
from .type import Subject


from nirs_help import (
    compute_contrast_custom,
    Contrast_custom
)

