# Copyright (c) OpenMMLab. All rights reserved.
from .dior import DIORDataset  # noqa: F401, F403
from .dota import DOTAv2Dataset  # noqa: F401, F403
from .dota import DOTADataset, DOTAv15Dataset
from .hrsc import HRSCDataset  # noqa: F401, F403
from .transforms import *  # noqa: F401, F403
from .rsar import RSARDataset
from .sived import SIVEDDataset
from .rsdd import RSDDDataset
from .srsdd import SRSDDDataset


__all__ = [
    'DOTADataset', 'DOTAv15Dataset', 'DOTAv2Dataset', 'HRSCDataset',
    'DIORDataset', 'RSARDataset', 'SIVEDDataset', 'RSDDDataset', 'SRSDDDataset'
]
