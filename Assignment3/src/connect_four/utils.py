from typing import Set

import numpy as np


def available_actions(grid) -> Set[int]:
    return set(np.where(grid == 0)[1])
