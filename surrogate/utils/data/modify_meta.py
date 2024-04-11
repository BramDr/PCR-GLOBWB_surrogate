from typing import Optional
import numpy as np


def modify_meta(meta: dict,
                feature: Optional[str] = None,
                verbose: int = 1) -> dict:

    if feature is not None:
        meta["features"] = np.array([feature])

    if verbose > 0:
        print("Modified meta")
    return meta
