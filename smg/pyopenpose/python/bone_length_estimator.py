import numpy as np

from collections import deque
from typing import Deque, Dict, Optional, Tuple

from .skeleton_detector import SkeletonDetector


class BoneLengthEstimator:
    """Used to estimate the lengths of bones, to help with filtering out poor keypoints."""

    # CONSTRUCTOR

    def __init__(self, *, max_estimates_per_bone: int = 100):
        self.__bone_length_estimates: Dict[Tuple[str, str], Deque[float]] = {}
        self.__max_estimates_per_bone: int = max_estimates_per_bone

    # PUBLIC METHODS

    def get_bone_lengths(self) -> Dict[Tuple[str, str], float]:
        return {name: np.median(estimate) for name, estimate in self.__bone_length_estimates.items()}

    def update(self, skeleton: SkeletonDetector.Skeleton) -> None:
        for keypoint1, keypoint2 in skeleton.bones:
            bone_key: Tuple[str, str] = SkeletonDetector.make_bone_key(keypoint1, keypoint2)
            bone_length_estimate: float = np.linalg.norm(keypoint1.position - keypoint2.position)

            bone_length_estimates: Optional[Deque[float]] = self.__bone_length_estimates.get(bone_key)
            if bone_length_estimates is None:
                bone_length_estimates = deque([bone_length_estimate])
            else:
                bone_length_estimates.append(bone_length_estimate)
                if len(bone_length_estimates) == self.__max_estimates_per_bone:
                    bone_length_estimates = deque(sorted(bone_length_estimates))
                    bone_length_estimates.popleft()
                    bone_length_estimates.pop()

            self.__bone_length_estimates[bone_key] = bone_length_estimates
