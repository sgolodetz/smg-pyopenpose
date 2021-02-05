import numpy as np

from typing import Dict, List, Optional, Tuple

from .skeleton_detector import SkeletonDetector


class BoneLengthEstimator:
    """Used to estimate the lengths of bones, to help with filtering out poor keypoints."""

    # CONSTRUCTOR

    def __init__(self):
        self.__bone_length_estimates: Dict[Tuple[str, str], List[float]] = {}

    # PUBLIC METHODS

    def get_bone_lengths(self) -> Dict[Tuple[str, str], float]:
        return {name: np.median(estimate) for name, estimate in self.__bone_length_estimates.items()}

    def update(self, skeleton: SkeletonDetector.Skeleton) -> None:
        for keypoint1, keypoint2 in skeleton.bones:
            bone_key: Tuple[str, str] = SkeletonDetector.make_bone_key(keypoint1, keypoint2)
            bone_length_estimate: float = np.linalg.norm(keypoint1.position - keypoint2.position)

            bone_length: Optional[List[float]] = self.__bone_length_estimates.get(bone_key)
            if bone_length is None:
                bone_length = [bone_length_estimate]
            else:
                bone_length.append(bone_length_estimate)

            self.__bone_length_estimates[bone_key] = bone_length
