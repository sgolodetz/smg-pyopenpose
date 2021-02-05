import numpy as np

from collections import deque
from typing import Deque, Dict, Optional, Tuple

from .skeleton_detector import SkeletonDetector


class BoneLengthEstimator:
    """Used to estimate the lengths of bones, to help with filtering out poor keypoints."""

    # CONSTRUCTOR

    def __init__(self, *, max_estimates_per_bone: int = 100):
        """
        Construct a bone length estimator.

        .. note::
            We take the median of the current set of estimates for each bone to get its expected length. It's
            clearly too expensive to accumulate and then take the median of a large number of estimates for
            each bone over time, and it's also unnecessary, since most of the estimates are fairly good and
            fairly similar. To avoid this, once we hit a certain number of estimates, we drop estimates at
            the extreme ends of the range. The result won't be the true median of the input sequence, but
            that doesn't matter too much in practice.

        :param max_estimates_per_bone:  The maximum number of estimates to retain per bone.
        """
        self.__bone_length_estimates: Dict[Tuple[str, str], Deque[float]] = {}
        self.__max_estimates_per_bone: int = max_estimates_per_bone

    # PUBLIC METHODS

    def add_estimates(self, skeleton: SkeletonDetector.Skeleton) -> None:
        """
        Add bone length estimates based on the specified skeleton.

        .. note::
            It's clearly important not to reuse a bone length estimator for multiple people - it's designed
            to be passed the per-frame skeletons of the same person as they're tracked through the sequence.
            In multi-person scenarios, multiple bone length estimators should be used, and associating their
            per-frame skeletons correctly thus becomes extremely important.

        :param skeleton:    The skeleton.
        """
        # For each bone in the skeleton:
        for keypoint1, keypoint2 in skeleton.bones:
            bone_key: Tuple[str, str] = SkeletonDetector.make_bone_key(keypoint1, keypoint2)

            # Calculate the length of the bone in the skeleton, as an estimate of its true length.
            bone_length_estimate: float = np.linalg.norm(keypoint1.position - keypoint2.position)

            # Either start the list of estimates for the bone, or add to it.
            bone_length_estimates: Optional[Deque[float]] = self.__bone_length_estimates.get(bone_key)
            if bone_length_estimates is None:
                bone_length_estimates = deque([bone_length_estimate])
            else:
                bone_length_estimates.append(bone_length_estimate)

                # If we hit the maximum number of estimates for the bone:
                if len(bone_length_estimates) == self.__max_estimates_per_bone:
                    # Remove the lowest and highest estimates, and carry on.
                    # TODO: This could be made more efficient if necessary - the sorting is by no means essential.
                    bone_length_estimates = deque(sorted(bone_length_estimates))
                    bone_length_estimates.popleft()
                    bone_length_estimates.pop()

            self.__bone_length_estimates[bone_key] = bone_length_estimates

    def get_expected_bone_lengths(self) -> Dict[Tuple[str, str], float]:
        """
        Get the expected bone lengths, based on the current estimates held for each bone.

        :return:    The expected bone lengths, as a bone key -> expected length map.
        """
        return {name: np.median(estimate) for name, estimate in self.__bone_length_estimates.items()}
