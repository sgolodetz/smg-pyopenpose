from __future__ import annotations

import numpy as np

from typing import Any, Dict, List, Optional, Tuple

from ..cpp.pyopenpose import *
from .skeleton import Skeleton


class SkeletonDetector:
    """A 3D skeleton detector based on OpenPose."""

    # CONSTRUCTOR

    def __init__(self, params: Dict[str, Any]):
        """
        Construct a 3D skeleton detector based on OpenPose.

        :param params:  The parameters with which to configure OpenPose.
        """
        self.__wrapper: WrapperPython = WrapperPython()
        self.__wrapper.configure(params)
        self.__wrapper.start()

        # TODO: Support other pose models in the future.
        self.__pose_model: PoseModel = BODY_25
        self.__keypoint_names: Dict[int, str] = getPoseBodyPartMapping(self.__pose_model)

        pose_part_pair_list: List[int] = getPosePartPairs(self.__pose_model)
        pose_part_pairs: List[Tuple[int, int]] = list(zip(pose_part_pair_list[::2], pose_part_pair_list[1::2]))
        self.__keypoint_pairs: List[Tuple[str, str]] = [
            (self.__keypoint_names[i], self.__keypoint_names[j]) for i, j in pose_part_pairs
        ]

    # SPECIAL METHODS

    def __enter__(self):
        """No-op (needed to allow the detector's lifetime to be managed by a with statement)."""
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Destroy the detector at the end of the with statement that's used to manage its lifetime."""
        self.terminate()

    # PUBLIC STATIC METHODS

    @staticmethod
    def make_bone_key(keypoint1: Skeleton.Keypoint, keypoint2: Skeleton.Keypoint) -> Tuple[str, str]:
        """
        Make a key that can be used to look up a bone in a dictionary.

        :param keypoint1:   The keypoint at one end of the bone.
        :param keypoint2:   The keypoint at the other end of the bone.
        :return:            The key for the bone.
        """
        # noinspection PyTypeChecker
        return tuple(sorted([keypoint1.name, keypoint2.name]))

    @staticmethod
    def remove_bad_bones(skeleton: Skeleton, expected_bone_lengths: Dict[Tuple[str, str], float], *,
                         tolerance: float = 0.3) -> Skeleton:
        """
        Remove from the specified skeleton any bone whose estimated length is different from its expected length
        by more than the specified tolerance (in m).

        :param skeleton:                The skeleton.
        :param expected_bone_lengths:   The expected lengths of different bones (in m).
        :param tolerance:               The maximum fraction by which the estimated length of a bone can differ
                                        from its expected length without the bone being removed.
        :return:                        A copy of the skeleton from which the bad bones have been removed.
        """
        good_keypoints: Dict[str, Skeleton.Keypoint] = {}
        good_keypoint_pairs: List[Tuple[str, str]] = []

        # Determine the good bones and the keypoints they touch.
        for keypoint1, keypoint2 in skeleton.bones:
            bone_key: Tuple[str, str] = SkeletonDetector.make_bone_key(keypoint1, keypoint2)
            expected_bone_length: Optional[float] = expected_bone_lengths.get(bone_key)
            if expected_bone_length is not None:
                bone_length: float = np.linalg.norm(keypoint1.position - keypoint2.position)
                if np.abs(bone_length - expected_bone_length) / expected_bone_length <= tolerance:
                    good_keypoint_pairs.append((keypoint1.name, keypoint2.name))
                    if good_keypoints.get(keypoint1.name) is None:
                        good_keypoints[keypoint1.name] = keypoint1
                    if good_keypoints.get(keypoint2.name) is None:
                        good_keypoints[keypoint2.name] = keypoint2

        # Return a filtered skeleton.
        return Skeleton(good_keypoints, good_keypoint_pairs)

    # PUBLIC METHODS

    def detect_skeletons_2d(self, colour_image: np.ndarray) -> Tuple[List[Skeleton], np.ndarray]:
        """
        Detect 2D skeletons in a colour image using OpenPose.

        :param colour_image:    The colour image.
        :return:                A tuple consisting of the detected 2D skeletons and the OpenPose visualisation
                                of what it detected.
        """
        # Run OpenPose on the colour image.
        datum: Datum = Datum()
        datum.cvInputData = colour_image
        self.__wrapper.emplaceAndPop([datum])

        # If any 2D skeletons were detected:
        if len(datum.poseKeypoints.shape) > 0:
            # Assemble them into an easier-to-use format.
            skeletons: List[Skeleton] = []
            skeleton_count: int = datum.poseKeypoints.shape[0]

            for i in range(skeleton_count):
                pose_keypoints: np.ndarray = datum.poseKeypoints[i, :, :]
                pose_keypoint_count: int = pose_keypoints.shape[0]
                skeleton_keypoints: Dict[str, Skeleton.Keypoint] = {}
                for j in range(pose_keypoint_count):
                    score: float = pose_keypoints[j][2]
                    if score > 0.0:
                        name: str = self.__keypoint_names[j]
                        position: np.ndarray = pose_keypoints[j][:2]
                        skeleton_keypoints[name] = Skeleton.Keypoint(name, position, score)

                skeletons.append(Skeleton(skeleton_keypoints, self.__keypoint_pairs))

            return skeletons, datum.cvOutputData
        else:
            return [], colour_image

    def detect_skeletons_3d(self, colour_image: np.ndarray, ws_points: np.ndarray,
                            mask: np.ndarray) -> Tuple[List[Skeleton], np.ndarray]:
        """
        Detect 3D skeletons in an RGB-D image using OpenPose.

        :param colour_image:    The colour part of the RGB-D image.
        :param ws_points:       The world-space points image obtained from the depth part of the RGB-D image.
        :param mask:            A binary mask indicating which pixels have a valid world-space point.
        :return:                A tuple consisting of the detected 3D skeletons and the OpenPose visualisation
                                of what it detected.
        """
        skeletons_2d, output_image = self.detect_skeletons_2d(colour_image)
        return self.lift_skeletons_to_3d(skeletons_2d, ws_points, mask), output_image

    def lift_skeleton_to_3d(self, skeleton_2d: Skeleton, ws_points: np.ndarray, mask: np.ndarray) -> Skeleton:
        """
        Lift a 2D skeleton to 3D by back-projecting its keypoints into world space.

        :param skeleton_2d:     The 2D skeleton.
        :param ws_points:       The world-space points image.
        :param mask:            A binary mask indicating which pixels have a valid world-space point.
        :return:                The 3D skeleton.
        """
        keypoints_3d: Dict[str, Skeleton.Keypoint] = {}
        height, width = ws_points.shape[:2]

        # For each keypoint in the 2D skeleton:
        for keypoint_name, keypoint_2d in skeleton_2d.keypoints.items():
            # Determine the pixel in the image corresponding to the keypoint.
            x, y = np.round(keypoint_2d.position).astype(int)

            # If the pixel is within the image bounds, and has a valid world-space point:
            # noinspection PyChainedComparisons
            if 0 <= x < width and 0 <= y < height and mask[y, x] != 0:
                # Make the corresponding 3D keypoint and add it to the list.
                keypoints_3d[keypoint_name] = Skeleton.Keypoint(keypoint_name, ws_points[y, x], keypoint_2d.score)

        # Make a 3D skeleton from the list of 3D keypoints, and return it.
        return Skeleton(keypoints_3d, self.__keypoint_pairs)

    def lift_skeletons_to_3d(self, skeletons_2d: List[Skeleton], ws_points: np.ndarray,
                             mask: np.ndarray) -> List[Skeleton]:
        """
        Lift a set of 2D skeletons to 3D by back-projecting their keypoints into world space.

        :param skeletons_2d:    The 2D skeletons.
        :param ws_points:       The world-space points image.
        :param mask:            A binary mask indicating which pixels have a valid world-space point.
        :return:                The 3D skeletons.
        """
        return [self.lift_skeleton_to_3d(skeleton_2d, ws_points, mask) for skeleton_2d in skeletons_2d]

    def terminate(self) -> None:
        """Destroy the detector."""
        self.__wrapper.stop()
