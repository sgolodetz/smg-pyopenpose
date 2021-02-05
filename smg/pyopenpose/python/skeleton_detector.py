from __future__ import annotations

import numpy as np

from typing import Any, Dict, List, Optional, Tuple

from ..cpp.pyopenpose import *


class SkeletonDetector:
    """A 3D skeleton detector based on OpenPose."""

    # NESTED TYPES

    class Keypoint:
        """A keypoint (either 2D or 3D)."""

        # CONSTRUCTOR

        def __init__(self, name: str, position: np.ndarray, score: float):
            """
            Construct a keypoint.

            :param name:        The name of the keypoint.
            :param position:    The position of the keypoint (either 2D or 3D).
            :param score:       The score that OpenPose assigned to the keypoint (a float in [0,1]).
            """
            self.__name: str = name
            self.__position: np.ndarray = position
            self.__score: float = score

        # PROPERTIES

        @property
        def name(self) -> str:
            """
            Get the name of the keypoint.

            :return:    The name of the keypoint.
            """
            return self.__name

        @property
        def position(self) -> np.ndarray:
            """
            Get the position of the keypoint.

            :return:    The position of the keypoint.
            """
            return self.__position

        @property
        def score(self) -> float:
            """
            Get the score that OpenPose assigned to the keypoint.

            :return:    The score that OpenPose assigned to the keypoint (a float in [0,1]).
            """
            return self.__score

    class Skeleton:
        """A skeleton."""

        # CONSTRUCTOR

        def __init__(self, keypoints: Dict[str, SkeletonDetector.Keypoint], keypoint_pairs: List[Tuple[str, str]]):
            """
            Construct a skeleton.

            :param keypoints:       The keypoints that have been detected for the skeleton.
            :param keypoint_pairs:  Pairs of names denoting keypoints that should be joined by bones.
            """
            self.__keypoints: Dict[str, SkeletonDetector.Keypoint] = keypoints

            # Filter the pairs of names, keeping only those for which both keypoints have been detected.
            self.__keypoint_pairs: List[Tuple[str, str]] = [
                (i, j) for i, j in keypoint_pairs if i in self.__keypoints and j in self.__keypoints
            ]

        # PROPERTIES

        @property
        def bones(self) -> List[Tuple[SkeletonDetector.Keypoint, SkeletonDetector.Keypoint]]:
            """
            Get the bones of the skeleton.

            :return:    The bones of the skeleton, as a list of detected keypoint pairs.
            """
            return [(self.__keypoints[i], self.__keypoints[j]) for i, j in self.__keypoint_pairs]

        @property
        def keypoints(self) -> Dict[str, SkeletonDetector.Keypoint]:
            """
            Get the detected keypoints of the skeleton.

            :return:    The detected keypoints of the skeleton, as a keypoint name -> keypoint map.
            """
            return self.__keypoints

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
    def make_bone_key(keypoint1: Keypoint, keypoint2: Keypoint) -> Tuple[str, str]:
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
                         tolerance: float = 0.1) -> Skeleton:
        """
        Remove from the specified skeleton any bone whose estimated length is different from its expected length
        by more than the specified tolerance (in m).

        :param skeleton:                The skeleton.
        :param expected_bone_lengths:   The expected lengths of different bones (in m).
        :param tolerance:               The maximum amount by which the estimated length of a bone can differ from
                                        its expected length without the bone being removed.
        :return:                        A copy of the skeleton from which the bad bones have been removed.
        """
        good_keypoint_pairs: List[Tuple[str, str]] = []

        for keypoint1, keypoint2 in skeleton.bones:
            bone_key: Tuple[str, str] = SkeletonDetector.make_bone_key(keypoint1, keypoint2)
            expected_bone_length: Optional[float] = expected_bone_lengths.get(bone_key)
            if expected_bone_length is not None:
                bone_length: float = np.linalg.norm(keypoint1.position - keypoint2.position)
                if np.abs(bone_length - expected_bone_length) <= tolerance:
                    good_keypoint_pairs.append((keypoint1.name, keypoint2.name))

        return SkeletonDetector.Skeleton(skeleton.keypoints, good_keypoint_pairs)

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
            skeletons: List[SkeletonDetector.Skeleton] = []
            skeleton_count: int = datum.poseKeypoints.shape[0]

            for i in range(skeleton_count):
                pose_keypoints: np.ndarray = datum.poseKeypoints[i, :, :]
                pose_keypoint_count: int = pose_keypoints.shape[0]
                skeleton_keypoints: Dict[str, SkeletonDetector.Keypoint] = {}
                for j in range(pose_keypoint_count):
                    score: float = pose_keypoints[j][2]
                    if score > 0.0:
                        name: str = self.__keypoint_names[j]
                        position: np.ndarray = pose_keypoints[j][:2]
                        skeleton_keypoints[name] = SkeletonDetector.Keypoint(name, position, score)

                skeletons.append(SkeletonDetector.Skeleton(skeleton_keypoints, self.__keypoint_pairs))

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
        keypoints_3d: Dict[str, SkeletonDetector.Keypoint] = {}
        height, width = ws_points.shape[:2]

        # For each keypoint in the 2D skeleton:
        for keypoint_name, keypoint_2d in skeleton_2d.keypoints.items():
            # Determine the pixel in the image corresponding to the keypoint.
            x, y = np.round(keypoint_2d.position).astype(int)

            # If the pixel is within the image bounds, and has a valid world-space point:
            # noinspection PyChainedComparisons
            if 0 <= x < width and 0 <= y < height and mask[y, x] != 0:
                # Make the corresponding 3D keypoint and add it to the list.
                keypoints_3d[keypoint_name] = SkeletonDetector.Keypoint(
                    keypoint_name, ws_points[y, x], keypoint_2d.score
                )

        # Make a 3D skeleton from the list of 3D keypoints, and return it.
        return SkeletonDetector.Skeleton(keypoints_3d, self.__keypoint_pairs)

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
