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

    # PUBLIC METHODS

    def detect_skeletons_2d(self, colour_image: np.ndarray) -> Tuple[List[Skeleton], np.ndarray]:
        """
        Detect 2D skeletons in the specified colour image using OpenPose.

        :param colour_image:    The colour image.
        :return:                A tuple consisting of the detected 2D skeletons and the OpenPose visualisation
                                of what's been detected.
        """
        datum: Datum = Datum()
        datum.cvInputData = colour_image
        self.__wrapper.emplaceAndPop([datum])

        if len(datum.poseKeypoints.shape) > 0:
            skeletons: List[SkeletonDetector.Skeleton] = []
            skeleton_count: int = datum.poseKeypoints.shape[0]

            for i in range(skeleton_count):
                pose_keypoints: np.ndarray = datum.poseKeypoints[i, :, :]
                pose_keypoint_count: int = pose_keypoints.shape[0]
                skeleton_keypoints: Dict[str, SkeletonDetector.Keypoint] = {}
                for j in range(pose_keypoint_count):
                    keypoint_name: str = self.__keypoint_names[j]
                    keypoint_score: float = pose_keypoints[j][2]
                    if keypoint_score > 0.0:
                        skeleton_keypoints[keypoint_name] = SkeletonDetector.Keypoint(
                            keypoint_name, pose_keypoints[j][:2], keypoint_score
                        )
                skeletons.append(SkeletonDetector.Skeleton(skeleton_keypoints, self.__keypoint_pairs))

            return skeletons, datum.cvOutputData
        else:
            return [], colour_image

    def detect_skeletons_3d(self, colour_image: np.ndarray, ws_points: np.ndarray,
                            mask: np.ndarray) -> Tuple[List[Skeleton], np.ndarray]:
        skeletons_2d, output_image = self.detect_skeletons_2d(colour_image)
        return self.lift_skeletons_to_3d(skeletons_2d, ws_points, mask), output_image

    def lift_skeleton_to_3d(self, skeleton_2d: Skeleton, ws_points: np.ndarray, mask: np.ndarray) -> Skeleton:
        keypoints_3d: Dict[str, SkeletonDetector.Keypoint] = {}

        height, width = ws_points.shape[:2]

        # noinspection PyUnusedLocal
        keypoint_name: str
        # noinspection PyUnusedLocal
        keypoint_2d: SkeletonDetector.Keypoint

        for keypoint_name, keypoint_2d in skeleton_2d.keypoints.items():
            x, y = np.round(keypoint_2d.position).astype(int)

            # noinspection PyChainedComparisons
            if 0 <= x < width and 0 <= y < height and mask[y, x] != 0:
                keypoints_3d[keypoint_name] = SkeletonDetector.Keypoint(
                    keypoint_name, ws_points[y, x], keypoint_2d.score
                )

        return SkeletonDetector.Skeleton(keypoints_3d, self.__keypoint_pairs)

    def lift_skeletons_to_3d(self, skeletons_2d: List[Skeleton], ws_points: np.ndarray,
                             mask: np.ndarray) -> List[Skeleton]:
        return [self.lift_skeleton_to_3d(skeleton_2d, ws_points, mask) for skeleton_2d in skeletons_2d]

    def terminate(self) -> None:
        """Destroy the detector."""
        self.__wrapper.stop()
