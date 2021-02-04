from __future__ import annotations

import cv2
import numpy as np

from typing import Any, Dict, List, Optional, Tuple

from ..cpp.pyopenpose import *


class SkeletonDetector:
    """A 3D skeleton detector based on OpenPose."""

    # NESTED TYPES

    class Keypoint:
        """A keypoint detected by OpenPose."""

        # CONSTRUCTOR

        def __init__(self, name: str, position: Tuple[float, float], score: float):
            self.__name: str = name
            self.__position: Tuple[float, float] = position
            self.__score: float = score

        # PROPERTIES

        @property
        def name(self) -> str:
            return self.__name

        @property
        def position(self) -> Tuple[float, float]:
            return self.__position

        @property
        def score(self) -> float:
            return self.__score

    class Skeleton2D:
        """A 2D skeleton."""

        # CONSTRUCTOR

        def __init__(self, keypoints: Dict[str, SkeletonDetector.Keypoint]):
            self.__keypoints: Dict[str, SkeletonDetector.Keypoint] = keypoints

        # PROPERTIES

        @property
        def keypoints(self) -> Dict[str, SkeletonDetector.Keypoint]:
            return self.__keypoints

    # CONSTRUCTOR

    def __init__(self, params: Dict[str, Any]):
        self.__wrapper: WrapperPython = WrapperPython()
        self.__wrapper.configure(params)
        self.__wrapper.start()

        # TODO: Support other pose models in the future.
        self.__pose_model: PoseModel = BODY_25
        self.__keypoint_names: Dict[int, str] = getPoseBodyPartMapping(self.__pose_model)

    # SPECIAL METHODS

    def __enter__(self):
        """No-op (needed to allow the detector's lifetime to be managed by a with statement)."""
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Destroy the detector at the end of the with statement that's used to manage its lifetime."""
        self.terminate()

    # PUBLIC METHODS

    def detect_skeletons_2d(self, image: np.ndarray, *, visualise_output: bool = True) -> Optional[List[Skeleton2D]]:
        datum: Datum = Datum()
        datum.cvInputData = image
        self.__wrapper.emplaceAndPop([datum])

        if visualise_output:
            cv2.imshow("Detected Keypoints", datum.cvOutputData)

        if len(datum.poseKeypoints.shape) > 0:
            skeletons: List[SkeletonDetector.Skeleton2D] = []
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
                            keypoint_name, tuple(pose_keypoints[j][:2]), keypoint_score
                        )
                skeletons.append(SkeletonDetector.Skeleton2D(skeleton_keypoints))

            return skeletons
        else:
            return None

    def terminate(self) -> None:
        self.__wrapper.stop()
