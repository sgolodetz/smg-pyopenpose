import numpy as np

from typing import Dict, List, Tuple


class Skeleton:
    """A skeleton."""

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

    # CONSTRUCTOR

    def __init__(self, keypoints: Dict[str, Keypoint], keypoint_pairs: List[Tuple[str, str]]):
        """
        Construct a skeleton.

        :param keypoints:       The keypoints that have been detected for the skeleton.
        :param keypoint_pairs:  Pairs of names denoting keypoints that should be joined by bones.
        """
        self.__keypoints: Dict[str, Skeleton.Keypoint] = keypoints

        # Filter the pairs of names, keeping only those for which both keypoints have been detected.
        self.__keypoint_pairs: List[Tuple[str, str]] = [
            (i, j) for i, j in keypoint_pairs if i in self.__keypoints and j in self.__keypoints
        ]

    # PROPERTIES

    @property
    def bones(self) -> List[Tuple[Keypoint, Keypoint]]:
        """
        Get the bones of the skeleton.

        :return:    The bones of the skeleton, as a list of detected keypoint pairs.
        """
        return [(self.__keypoints[i], self.__keypoints[j]) for i, j in self.__keypoint_pairs]

    @property
    def keypoints(self) -> Dict[str, Keypoint]:
        """
        Get the detected keypoints of the skeleton.

        :return:    The detected keypoints of the skeleton, as a keypoint name -> keypoint map.
        """
        return self.__keypoints
