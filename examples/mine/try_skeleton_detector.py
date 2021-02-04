import cv2
import numpy as np
import smg.pyopenpose as op

from typing import Any, Dict, List

from smg.openni import OpenNICamera
from smg.pyopenpose import SkeletonDetector


def main() -> None:
    np.set_printoptions(suppress=True)

    params: Dict[str, Any] = {"model_folder": "D:/openpose-1.6.0/models/"}
    with SkeletonDetector(params) as skeleton_detector:
        with OpenNICamera(mirror_images=True) as camera:
            while True:
                colour_image, depth_image = camera.get_images()
                skeletons_2d: List[SkeletonDetector.Skeleton2D] = skeleton_detector.detect_skeletons_2d(
                    colour_image, visualise_output=True
                )

                c: int = cv2.waitKey(1)
                if c == ord("q"):
                    break


if __name__ == "__main__":
    main()
