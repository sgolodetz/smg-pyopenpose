import cv2
import numpy as np

from typing import Any, Dict, List

from smg.openni import OpenNICamera
from smg.pyopenpose import SkeletonDetector
from smg.utility import GeometryUtil


def main() -> None:
    np.set_printoptions(suppress=True)

    params: Dict[str, Any] = {"model_folder": "D:/openpose-1.6.0/models/"}
    with SkeletonDetector(params) as skeleton_detector:
        with OpenNICamera(mirror_images=True) as camera:
            while True:
                colour_image, depth_image = camera.get_images()
                skeletons_2d: List[SkeletonDetector.Skeleton] = skeleton_detector.detect_skeletons_2d(
                    colour_image, visualise_output=True
                )

                ws_points: np.ndarray = GeometryUtil.compute_world_points_image_fast(
                    depth_image, np.eye(4), camera.get_depth_intrinsics()
                )

                skeletons_3d: List[SkeletonDetector.Skeleton] = skeleton_detector.lift_skeletons_to_3d(
                    skeletons_2d, ws_points
                )

                cv2.imshow("Depth Image", depth_image / 2)

                c: int = cv2.waitKey(1)
                if c == ord("q"):
                    break


if __name__ == "__main__":
    main()
