import cv2
import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from OpenGL.GL import *
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Tuple

from smg.opengl import OpenGLMatrixContext, OpenGLUtil
from smg.openni import OpenNICamera
from smg.pyopenpose import SkeletonDetector
from smg.rigging.cameras import SimpleCamera
from smg.rigging.controllers import KeyboardCameraController
from smg.rigging.helpers import CameraPoseConverter
from smg.utility import GeometryUtil


def update_bone_lengths(bone_lengths: Dict[str, List[float]], skeleton: SkeletonDetector.Skeleton) \
        -> Dict[str, List[float]]:
    for keypoint1, keypoint2 in skeleton.bones:
        bone_name: str = str(sorted([keypoint1.name, keypoint2.name]))
        bone_length: float = np.linalg.norm(keypoint1.position - keypoint2.position)
        lengths_for_bone: List[float] = bone_lengths.get(bone_name, [])
        lengths_for_bone.append(bone_length)
        bone_lengths[bone_name] = lengths_for_bone

    print(bone_lengths)

    return bone_lengths


def render_skeleton(skeleton: SkeletonDetector.Skeleton) -> None:
    # Obtained from the keypoints_pose_25.png file.
    bone_colours: Dict[str, np.ndarray] = {
        "['MidHip', 'Neck']": np.array([153., 0., 0.]),
        "['Neck', 'RShoulder']": np.array([153., 51., 0.]),
        "['LShoulder', 'Neck']": np.array([102., 153., 0.]),
        "['RElbow', 'RShoulder']": np.array([153., 102., 0.]),
        "['RElbow', 'RWrist']": np.array([153., 153., 0.]),
        "['LElbow', 'LShoulder']": np.array([51., 153., 0.]),
        "['LElbow', 'LWrist']": np.array([0., 153., 0.]),
        "['MidHip', 'RHip']": np.array([0., 153., 51.]),
        "['RHip', 'RKnee']": np.array([0., 153., 102.]),
        "['RAnkle', 'RKnee']": np.array([0., 153., 153.]),
        "['LHip', 'MidHip']": np.array([0., 102., 153.]),
        "['LHip', 'LKnee']": np.array([0., 51., 153.]),
        "['LAnkle', 'LKnee']": np.array([0., 0., 153.]),
        "['Neck', 'Nose']": np.array([153., 0., 51.]),
        "['Nose', 'REye']": np.array([153., 0., 102.]),
        "['REar', 'REye']": np.array([153., 0., 153.]),
        "['LEye', 'Nose']": np.array([102., 0., 153.]),
        "['LEar', 'LEye']": np.array([51., 0., 153.]),
        "['REar', 'RShoulder']": np.array([0., 0., 255.]),
        "['LEar', 'LShoulder']": np.array([0., 0., 255.]),
        "['LAnkle', 'LBigToe']": np.array([0., 0., 153.]),
        "['LBigToe', 'LSmallToe']": np.array([0., 0., 153.]),
        "['LAnkle', 'LHeel']": np.array([0., 0., 153.]),
        "['RAnkle', 'RBigToe']": np.array([0., 153., 153.]),
        "['RBigToe', 'RSmallToe']": np.array([0., 153., 153.]),
        "['RAnkle', 'RHeel']": np.array([0., 153., 153.])
    }

    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    pos: np.ndarray = np.array([0.0, 0.0, -1.0, 0.0])
    glLightfv(GL_LIGHT0, GL_POSITION, pos)
    glEnable(GL_LIGHT1)
    glLightfv(GL_LIGHT1, GL_DIFFUSE, np.array([1, 1, 1, 1]))
    glLightfv(GL_LIGHT1, GL_SPECULAR, np.array([1, 1, 1, 1]))
    glLightfv(GL_LIGHT1, GL_POSITION, -pos)

    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    for keypoint_name, keypoint in skeleton.keypoints.items():
        glColor3f(1 - keypoint.score, keypoint.score, 0.0)
        OpenGLUtil.render_sphere(keypoint.position, 0.03, slices=10, stacks=10)

    for keypoint1, keypoint2 in skeleton.bones:
        bone_name: str = str(sorted([keypoint1.name, keypoint2.name]))
        bone_colour: Optional[np.ndarray] = bone_colours.get(bone_name)
        if bone_colour is not None:
            bone_colour = bone_colour / 153
            glColor3f(*bone_colour)
            OpenGLUtil.render_cylinder(keypoint1.position, keypoint2.position, 0.025, 0.025, slices=10)

    glDisable(GL_COLOR_MATERIAL)
    glDisable(GL_LIGHTING)


def main() -> None:
    np.set_printoptions(suppress=True)

    # Initialise PyGame and create the window.
    pygame.init()
    window_size: Tuple[int, int] = (640, 480)
    pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("Skeleton Detector")

    # Enable the z-buffer.
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    # Construct the camera controller.
    camera_controller: KeyboardCameraController = KeyboardCameraController(
        SimpleCamera([0, 0, 0], [0, 0, 1], [0, -1, 0]), canonical_angular_speed=0.05, canonical_linear_speed=0.1
    )

    params: Dict[str, Any] = {"model_folder": "D:/openpose-1.6.0/models/"}
    with SkeletonDetector(params) as skeleton_detector:
        with OpenNICamera(mirror_images=True) as camera:
            bone_lengths: Dict[str, List[float]] = {}

            while True:
                # Process any PyGame events.
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        cv2.destroyAllWindows()
                        return

                colour_image, depth_image = camera.get_images()
                skeletons_2d, output_image = skeleton_detector.detect_skeletons_2d(colour_image)

                ws_points: np.ndarray = GeometryUtil.compute_world_points_image_fast(
                    depth_image, np.eye(4), camera.get_depth_intrinsics()
                )

                mask: np.ndarray = np.where(depth_image != 0, 255, 0).astype(np.uint8)

                skeletons_3d: List[SkeletonDetector.Skeleton] = skeleton_detector.lift_skeletons_to_3d(
                    skeletons_2d, ws_points, mask
                )

                # for skeleton_3d in skeletons_3d:
                #     update_bone_lengths(bone_lengths, skeleton_3d)

                depth_image_uc: np.ndarray = np.clip(depth_image * 255 / 5, 0, 255).astype(np.uint8)
                if output_image is not None:
                    blended_image: np.ndarray = np.zeros(colour_image.shape, dtype=np.uint8)
                    for i in range(3):
                        blended_image[:, :, i] = (output_image[:, :, i] * 0.5 + depth_image_uc * 0.5).astype(np.uint8)
                    cv2.imshow("Depth Image", output_image)
                else:
                    cv2.imshow("Depth Image", depth_image_uc)

                c: int = cv2.waitKey(1)
                if c == ord("q"):
                    break

                # Allow the user to control the camera.
                camera_controller.update(pygame.key.get_pressed(), timer() * 1000)

                # Clear the colour and depth buffers.
                glClearColor(1.0, 1.0, 1.0, 1.0)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                # Set the projection matrix.
                with OpenGLMatrixContext(GL_PROJECTION, lambda: OpenGLUtil.set_projection_matrix(
                    camera.get_colour_intrinsics(), *window_size
                )):
                    # Set the model-view matrix.
                    with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.load_matrix(
                        CameraPoseConverter.pose_to_modelview(camera_controller.get_pose())
                    )):
                        # Render a voxel grid.
                        glColor3f(0.0, 0.0, 0.0)
                        OpenGLUtil.render_voxel_grid([-2, -2, -2], [2, 0, 2], [1, 1, 1], dotted=True)

                        # Render the 3D skeletons.
                        for skeleton_3d in skeletons_3d:
                            render_skeleton(skeleton_3d)

                # Swap the front and back buffers.
                pygame.display.flip()


if __name__ == "__main__":
    main()
