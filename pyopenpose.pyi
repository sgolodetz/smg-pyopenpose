import numpy as np

from typing import Any, Dict, List


# FUNCTIONS

# noinspection PyPep8Naming
def getPoseBodyPartMapping(poseModel: PoseModel) -> Dict[int, str]: ...

# noinspection PyPep8Naming
def getPosePartPairs(poseModel: PoseModel) -> List[int]: ...


# CLASSES

class Datum:
    cvInputData: np.ndarray
    cvOutputData: np.ndarray
    poseKeypoints: np.ndarray

    def __init__(self): ...


# noinspection PyPep8Naming
class WrapperPython:
    def configure(self, params: Dict[str, Any]): ...
    def emplaceAndPop(self, l: List[Datum]) -> None: ...
    def execute(self) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def waitAndEmplace(self, l: List[Datum]) -> None: ...
    def waitAndPop(self, l: List[Datum]) -> bool: ...


# ENUMERATIONS

class PoseModel(int):
    pass

BODY_25: PoseModel
COCO_18: PoseModel
MPI_15: PoseModel
MPI_15_4: PoseModel
BODY_25B: PoseModel
BODY_135: PoseModel
