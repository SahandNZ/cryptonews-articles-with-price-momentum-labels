import os
import pathlib
from typing import List


def create_directory_recursively(path: str) -> bool:
    if not os.path.exists(path):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def get_sub_directories(path: str) -> List[str]:
    return os.listdir(path)
