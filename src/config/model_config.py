from typing import Dict, List
from pathlib import Path

NUMBER_OF_TRAIN_WEEKS: float = 4

TRAINING_FEATURES : List[str] = ['week', 'hour', 'gnodeb', 'cell', 'beam']

SAVE_MODEL : bool = False

MODEL_PATH: Path = Path("../experiments/models")
MODEL_NAME = 'benchmark_model.pk'