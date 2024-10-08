from typing import Dict, List
from pathlib import Path


DATA_PATH: Path = Path("./data/raw")
PROCESSED_DATA_PATH: Path = Path("./data/processed")

DATA_FILES: Dict[str, str] = {
    "DLPRB": "traffic_DLPRB.csv",
    "DLThptime": "traffic_DLThpTime.csv",
    "DLThpvol": "traffic_DLThpVol.csv",
    "MR_number": "traffic_MR_number.csv",
}

SUB_FILE : str = 'SampleSubmission.csv'

SECOND_APPROACH: bool = True


COLUMNS: Dict[str, str] = {
    "GNODEB": "gnodeb",
    "CELL": "cell",
    "BEAM": "beam",
    "GNODEB_CELL_BEAM": "gnodeb_cell_beam",
    "HOUR": "hour",
    "DLPRB": "DLPRB",
    "DLThptime": "DLThptime",
    "DLThpvol": "DLThpvol",
    "MR_number": "MR_number",
    "WEEK": "week",
    "ID": "ID",
}


CATEGORICAL_FEATURES: List[str] = [COLUMNS["GNODEB"],COLUMNS["CELL"], COLUMNS["BEAM"]]
NUMERICAL_FEATURES: List[str] = [
    COLUMNS["DLPRB"],
    COLUMNS["DLThptime"],
    COLUMNS["DLThpvol"],
    COLUMNS["MR_number"],
]

PCA_FEATURES: List[str] = NUMERICAL_FEATURES  # Adjust this list as needed
N_PCA_COMPONENTS: int = 5

FINAL_PCA_FEATURES: List[str] = (
    [COLUMNS["HOUR"], COLUMNS["WEEK"]]
    + [f"PC{i+1}" for i in range(N_PCA_COMPONENTS)]
    + ["mean_traffic_per_gnodeb", "std_traffic_per_gnodeb"]
)

TARGET_COLUMN: str = "DLThpvol"
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42
