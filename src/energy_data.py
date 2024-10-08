import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import pandas as pd
from src.data_processing import EnergyDataset
import src.config.data_config as dc

def prepare_data():
    dataset = EnergyDataset(dc.DATA_PATH)
    dataset.load_data()
    
    if dc.SECOND_APPROACH:
        dataset.process_trainset()
        dataset.process_testset()
        train_set.to_csv(dc.PROCESSED_DATA_PATH / 'train_set_v1.csv', index=False)
        test_set.to_csv(dc.PROCESSED_DATA_PATH / 'test_set_v1.csv', index=False)
    else:
        dataset.process_trainset()
        dataset.process_testset()
        
        full_data = dataset.get_fulldata()
        train_size = dataset.train_set.shape[0]

        train_set = full_data.iloc[:train_size, :]
        test_set = full_data.iloc[train_size:, :]

        train_set.to_csv(dc.PROCESSED_DATA_PATH / 'train_set.csv', index=False)
        test_set.to_csv(dc.PROCESSED_DATA_PATH / 'test_set.csv', index=False)

    print("Data preparation completed and saved.")
