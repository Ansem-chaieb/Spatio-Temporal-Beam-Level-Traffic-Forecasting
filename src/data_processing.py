import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from typing import List, Dict
import pandas as pd
from pathlib import Path
from icecream import ic

import src.config.data_config as dc


class EnergyDataset:
    def __init__(self, data_path: str = dc.DATA_PATH):
        self.data_path = Path(data_path)
        self.data: Dict[str, pd.DataFrame] = {}
        self.train_set: pd.DataFrame = pd.DataFrame()
        self.test_set: pd.DataFrame = pd.DataFrame()

    def load_data(self):
        for key, file_name in dc.DATA_FILES.items():
            full_path = self.data_path / file_name
            self.data[key] = pd.read_csv(full_path).drop("Unnamed: 0", axis=1)

    def process_trainset(self):
        for key, df in self.data.items():
            column_name = dc.COLUMNS[key]
            self.data[key] = self._melt_and_add_hour(
                df, dc.COLUMNS["GNODEB_CELL_BEAM"], column_name
            )
            self.data[key] = self._split_gnodeb_cell_beam(self.data[key])
      
        self.train_set = pd.concat([self.data['DLPRB'],self.data['DLThptime']['DLThptime']],axis = 1)
        self.train_set = pd.concat([self.train_set,self.data['DLThpvol']['DLThpvol']],axis = 1)
        self.train_set = pd.concat([self.train_set,self.data['MR_number']['MR_number']],axis = 1)
      
        columns_to_keep = (
            [dc.COLUMNS["GNODEB_CELL_BEAM"]]
            + dc.CATEGORICAL_FEATURES
            + [dc.COLUMNS['HOUR']]
            + dc.NUMERICAL_FEATURES
        )
        self.train_set = self.train_set[columns_to_keep]
        
        rows_per_week = 168 * 30 * 3 * 32
        self.train_set[dc.COLUMNS["WEEK"]] = (self.train_set.index // rows_per_week) + 1

    def process_testset(self):
        test_hours = list(range(168 * 5, 168 * 6)) + list(range(168 * 10, 168 * 11))
        test_ids = [
            (hour, gnodeb, cell, beam)
            for hour in test_hours
            for gnodeb in range(30)
            for cell in range(3)
            for beam in range(32)
        ]

        self.test_set = pd.DataFrame(
            test_ids, columns=[dc.COLUMNS["HOUR"]] + dc.CATEGORICAL_FEATURES
        )
        self.test_set[dc.COLUMNS["ID"]] = self.test_set.apply(self._create_id, axis=1)
        self.test_set[dc.COLUMNS["HOUR"]] = self.test_set[dc.COLUMNS["HOUR"]] % 168
        self.test_set[dc.COLUMNS["WEEK"]] = 6
        self.test_set.loc[len(test_ids) // 2 :, dc.COLUMNS["WEEK"]] = 11

    def get_fulldata(self) -> pd.DataFrame:
        ic(self.train_set.columns)
        ic(self.train_set)
        ic(self.test_set.columns)
        ic(self.test_set)
        return pd.concat([self.train_set, self.test_set], axis=0)

    @staticmethod
    def _melt_and_add_hour(
        df: pd.DataFrame, var_name: str, value_name: str
    ) -> pd.DataFrame:
        df = df.melt(var_name=var_name, value_name=value_name)
        df[dc.COLUMNS["HOUR"]] = df.index % 168
        return df

    @staticmethod
    def _split_gnodeb_cell_beam(df: pd.DataFrame) -> pd.DataFrame:
        df[dc.CATEGORICAL_FEATURES] = df[dc.COLUMNS["GNODEB_CELL_BEAM"]].str.split(
            "_", expand=True
        )
        for col in dc.CATEGORICAL_FEATURES:
            df[col] = df[col].astype(int)
        return df

    @staticmethod
    def _create_id(row: pd.Series) -> str:
        week_range = "5w-6w" if row[dc.COLUMNS["HOUR"]] < 168 * 10 else "10w-11w"
        return f"traffic_DLThpVol_test_{week_range}_{row[dc.COLUMNS['HOUR']]}_{row[dc.COLUMNS['GNODEB']]}_{row[dc.COLUMNS['CELL']]}_{row[dc.COLUMNS['BEAM']]}"


