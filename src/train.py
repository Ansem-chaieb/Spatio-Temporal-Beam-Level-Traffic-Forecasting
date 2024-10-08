import os

import pandas as pd
import numpy as np
from src.cnn_lstm import CNNLSTM
from src.data_processing import EnergyDataset
import src.config.data_config as dc
import src.config.model_config as mc
from icecream import ic
import warnings
warnings.filterwarnings('ignore')


def train_model():
    if dc.SECOND_APPROACH:
        dataset = EnergyDataset(dc.DATA_PATH)
        dataset.load_data()
    
        train_set = pd.read_csv(dc.PROCESSED_DATA_PATH / 'train_set_v1.csv')
        test_set = pd.read_csv(dc.DATA_PATH / 'test_set_v1.csv')
        
        full_path = dc.DATA_PATH / dc.SUB_FILE
        submission = pd.read_csv(full_path)
        submission = submission.set_index ( "ID", drop = False )
        submission[ "Target" ] = submission["Target"].astype ("float16")
        
        n_base = 30
        n_cell =  3
        n_beam = 32
        rs     =  123
        
        cnnlsttm = CNNLSTM()
        cnnlsttm.build_model()
        cnnlsttm.compile_model()

        for base in range ( n_base ) :
          for cell in range ( n_cell ) :
            for beam in range ( n_beam ) :
                rs     += 1
                mod_col = f"{ base }_{ cell }_{ beam }"
                
                train_dataset, _ = cnnlsttm.prepare_data(train_data, None, mc.TRAINING_FEATURES, dataset.data['DLPRB'][ mod_col ] )
                cnnlsttm.train(train_dataset, val_dataset=None)
                
                pred = cnnlsttm.model.predict(test_set)
                pred = np.clip(pred.flatten(), 0, 255)
                for k in range ( 168 ) :
                    submission.at[ "traffic_DLThpVol_test_5w-6w_"   + str (       k     ) + "_" + mod_col, "Target" ] = pred[   k     ]
                    submission.at[ "traffic_DLThpVol_test_10w-11w_" + str ( 168 - k - 1 ) + "_" + mod_col, "Target" ] = pred[ - k - 1 ]

        submission.to_csv(f"{mc.MODEL_PATH / mc.MODEL_NAME}.csv", index=False)
    else: 
        train_set = pd.read_csv(dc.PROCESSED_DATA_PATH / 'train_set.csv')

        rows_per_week = 168 * 30 * 3 * 32
        train_rows = rows_per_week * mc.NUMBER_OF_TRAIN_WEEKS

        train_data = train_set[:train_rows]
        val_data = train_set[train_rows:]

        # Initialize model
        cnnlsttm = CNNLSTM()
        train_dataset, val_dataset = cnnlsttm.prepare_data(train_data, val_data, mc.TRAINING_FEATURES, dc.COLUMNS['DLThpvol'])

        cnnlsttm.build_model()
        ic(cnnlsttm.model.summary())
        cnnlsttm.compile_model()
        cnnlsttm.train(train_dataset, val_dataset)
        loss = cnnlsttm.evaluate(val_dataset)
        print(f'Test Loss: {loss}')

        if mc.SAVE_MODEL:
            cnnlsttm.save_model(mc.MODEL_PATH / mc.MODEL_NAME)

    print("Model training completed.")
