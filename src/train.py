import pandas as pd
from src.cnn_lstm import CNNLSTM
import src.config.data_config as dc
import src.config.model_config as mc

def train_model():
    train_set = pd.read_csv(dc.DATA_PATH / 'train_set.csv')

    rows_per_week = 168 * 30 * 3 * 32
    train_rows = rows_per_week * mc.NUMBER_OF_TRAIN_WEEKS
    input_shape = (train_rows, len(mc.TRAINING_FEATURES))

    train_data = train_set[:train_rows]
    val_data = train_set[train_rows:]

    # Initialize model
    cnnlsttm = CNNLSTM(input_shape)
    train_dataset, val_dataset = cnnlsttm.prepare_data(train_data, val_data, mc.TRAINING_FEATURES, dc.COLUMNS['DLThpvol'])

    cnnlsttm.train(train_dataset, val_dataset)
    loss = cnnlsttm.evaluate(val_dataset)
    print(f'Test Loss: {loss}')

    if mc.SAVE_MODEL:
        cnnlsttm.save_model(mc.MODEL_PATH / mc.MODEL_NAME)

    print("Model training completed.")
