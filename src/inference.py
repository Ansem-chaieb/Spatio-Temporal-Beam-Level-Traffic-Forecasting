import pandas as pd
from src.cnn_lstm import CNNLSTM
import src.config.data_config as dc
import src.config.model_config as mc

def run_inference():
    test_set = pd.read_csv(dc.DATA_PATH / 'test_set.csv')

    input_shape = (None, len(mc.TRAINING_FEATURES))
    cnnlsttm = CNNLSTM(input_shape)
    cnnlsttm.load_model(mc.MODEL_PATH / mc.MODEL_NAME)

    test_dataset = cnnlsttm.prepare_test_data(test_set, mc.TRAINING_FEATURES)
    predictions = cnnlsttm.predict(test_dataset)

    # Create submission file
    full_path = dc.DATA_PATH / dc.SUB_FILE
    submission = pd.read_csv(full_path)
    submission['Target'] = predictions

    submission.to_csv(f"{mc.MODEL_PATH / mc.MODEL_NAME}.csv", index=False)
    print("Inference completed and submission file generated.")
