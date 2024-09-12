import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, SpatialDropout1D, Flatten, LSTM, Bidirectional, Concatenate, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd

import src.config.data_config as dc
import src.config.model_config as mc



        
class CNNLSTM:
    def __init__(self,  learning_rate=0.001):
        self.learning_rate = learning_rate
        self.model = None
        self.history = None

    def build_model(self):
        input_layer = Input(shape=(self.x_train.shape[1], self.x_train.shape[2]))
        
        # CNN layers
        cnn_layer = self._build_cnn_layers(input_layer)

        # LSTM layers
        lstm_layer = self._build_lstm_layers(input_layer)

        # Combine CNN and LSTM
        combined = Concatenate()([cnn_layer, lstm_layer])

        # Dense layers
        dense_layer = self._build_dense_layers(combined)

        # Output layer
        output_layer = Dense(1, dtype='float32')(dense_layer)

        self.model = Model(inputs=input_layer, outputs=output_layer)

    def _build_cnn_layers(self, input_layer):
        cnn = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(input_layer)
        cnn = BatchNormalization()(cnn)
        cnn = SpatialDropout1D(0.2)(cnn)

        cnn = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(cnn)
        cnn = BatchNormalization()(cnn)
        cnn = SpatialDropout1D(0.2)(cnn)

        cnn = Conv1D(filters=32, kernel_size=7, activation='relu', padding='same')(cnn)
        cnn = BatchNormalization()(cnn)
        return Flatten()(cnn)

    def _build_lstm_layers(self, input_layer):
        lstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(input_layer)
        return Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))(lstm)

    def _build_dense_layers(self, input_layer):
        dense = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(input_layer)
        dense = Dropout(0.5)(dense)
        dense = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(input_layer)
        return Dropout(0.5)(dense)

    def compile_model(self):
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='mae')

    def prepare_data(self, train_set, val_set, features, target, batch_size=32):
        self.x_train, self.y_train = self._prepare_dataset(train_set, features, target)
        self.x_val, self.y_val = self._prepare_dataset(val_set, features, target)

        train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        train_dataset = train_dataset.cache().shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val))
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return train_dataset, val_dataset
    
    def prepare_test_data(self,test_set, features, batch_size=32):
        test_set = test_set[features]
        test_set = test_set.values
        x_test = test_set.reshape((test_set.shape[0], test_set.shape[1], 1))
        
        test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
        test_datase = test_dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        
        return test_dataset

    def _prepare_dataset(self, dataset, features, target):
        x = dataset[features].values
        y = dataset[target].values
        x = x.reshape((x.shape[0], x.shape[1], 1))
        return x, y

    def train(self, train_dataset, val_dataset, epochs=5, patience=10):
        callbacks = [
            EarlyStopping(patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=patience // 2, min_lr=1e-6)
        ]

        self.history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks
        )

    def evaluate(self, test_dataset):
        return self.model.evaluate(test_dataset)

    def predict(self, x):
        return self.model.predict(x)

    def save_model(self, filepath):
        self.model.save(filepath)

    @classmethod
    def load_model(cls, filepath):
        loaded_model = tf.keras.models.load_model(filepath)
        instance = cls(loaded_model.input_shape[1:])
        instance.model = loaded_model
        return instance
