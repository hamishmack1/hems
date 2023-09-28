"""Typical usage of opt_routine_2.py

Author:
    Hamish Mackinlay
"""

from opt_routine_2 import *

base_path = os.path.dirname(__file__)

# Dense model preparation
# train_data, val_data, test_data = prepare_data(read_train_data("global.csv", base_path))
# inputs = train_data[:4]
# outputs = train_data[4]

# dense_pfa_model = build_dense_pfa_model()
# model = dense_pfa_model

# LSTM model preparation
train_data, val_data, test_data = prepare_data_lstm(read_train_data_lstm("global.csv", base_path))
inputs = train_data[0]
outputs = train_data[1]
lstm_pfa_model = build_lstm_pfa_model()
model = lstm_pfa_model

history = model.fit(inputs,
                    outputs,
                    epochs=300,
                    validation_data=val_data)

# grid_power = dense_pfa_model.predict(test_data[:4])

mae_history = history.history["mae"]
val_mae_history = history.history["val_mae"]