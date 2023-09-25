"""Typical usage of opt_routine_2.py

Author:
    Hamish Mackinlay
"""

from opt_routine_2 import *

base_path = os.path.dirname(__file__)

training_data = read_train_data("global.csv", base_path)
train_data, val_data, test_data = prepare_data(training_data)

pfa_model = build_pfa_model()

history = pfa_model.fit(train_data[:4],
                        train_data[4],
                        epochs=200,
                        validation_data=(val_data[:4], val_data[4]))

grid_power = pfa_model.predict(test_data[:4])

mae_history = history.history["mae"]
val_mae_history = history.history["val_mae"]

