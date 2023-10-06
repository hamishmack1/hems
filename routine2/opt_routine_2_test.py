"""Typical usage of opt_routine_2.py

Author:
    Hamish Mackinlay
"""

from opt_routine_2 import *

base_path = os.path.dirname(__file__)

step_headings, data = read_train_data("global.csv", base_path)

for i, step in enumerate(step_headings):
    train_data, val_data, test_data = prepare_data(data[i])
    build_model(train_data, val_data, step)
    model = keras.models.load_model("timestep_models/" + step + ".keras")
    print(f"Test MAE: {model.evaluate(test_data[:,:4], test_data[:,4])[1]:.3f}")

# mae_history = history.history["mae"]
# val_mae_history = history.history["val_mae"]