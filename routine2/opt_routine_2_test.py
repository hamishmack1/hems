"""Typical usage of opt_routine_2.py

Author:
    Hamish Mackinlay
"""

from opt_routine_2 import *

base_path = os.path.dirname(__file__)

# build_timestep_models("global.csv", base_path)

step_headings, data = read_train_data("global.csv", base_path)

train_data, val_data, test_data = prepare_data(data[18])
history = build_model(train_data, val_data, step_headings[18])
model = keras.models.load_model("timestep_models/" + step_headings[18] + ".keras")
print(f"Test MAE: {model.evaluate(test_data[:,:4], test_data[:,4:6])[1]:.3f}")


grid_power_preds = model.predict(test_data[:,:4])
print(test_data[:,4:6])
print(grid_power_preds)

# print(test_data[:,4])
# print(grid_power_preds)

# mae_history = history.history["mae"]
# val_mae_history = history.history["val_mae"]

# plt.plot(range(1, len(mae_history) + 1), mae_history, label="training")
# plt.plot(range(1, len(val_mae_history) + 1), val_mae_history, label="validation")
# plt.xlabel("Epochs")
# plt.ylabel("MAE")
# plt.legend()
# plt.show()

start = 0
end = None

gi = grid_power_preds[0].flatten()[start:end] - grid_power_preds[1].flatten()[start:end] 
gi_real = test_data[:,4][start:end] - test_data[:,5][start:end]

plt.plot(range(len(gi)), gi, label="Prediction")
plt.plot(range(len(gi)), gi_real, label="Actual")
plt.xlabel("Time")
plt.ylabel("Power (kWh)")
plt.legend()
plt.show()