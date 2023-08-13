# HEMS Optimization Routine 2
# Author: Ha ==h Mackinlay

import os
import numpy as np
from pyomo.environ import *
from tensorflow import keras
from keras import layers
import csv

# Read Training Data

fname = os.path.join("training_data.csv")

with open(fname) as f:
    data = f.read()

lines = data.split("\n")
header = lines[0].split(",")
lines = lines[1:]

# Parse data

# Inputs -> Consumption, Generation
# Outputs -> Grid Import, Grid Export, Battery Expenditure

length = int(len(lines) / 6)

forecast_data = np.zeros((length, 2, len(header) - 2))
grid_import = np.zeros((length, len(header) - 2))
grid_export = np.zeros((length, len(header) - 2))
bat_expend = np.zeros((length, len(header) - 2))
bat_soc = np.zeros((length, len(header) - 2))

counter = 0
for line in lines:
    split_line = line.split(",")
    category = split_line[0]
    values = [float(x) for x in split_line[2:]]

    if category == "GC":
        forecast_data[counter, 0, :] = values
    elif category == "GG":
        forecast_data[counter, 1, :] = values
    elif category == "GI":
        grid_import[counter, :] = values
    elif category == "GE":
        grid_export[counter, :] = values
    elif category == "BC":
        bat_expend[counter, :] = values
    elif category == "SOC":
        bat_soc[counter, :] = values
        counter += 1

# Preparing data

# Problem: Given current battery SOC, and forecast of demand and generation, predict todays optimal energy allocation decisions.

num_train_samples = int(0.5 * length)
num_val_samples = int(0.25 * length)
num_test_samples = length - num_train_samples - num_val_samples

train_forecast = forecast_data[:num_train_samples]
train_grid_import = grid_import[:num_train_samples]
train_grid_export = grid_export[:num_train_samples]
train_bat_expend = bat_expend[:num_train_samples] / 10

val_forecast = forecast_data[num_train_samples:num_train_samples+num_val_samples]
val_grid_import = grid_import[num_train_samples:num_train_samples+num_val_samples]
val_grid_export = grid_export[num_train_samples:num_train_samples+num_val_samples]
val_bat_expend = bat_expend[num_train_samples:num_train_samples+num_val_samples] / 10

test_forecast = forecast_data[num_train_samples+num_val_samples:]
test_grid_import = grid_import[num_train_samples+num_val_samples:]
test_grid_export = grid_export[num_train_samples+num_val_samples:]
test_bat_expend = bat_expend[num_train_samples+num_val_samples:]

# Build Model

forecast = keras.Input(shape=(2, 48), name="forecast")

features = layers.LSTM(48)(forecast)

g_imp = layers.Dense(48, activation="relu", name="grid_import")(features)
g_exp = layers.Dense(48, activation="relu", name="grid_export")(features)
b_exp = layers.Dense(48, activation="tanh", name="bat_expend")(features)

model = keras.Model(inputs=forecast,
                    outputs=[g_imp, g_exp, b_exp])

model.compile(optimizer="rmsprop",
              loss="mse",
              metrics=["mae"])
model.fit([train_forecast],
          [train_grid_import, train_grid_export, train_bat_expend],
          epochs=50,
          validation_data=([val_forecast], [val_grid_import, val_grid_export, val_bat_expend]))
grid_import_preds, grid_export_preds, bat_expend_preds = model.predict([test_forecast])

bat_expend_preds *= bat_expend_preds

print(test_grid_import)
print(grid_import_preds)