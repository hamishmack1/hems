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

consumption = np.zeros((length, len(header) - 2, 1))
generation = np.zeros((length, len(header) - 2, 1))
grid_import = np.zeros((length, len(header) - 2, 1))
grid_export = np.zeros((length, len(header) - 2, 1))
bat_expend = np.zeros((length, len(header) - 2, 1))
bat_soc = np.zeros((length, len(header) - 2, 1))

counter = 0
for line in lines:
    split_line = line.split(",")
    category = split_line[0]
    values = [float(x) for x in split_line[2:]]

    for i in range(len(values)):
        if category == "GC":
            consumption[counter,i] = values[i]
        elif category == "GG":
            generation[counter, i] = values[i]
        elif category == "GI":
            grid_import[counter, i] = values[i]
        elif category == "GE":
            grid_export[counter, i] = values[i]
        elif category == "BC":
            bat_expend[counter, i] = values[i]
        elif category == "SOC":
            bat_soc[counter, i] = values[i]
            if i == len(values) - 1:
                counter += 1

# Preparing data

# Problem: Given current battery SOC, and forecast of demand and generation for the next week, predict todays optimal energy allocation decisions.

num_train_samples = int(0.5 * length)
num_val_samples = int(0.25 * length)
num_test_samples = length - num_train_samples - num_val_samples

train_consumption = consumption[:num_train_samples]
train_generation = generation[:num_train_samples]
train_grid_import = grid_import[:num_train_samples]
train_grid_export = grid_export[:num_train_samples]
train_bat_expend = bat_expend[:num_train_samples]

val_consumption = consumption[num_train_samples:num_train_samples+num_val_samples]
val_generation = generation[num_train_samples:num_train_samples+num_val_samples]
val_grid_import = grid_import[num_train_samples:num_train_samples+num_val_samples]
val_grid_export = grid_export[num_train_samples:num_train_samples+num_val_samples]
val_bat_expend = bat_expend[num_train_samples:num_train_samples+num_val_samples]

test_consumption = consumption[num_train_samples+num_val_samples:]
test_generation = generation[num_train_samples+num_val_samples:]
test_grid_import = grid_import[num_train_samples+num_val_samples:]
test_grid_export = grid_export[num_train_samples+num_val_samples:]
test_bat_expend = bat_expend[num_train_samples+num_val_samples:]

# Build Model

con = keras.Input(shape=(48, 1), name="consumption")
gen = keras.Input(shape=(48, 1), name="generation")

features = layers.Concatenate()([con, gen])
features = layers.LSTM(16, input_shape=(None, 48, 1))(features)
features = layers.Dropout(0.5)(features)

# g_imp = layers.Dense(48, activation="sigmoid", name="grid_import")(features)
# g_exp = layers.Dense(48, activation="sigmoid", name="grid_export")(features)
# b_exp = layers.Dense(48, activation="sigmoid", name="bat_expend")(features)

g_imp = layers.Dense(48, activation="relu", name="grid_import")(features)
g_exp = layers.Dense(48, activation="relu", name="grid_export")(features)
b_exp = layers.Dense(48, activation="relu", name="bat_expend")(features)



model = keras.Model(inputs=[con, gen],
                    outputs=[g_imp, g_exp, b_exp])

callbacks = [
    keras.callbacks.ModelCheckpoint("hems",
                                    save_best_only=True)
]
model.compile(optimizer="rmsprop",
              loss="mse",
              metrics=["mae"])
model.fit([train_consumption, train_generation],
          [train_grid_import, train_grid_export, train_bat_expend],
          epochs=10,
          validation_data=([val_consumption, val_generation], [val_grid_import, val_grid_export, val_bat_expend]),
          callbacks=callbacks)



# model.evaluate([val_consumption, val_generation],
#                [val_grid_import, val_grid_export, val_bat_expend])

grid_import_preds, grid_export_preds, bat_expend_preds = model.predict(
    [test_consumption, test_generation])

print(test_grid_import)
print(grid_import_preds)