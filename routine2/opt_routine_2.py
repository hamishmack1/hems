# HEMS Optimization Routine 2
# Author: Hamish Mackinlay

import os
import numpy as np
from pyomo.environ import *
from tensorflow import keras
from keras import layers

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

length = int(len(lines) / 4)

consumption = np.zeros((length, len(header) - 2))
generation = np.zeros((length, len(header) - 2))
tariff = np.zeros((length, len(header) - 2))
grid_power = np.zeros((length, len(header) - 2))

counter = 0
for line in lines:
    split_line = line.split(",")
    category = split_line[0]
    values = [float(x) for x in split_line[2:]]

    if category == "GC":
        consumption[counter, :] = values
    elif category == "GG":
        generation[counter, :] = values
    elif category == "ToU":
        tariff[counter, :] = values
    elif category == "GP":
        grid_power[counter, :] = values
        counter += 1

# Preparing data

# Problem: Given current battery SOC, and forecast of demand and generation, predict todays optimal energy allocation decisions.

num_train_samples = int(0.5 * length)
num_val_samples = int(0.25 * length)
num_test_samples = length - num_train_samples - num_val_samples

train_consumption = consumption[:num_train_samples]
train_generation = generation[:num_train_samples]
train_tariff = tariff[:num_train_samples]
train_grid_power = grid_power[:num_train_samples]

val_consumption = consumption[num_train_samples:num_train_samples+num_val_samples]
val_generation = generation[num_train_samples:num_train_samples+num_val_samples]
val_tariff = tariff[num_train_samples:num_train_samples+num_val_samples]
val_grid_power = grid_power[num_train_samples:num_train_samples+num_val_samples]

test_consumption = consumption[num_train_samples+num_val_samples:]
test_generation = generation[num_train_samples+num_val_samples:]
test_tariff = tariff[num_train_samples+num_val_samples:]
test_grid_power = grid_power[num_train_samples+num_val_samples:]

# Build Model

con = keras.Input(shape=(48,), name="forecast")
gen = keras.Input(shape=(48,), name="generation")
tar = keras.Input(shape=(48,), name="tariff")

features = layers.Concatenate()([con, gen, tar])
features = layers.Dense(96)(features)

g_power = layers.Dense(48, activation="relu", name="grid_power")(features)

model = keras.Model(inputs=[con, gen, tar],
                    outputs=[g_power])

model.compile(optimizer="rmsprop",
              loss="mse",
              metrics=["mae"])
history = model.fit([train_consumption, train_generation, train_tariff],
          [train_grid_power],
          epochs=200,
          validation_data=([val_consumption, val_generation, val_tariff], [val_grid_power]))
grid_power_preds = model.predict([test_consumption, test_generation, test_tariff])

mae_history = history.history["mae"]
val_mae_history = history.history["val_mae"]