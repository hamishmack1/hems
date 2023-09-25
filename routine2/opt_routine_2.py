# HEMS Optimization Routine 2
# Author: Hamish Mackinlay

import os
import numpy as np
from pyomo.environ import *
from tensorflow import keras
from keras import layers

def read_train_data(fname, base_path):
    """Reads training data and intialises variables.

    Args:
        fname: String specifying file name.
        base_path: String specifying path where script is executed.

    Returns:
        2-d arrays for each variable respectively.
    """

    fname = os.path.join(base_path, "..", "training_data", fname)

    with open(fname) as f:
        data = f.read()

    lines = data.split("\n")
    header = lines[0].split(",")
    lines = lines[1:]
    length = int(len(lines) / 5)

    # Inputs -> Consumption, Generation, Electricity Price, Battery SOC
    # Outputs -> Grid Power

    consumption = np.zeros((length, len(header) - 2))
    generation = np.zeros((length, len(header) - 2))
    tariff = np.zeros((length, len(header) - 2))
    bat_soc = np.zeros((length, 1))
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
        elif category == "SOC":
            bat_soc[counter, :] = values
        elif category == "GP":
            grid_power[counter, :] = values
            counter += 1

    return (length, consumption, generation, tariff, bat_soc, grid_power)


# Problem: Given current battery SOC, and forecast of demand and generation, predict todays optimal energy allocation decisions.

def prepare_data(data):
    """Separates training data into training, validation and testing sets.

    Args:
        length: Int specifying number of days in training data.
        consumption: 2-d array specifying consumption values over all time-steps.
        generation: 2-d array specifying generation values over all time-steps.
        tariff: 2-d array specifying tariff values over all time-steps.
        bat_soc: 2-d array specifying SOC values over all time-steps.
        grid_power: 2-d array specifying grid_power values over all time-steps.

    Returns:
        train_data: 3-d array containing training sets for each variable.
        val_data: 3-d array containing validation sets for each variable.
        test_data: 3-d array containing testing sets for each variable.
    """
    length, consumption, generation, tariff, bat_soc, grid_power = data

    num_train_samples = int(0.5 * length)
    num_val_samples = int(0.25 * length)

    train_consumption = consumption[:num_train_samples]
    train_generation = generation[:num_train_samples]
    train_tariff = tariff[:num_train_samples]
    train_soc = bat_soc[:num_train_samples]
    train_grid_power = grid_power[:num_train_samples]
    train_data = [train_consumption, train_generation, train_tariff, train_soc,
                  train_grid_power]

    val_consumption = consumption[num_train_samples:num_train_samples+num_val_samples]
    val_generation = generation[num_train_samples:num_train_samples+num_val_samples]
    val_tariff = tariff[num_train_samples:num_train_samples+num_val_samples]
    val_soc = bat_soc[num_train_samples:num_train_samples+num_val_samples]
    val_grid_power = grid_power[num_train_samples:num_train_samples+num_val_samples]
    val_data = [val_consumption, val_generation, val_tariff, val_soc,
                val_grid_power]

    test_consumption = consumption[num_train_samples+num_val_samples:]
    test_generation = generation[num_train_samples+num_val_samples:]
    test_tariff = tariff[num_train_samples+num_val_samples:]
    test_soc = bat_soc[num_train_samples+num_val_samples:]
    test_grid_power = grid_power[num_train_samples+num_val_samples:]
    test_data = [test_consumption, test_generation, test_tariff, test_soc,
                test_grid_power]
    
    return train_data, val_data, test_data


def build_pfa_model():
    """Initialises policy function approximation (PFA) model using keras.

    Returns:
        model: Initialised PFA model.
    """

    con = keras.Input(shape=(48,), name="forecast")
    gen = keras.Input(shape=(48,), name="generation")
    tar = keras.Input(shape=(48,), name="tariff")
    soc = keras.Input(shape=(1,), name="soc")

    features = layers.Concatenate()([con, gen, tar, soc])
    features = layers.Dense(144)(features)
    features = layers.Dropout(0.125)(features)

    g_power = layers.Dense(48, activation="relu", name="grid_power")(features)

    model = keras.Model(inputs=[con, gen, tar, soc],
                        outputs=[g_power])

    model.compile(optimizer="rmsprop",
                loss="mse",
                metrics=["mae"])
    
    return model