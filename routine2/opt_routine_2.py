"""HEMS optimisation routine 2 helper functions.

This script provides a collection of helper functions for PFA (Policy Function
Approximation) based HEMS optimisation. It enables reading and preparing of
training data, building of ANN PFA models, next pre-decision state checks, as
well as parsing historical/forecast customer data and visualisations of
solutions for testing.

This script is designed to create a fast and low memory optimisation process for
managing home energy consumption and assets efficiently.

Author:
    Hamish Mackinlay
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import lite
from tensorflow import keras
from pandas import read_csv
import matplotlib.pyplot as plt

# from memory_profiler import profile


timestep_headings = ["0:30","1:00","1:30","2:00","2:30","3:00","3:30","4:00",
                     "4:30","5:00","5:30","6:00","6:30","7:00","7:30","8:00",
                     "8:30","9:00","9:30","10:00","10:30","11:00","11:30",
                     "12:00","12:30","13:00","13:30","14:00","14:30","15:00",
                     "15:30","16:00","16:30","17:00","17:30","18:00","18:30",
                     "19:00","19:30","20:00","20:30","21:00","21:30","22:00",
                     "22:30","23:00","23:30","0:00"]


def read_train_data(fname, base_path):
    """Parses training data and organises variables according to timestep.

    Args:
        fname: Training data .csv file name
        base_path: Path where script is executed

    Returns:
        timestep_data: 3-day numpy array of timestep data.    
    """
    fname = os.path.join(base_path, "..", "training_data", fname)

    data = read_csv(fname).drop(["Consumption Category", "date"], axis=1)

    timesteps = len(data.columns)
    days = len(data) // 5

    timestep_data = np.zeros((timesteps, days, 5))
    for step in range(timesteps):
        timestep_data[step] = data.iloc[:,step].to_numpy().reshape((days,5))

    return timestep_data


def prepare_data(raw_data):
    """Seperates raw data into training, validation, and test sets.

    Args:
        raw_data: 2-d numpy array of specific timestep data.

    Return:
        train_data: 2-d numpy array specifying first half of raw data.
        val_data: 2-d numpy array specifying third quarter of raw data.
        test_data: 2-d numpy array specifying remaining quarter of raw data.
    """
    length = len(raw_data)
    num_train_samples = int(0.5 * length)
    num_val_samples = int(0.25 * length)

    train_data = raw_data[:num_train_samples]
    val_data = raw_data[num_train_samples:num_train_samples+num_val_samples]
    test_data = raw_data[num_train_samples+num_val_samples:]

    return train_data, val_data, test_data


def build_model(train_data, val_data, model_name):
    """Initialises policy function approximation (PFA) model for a specific
            timestep.

    Args:
        train_data: 2-d numpy array specifying first half of raw data.
        val_data: 2-d numpy array specifying third quarter of raw data.
        model_name: String specifying timestep.

    Returns:
        model: Initialised PFA model for specific timestep.
    """
    con = keras.Input(shape=(1,), name="forecast")
    gen = keras.Input(shape=(1,), name="generation")
    tar = keras.Input(shape=(1,), name="tariff")
    soc = keras.Input(shape=(1,), name="soc")

    inputs = keras.layers.Concatenate()([con, gen, tar, soc])

    features = keras.layers.Dense(256, activation="relu")(inputs)
    features = keras.layers.Dense(256, activation="relu")(features)
    grid_power = keras.layers.Dense(1, activation="linear", name="grid_power")(features)

    model = keras.Model(inputs=inputs,
                        outputs=[grid_power])
    
    model_path = os.path.join("timestep_models", model_name + ".h5")
    callbacks = [ keras.callbacks.ModelCheckpoint(model_path,
                                                  monitor="val_loss",
                                                  save_best_only=True)
    ]
    model.compile(optimizer="RMSprop",
                loss=["mse"],
                metrics=["mae"])

    history = model.fit(train_data[:,:4],
                train_data[:,4],
                epochs=100,
                validation_data=(val_data[:,:4], val_data[:,4]),
                callbacks=callbacks,
                verbose=1)
    
    return history


def build_timestep_models(fname, base_path):
    """Builds and saves policy function approximation (PFA) model for all
            timesteps.

    Args:
        fname: Training data .csv file name
        base_path: Path where script is executed
    """
    data = read_train_data(fname, base_path)

    for i, step in enumerate(timestep_headings):
        train_data, val_data, test_data = prepare_data(data[i])
        build_model(train_data, val_data, step)
        model = keras.models.load_model("timestep_models/" + step + ".h5")
        os.remove("timestep_models/" + step + ".h5")

        # Un-comment below if analsing model accuracy.
        # print(f"Test MAE: {model.evaluate(test_data[:,:4], test_data[:,4])[1]:.3f}")

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        with open("timestep_models/" + step + ".tflite", "wb") as f:
            f.write(tflite_model)



def load_timestep_models():
    """Load timestep PFA models from memory and allocate tensors.
    
    Returns:
        models: Array containing all timestep models.
    """
    models = []
    for step in timestep_headings:
        interpreter = tf.lite.Interpreter("timestep_models/" + step + ".tflite")
        interpreter.allocate_tensors()
        models.append(interpreter)
        
    return models


def read_data(file_names, base_path):
    """Reads historical/forecast data and initialises dependent variables.

    Args:
        file_names: Array of data files.
        base_path: Path where script is executed.

    Returns:
        Arrays for independent and dependent variables respectively.
    """
    lines = []
    for file in file_names:
        fname = os.path.join(base_path, "..", "raw_data", file)

        with open(fname) as f:
            data = f.read()

        line_data = data.split("\n")
        header = line_data[0].split(",")
        line_data = line_data[1:]

        lines += line_data

    # Parse data

    length = int(len(lines) / 2)
    consumption = np.zeros((length, len(header) - 5), np.float32)
    generation = np.zeros((length, len(header) - 5), np.float32)
    tou_tariff = np.zeros((length, len(header) - 5), np.float32)
    grid_power = np.zeros((length, len(header) - 5), np.float32)
    bat_charge = np.zeros((length, len(header) - 5), np.float32)
    bat_soc = np.zeros((length, len(header) - 5), np.float32)

    # ToU Pricing
    
    peak_tarrif = 0.539         # $AUD/kWh (3pm - 9pm)
    offpeak_tarrif = 0.1495     # $AUD/kWh (All other times)
    daily_import_cost = [offpeak_tarrif] * 29 + [peak_tarrif] * 13 + [offpeak_tarrif] * 6

    date = []

    counter = 0
    for line in lines:
        line_split = line.split(",")
        category = line_split[3]
        values = [float(x) for x in line_split[5:]]
        if category == "GC":
            consumption[counter] = values
            tou_tariff[counter] = daily_import_cost
            date.append(line_split[4])
        elif category == "GG":
            generation[counter] = values
            counter += 1

    days = None
    return consumption[:days], generation[:days], tou_tariff[:days], \
            grid_power[:days], bat_charge[:days], bat_soc[:days], date[:days]


def get_soc(grid_power, bat_charge, soc, eta):
    """Calculate next pre-decision SOC state and adjust grid power and battery
        charge values accordinly considering SOC constraints.

    Args:
        grid_power: Float32 pre-adjustment grid power value.
        bat_charge: Float32 pre-adjustment battery charge value.
        soc: Float32 value specifying current SOC.

    Returns:
        grid_power: Float32 adjusted grid power value.
        bat_charge: Float32 adjusted battery charge value.
        next_soc: Float32 next pre-decision SOC value.
    """

    if bat_charge < 0:

        next_soc = soc + (1/eta)*bat_charge
        if next_soc < 2:
            diff = 2 - next_soc
            grid_power += diff
            bat_charge += diff
            next_soc = 2

    elif bat_charge > 0:

        next_soc = soc + eta*bat_charge
        if next_soc > 10:
            diff = next_soc - 10
            grid_power -= diff
            bat_charge -= diff
            next_soc = 10

    return grid_power, bat_charge, next_soc



def plot_solution(consumption, generation, tou_tariff, grid_power, bat_charge,
                  bat_soc, start, end, base_path):
    """Plots optimal solution and saves as png file.

    Args:
        data: Arrays for all variables respectively.
        start: Int specifying day to begin plot.
        end: Int specifying day to end plot.
        dec_horizon: String specifying decision horizon.
        base_path: String specifying path where script is executed.
    """

    title = "Optimal SOC - PFA Algorithm"

    x_axis = range(len(consumption)*48)[start*48:end*48]
    consumption_sliced = consumption.flatten()[start*48:end*48]
    generation_sliced = generation.flatten()[start*48:end*48]
    tou_tariff_sliced = tou_tariff.flatten()[start*48:end*48]
    grid_power_sliced = grid_power.flatten()[start*48:end*48]
    bat_charge_sliced = bat_charge.flatten()[start*48:end*48]
    bat_soc_sliced = bat_soc.flatten()[start*48:end*48]

    fig, ax1 = plt.subplots()
    fig.set_size_inches(10,5)

    ax1.set_xlabel("Time (0.5hrs)")
    ax1.set_ylabel("Power (kW)")
    ax1.plot(x_axis, consumption_sliced, label="consumption", linestyle='dashed')
    ax1.plot(x_axis, generation_sliced, label="generation", linestyle='dashed')
    ax1.plot(x_axis, grid_power_sliced, label="grid power")
    ax1.plot(x_axis, bat_charge_sliced, label="bat charge")

    ax1.set_title(title)

    ax2 = ax1.twinx()

    color = "tab:blue"
    ax2.set_ylabel("Battery SOC (kWh)", color=color)
    ax2.plot(x_axis, bat_soc_sliced, label="bat soc")
    ax2.fill_between(x_axis, 0, 10, where=(tou_tariff_sliced == max(tou_tariff_sliced)),
                     alpha=0.25, label="peak tariff", color="tab:grey")
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    fig.legend(loc='upper left', fontsize="8")

    fig.savefig(os.path.join(base_path, "results", title+".png"))