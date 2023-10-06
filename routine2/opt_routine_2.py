# HEMS Optimization Routine 2
# Author: Hamish Mackinlay

import os
import numpy as np
from pyomo.environ import *
from tensorflow import keras
from keras import layers
from pandas import *
import matplotlib.pyplot as plt

def read_train_data(fname, base_path):

    fname = os.path.join(base_path, "..", "training_data", fname)

    data = read_csv(fname).drop(["Consumption Category", "date"], axis=1)
    fields = data.columns.to_list()

    timesteps = len(data.columns)
    days = len(data) // 6

    timestep_data = np.zeros((timesteps, days, 6))
    for step in range(timesteps):
        timestep_data[step] = data.iloc[:,step].to_numpy().reshape((days,6))

    return fields, timestep_data


def prepare_data(raw_data):

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

    Returns:
        model: Initialised PFA model for specific timestep.
    """

    con = keras.Input(shape=(1,), name="forecast")
    gen = keras.Input(shape=(1,), name="generation")
    tar = keras.Input(shape=(1,), name="tariff")
    soc = keras.Input(shape=(1,), name="soc")

    inputs = layers.Concatenate()([con, gen, tar, soc])
    features = layers.Dense(4, activation="softmax")(inputs)
    g_imp = layers.Dense(1, activation="relu", name="g_imp")(features)
    g_exp = layers.Dense(1, activation="relu", name="g_exp")(features)

    model = keras.Model(inputs=inputs,
                        outputs=[g_imp, g_exp])
    
    model_path = os.path.join("timestep_models", model_name + ".keras")
    callbacks = [ keras.callbacks.ModelCheckpoint(model_path,
                                                  save_best_only=True)
    ]
    model.compile(optimizer="rmsprop",
                loss="mse",
                metrics=["mae"])
    history = model.fit(train_data[:,:4],
                    [train_data[:,4], train_data[:,5]],
                    epochs=500,
                    validation_data=(val_data[:,:4], [val_data[:,4], val_data[:,5]]),
                    callbacks=callbacks,
                    verbose=1)
    
    return history

def build_timestep_models(fname, base_path):
    
    step_headings, data = read_train_data(fname, base_path)

    for i, step in enumerate(step_headings):
        train_data, val_data, test_data = prepare_data(data[i])
        build_model(train_data, val_data, step)
        model = keras.models.load_model("timestep_models/" + step + ".keras")
        print(f"Test MAE: {model.evaluate(test_data[:,:4], test_data[:,4])[1]:.3f}")