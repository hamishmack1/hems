"""Typical usage of opt_routine_2.py

Author:
    Hamish Mackinlay
"""

from opt_routine_2 import *

base_path = os.path.dirname(__file__)

training_data = "global.csv"

historical_data = ["Customer2(2010-2011).csv",
                    "Customer2(2011-2012).csv",
                    "Customer2(2012-2013).csv"]

# build_timestep_models(training_data, base_path)

timestep_models = load_timestep_models()

consumption, generation, tou_tariff, grid_power, bat_charge, bat_soc, date = read_data(historical_data, base_path)

generation *= 4.075

days = range(len(consumption))
timesteps = range(len(timestep_headings))

# Set initial battery SOC
bat_soc[0,0] = 2

for day in days:
    print("Day: " + day)
    for step in timesteps:

        inputs = [[consumption[day,step], generation[day,step], tou_tariff[day,step], bat_soc[day,step]]]
        grid_power[day,step] = timestep_models[step].predict(inputs)[0]
        bat_charge[day,step] = generation[day,step] - consumption[day,step] + grid_power[day,step]
        
        if step < 47:
            if bat_charge[day,step] <= 0:
                bat_soc[day,step+1] = bat_soc[day,step] + (1/0.9)*bat_charge[day,step]

                if bat_soc[day,step+1] < 2:
                    difference = 2 - bat_soc[day,step+1]
                    bat_charge[day,step] += difference
                    grid_power[day,step] += difference
                    bat_soc[day,step+1] = 2

            else:
                bat_soc[day,step+1] = bat_soc[day,step] + 0.9*bat_charge[day,step]

                if bat_soc[day,step+1] > 10:
                    difference = bat_soc[day,step+1] - 10
                    bat_charge[day,step] -= difference
                    grid_power[day,step] -= difference
                    bat_soc[day,step+1] = 10
        else:
            if bat_charge[day,step] <= 0:
                bat_soc[day+1,0] = bat_soc[day,step] + (1/0.9)*bat_charge[day,step]

                if bat_soc[day+1,0] < 2:
                    difference = 2 - bat_soc[day,step+1]
                    bat_charge[day,step] += difference
                    grid_power[day,step] += difference
                    bat_soc[day+1,0] = 2

            else:
                bat_soc[day+1,0] = bat_soc[day,step] + 0.9*bat_charge[day,step]

                if bat_soc[day+1,0] > 10:
                    difference = bat_soc[day,step+1] - 10
                    bat_charge[day,step] -= difference
                    grid_power[day,step] -= difference
                    bat_soc[day+1,0] = 10

    if day == 108:
        break

plot_solution(consumption, generation, tou_tariff, grid_power, bat_charge,
                  bat_soc, 100, 107, base_path)























# step_headings, data = read_train_data("global.csv", base_path)

# train_data, val_data, test_data = prepare_data(data[18])
# history = build_model(train_data, val_data, step_headings[18])
# model = keras.models.load_model("timestep_models/" + step_headings[18] + ".keras")

# stats = model.evaluate(test_data[:,:4], test_data[:,4])
# print(f"Test MAE: {stats[1]:.3f}")

# grid_power_preds = model.predict(test_data[:,:4])

# mae_history = history.history["mae"]
# val_mae_history = history.history["val_mae"]

# plt.plot(range(1, len(mae_history) + 1), mae_history, label="training")
# plt.plot(range(1, len(val_mae_history) + 1), val_mae_history, label="validation")
# plt.xlabel("Epochs")
# plt.ylabel("MAE (kW)")
# plt.title("9:30 Timestep")
# plt.legend()
# plt.show()

# start = 0
# end = None

# grid_power_preds = grid_power_preds.flatten()
# for i in range(len(grid_power_preds)):
#     if abs(grid_power_preds[i]) < 0.002:
#         grid_power_preds[i] = 0

# gi = grid_power_preds[start:end] 
# gi_real = test_data[:,4][start:end]

# plt.plot(range(len(gi)), gi, label="Prediction")
# plt.plot(range(len(gi)), gi_real, label="Actual")
# plt.xlabel("Time")
# plt.ylabel("Power (kWh)")
# plt.legend()
# plt.show()

# function to add value labels
# def addlabels(x,y):
#     for i in range(len(x)):
#         plt.text(i, y[i], round(y[i], 3), ha = 'center')

# fig = plt.figure(figsize = (10, 5))
 
# # creating the bar plot
# plt.bar(units_str, evals, width=0.5)
# addlabels(units_str, evals)
 
# plt.xlabel("Units")
# plt.ylabel("Test MAE (kW)")
# plt.show()