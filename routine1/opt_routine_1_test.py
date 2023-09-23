"""Typical usage of opt_routine_1.py.

In this example three different decision horizons are tested for later analysis.

Author:
    Hamish Mackinlay
"""

from opt_routine_1 import *

base_path = os.path.dirname(__file__)

historical_data = ["Customer2(2010-2011).csv", 
                   "Customer2(2011-2012).csv", 
                   "Customer2(2012-2013).csv"]

# historical_data = ["Customer2(2010-2011).csv"]

# Model Configurations -> [Gen. Scaling Factor, Bat. Min., Bat. Max., Decision Horizon]

configurations = [[5, 0, 7, "2-Day Rolling", 2],
                  [5, 0, 7, "7-Day Rolling", 7],
                  [5, 0, 7, "Global"]]

c_exp = -0.10    # $AUD/kWh (static for now...)
x_bmin = -3.5    # Max. battery discharge per timestep   
x_bmax = 3.5     # Max. battery charge per timestep
eta = 0.9        # Battery charge efficiency factor

solutions = []

consumption, generation, tou_tariff, grid_power, bat_charge, bat_soc, date = read_data(historical_data, base_path)

for config in configurations:

    prev_soc = 1
    generation_scaled = generation * config[0]

    if config[3] == "Global":

        model = init_model(generation_scaled.flatten(), consumption.flatten(), tou_tariff.flatten(), x_bmin, x_bmax, config[1], config[2], eta, c_exp, prev_soc)
        model, solution = solve_model(model, "Global")
        grid_power, bat_charge, bat_soc = get_solution_global(model, grid_power, bat_charge, bat_soc)

    else:

        solution = 0
        for day in range(len(consumption)):
            print("Day:", day)
            end = day + config[4]
            model = init_model(generation_scaled[day:end].flatten(), consumption[day:end].flatten(), tou_tariff[day:end].flatten(), x_bmin, x_bmax, config[1], config[2], eta, c_exp, prev_soc)
            solved_instance = solve_model(model, config[3])
            solution += solved_instance[1]
            grid_power, bat_charge, bat_soc, prev_soc = get_solution(model, grid_power, bat_charge, bat_soc, day)

    solutions.append(solution)
    plot_solution(consumption, generation_scaled, tou_tariff, grid_power, bat_charge, bat_soc, 29, 30, config[3], base_path)

# Un-comment the below line if you wish to generate training data to be used by optimisation routine 2.
# build_training_data(consumption, generation_scaled, tou_tariff, bat_soc, grid_power, bat_charge, date, base_path)

with open(os.path.join(base_path, "results", "solutions.csv"), "w") as f:
    writer = csv.writer(f)
    writer.writerow(["Decision Horizon", "PV Size (kW)", "Battery Size (kWh)", "Cost ($AUD)"])
    for i, solution in enumerate(solutions):
        writer.writerow([configurations[i][3], "%.2f" % (1.62*configurations[i][0]),
                         configurations[i][2], solution])