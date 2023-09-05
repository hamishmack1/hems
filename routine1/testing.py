# Use this script to determine optimal decision horizon whilst also observing the effect of different solar cell sizes.
# Author: Hamish Mackinlay

from opt_routine_1 import *

base_path = os.path.dirname(__file__)

# historical_data = ["Customer2(2010-2011).csv", 
#                    "Customer2(2011-2012).csv", 
#                    "Customer2(2012-2013).csv"]

historical_data = ["Customer2(2010-2011).csv"]

# Model Configurations -> [Gen. Scaling Factor, Bat. Min., Bat. Max., Decision Horizon]
# model_configurations = [[1, 1, 5, "Daily"],
#                         [1, 1, 5, "2-day Rolling"],
#                         [1, 1, 5, "4-day Rolling"],
#                         [1, 1, 5, "Overall"], 
#                         [4, 2, 10, "Daily"],
#                         [4, 2, 10, "2-day Rolling"],
#                         [4, 2, 10, "4-day Rolling"],
#                         [4, 2, 10, "Overall"]]]

# model_configurations = [[1, 1, 5, "Daily"],
#                         [1, 1, 5, "2-day Rolling"]]

# model_configurations = [[1, 1, 5, "Daily"],
#                         [1, 1, 5, "Overall"]]

model_configurations = [[1, 1, 5, "Overall"]]

c_exp = 0.1     # $AUD/kWh (static for now...)
x_bmin = -2     # Max. battery discharge per timestep   
x_bmax = 2      # Max. battery charge per timestep
eta = 0.9       # Charge efficiency factor

solutions = []

for config in model_configurations:

    consumption, generation, tou_tariff, grid_power, bat_charge, bat_soc, date = read_data(historical_data, config[0],
                                                                                           base_path)

    model = init_model(consumption, generation, tou_tariff, x_bmin, x_bmax, config[1], config[2])

    model, solution = solve_model(model, config[1], eta, c_exp, config[3])
    solutions.append(solution)

    grid_power, bat_charge, bat_soc = get_solution(model, grid_power, bat_charge, bat_soc)

    plot_solution(consumption, generation, tou_tariff, grid_power, bat_charge, bat_soc, 0, 5, config[3], base_path)

with open(os.path.join(base_path, "results", "solutions.csv"), "w") as f:
    writer = csv.writer(f)
    writer.writerow(["Decision Horizon", "PV Size (kW)", "Battery Size (kWh)", "Cost ($AUD)"])
    for i, solution in enumerate(solutions):
        writer.writerow([model_configurations[i][3], 1.62*model_configurations[i][0],
                         model_configurations[i][2], solution])