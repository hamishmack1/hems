# Use this script to determine optimal decision horizon whilst also observing the effect of different solar cell sizes.
# Author: Hamish Mackinlay

from opt_routine_1_rolling import *

base_path = os.path.dirname(__file__)

# historical_data = ["Customer2(2010-2011).csv", 
#                    "Customer2(2011-2012).csv", 
#                    "Customer2(2012-2013).csv"]

historical_data = ["Customer2(2010-2011).csv"]

# Model Configurations -> [Gen. Scaling Factor, Bat. Min., Bat. Max., Decision Horizon]
# model_configurations = [[1, 1, 5, "Daily"],
#                         [1, 1, 5, "2-day Rolling"],
#                         [1, 1, 5, "Overall"], 
#                         [4, 2, 10, "Daily"],
#                         [4, 2, 10, "2-day Rolling"],
#                         [4, 2, 10, "Overall"]]

# model_configurations = [[1, 1, 5, "2-day Rolling"]]

# model_configurations = [[5, 2, 10, "Daily"],
#                         [5, 2, 10, "2-day Rolling"],
#                         [5, 2, 10, "4-day Rolling"],
#                         [5, 2, 10, "Overall"]]

# model_configurations = [[1, 1, 5, "Daily"],
#                         [1, 1, 5, "Overall"],
#                         [4, 2, 10, "Daily"],
#                         [4, 2, 10, "Overall"]]

config = [5, 0, 7, "2-day Rolling"]

c_exp = -0.10    # $AUD/kWh (static for now...)
x_bmin = -3.5     # Max. battery discharge per timestep   
x_bmax = 3.5      # Max. battery charge per timestep
eta = 0.9       # Charge efficiency factor

solutions = []
consumption, generation, tou_tariff, grid_power, bat_charge, bat_soc, date = read_data(historical_data, base_path)

generation_scaled = generation * config[0]

solution = 0
grid_import_total = 0
prev_soc = 2
print(len(consumption))
for i in range(len(consumption)):
    print("Index:", i)

    end = i + 4


    model = init_model(generation_scaled[i:end].flatten(), consumption[i:end].flatten(), tou_tariff[i:end].flatten(), x_bmin, x_bmax, config[1], config[2], eta, c_exp, prev_soc)
    solved_instance = solve_model(model)
    solution += solved_instance[1]
    grid_import_total += solved_instance[2]
    grid_power, bat_charge, bat_soc, prev_soc = get_solution(model, grid_power, bat_charge, bat_soc, i)

solutions.append(solution)
plot_solution(consumption, generation_scaled, tou_tariff, grid_power, bat_charge, bat_soc, 0, 30, config[3], base_path)
print(grid_import_total)

# build_training_data(consumption, generation_scaled, tou_tariff, bat_soc, grid_power, bat_charge, date, base_path)

with open(os.path.join(base_path, "results", "solutions2.csv"), "w") as f:
    writer = csv.writer(f)
    writer.writerow(["Decision Horizon", "PV Size (kW)", "Battery Size (kWh)", "Cost ($AUD)"])
    writer.writerow([config[3], "%.2f" % (1.62*config[0]),
                         config[2], solutions[0]])