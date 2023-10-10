"""Implementation of opt_routine_2.py

Author:
    Hamish Mackinlay
"""
from opt_routine_2 import *

base_path = os.path.dirname(__file__)

# Uncomment below 2 lines if training models prior to run time execution."
# training_data = "global.csv"
# build_timestep_models(training_data, base_path)

historical_data = ["Customer2(2010-2011).csv",
                    "Customer2(2011-2012).csv",
                    "Customer2(2012-2013).csv"]

timestep_models = load_timestep_models()

input_details = timestep_models[0].get_input_details()
output_details = timestep_models[0].get_output_details()

consumption, generation, tou_tariff, grid_power, bat_charge, bat_soc, date = read_data(historical_data, base_path)

generation *= 4.075

days = range(len(consumption))
timesteps = range(len(timestep_headings))

# Set initial battery SOC
bat_soc[0,0] = 2
eta = 0.9

cost = 0

# @profile
def solve(cost):

    for day in days:

        print("Day: " + str(day))

        for step in timesteps:

            inputs = [[consumption[day,step], generation[day,step], tou_tariff[day,step], bat_soc[day,step]]]
            timestep_models[step].set_tensor(input_details[0]['index'], inputs)
            timestep_models[step].invoke()

            x = timestep_models[step].get_tensor(output_details[0]['index'])
            x_b = generation[day,step] - consumption[day,step] + x

            if step < 47:
                grid_power[day,step], bat_charge[day,step], bat_soc[day,step+1] = get_soc(x, x_b, bat_soc[day,step], eta)
            elif day == 1014:
                grid_power[day,step], bat_charge[day,step] = generation[day,step] - consumption[day,step], 0
            else:
                grid_power[day,step], bat_charge[day,step], bat_soc[day+1,0] = get_soc(x, x_b, bat_soc[day,step], eta)

            if grid_power[day,step] < 0:
                cost += 0.10*grid_power[day,step]
            elif grid_power[day,step] > 0:
                cost += tou_tariff[day,step]*grid_power[day,step]

    print("COST: " + str(cost))

solve(cost)

# plot_solution(consumption, generation, tou_tariff, grid_power, bat_charge,
#                     bat_soc, 100, 107, base_path)