# HEMS Optimization Routine 1
# Author: Hamish Mackinlay

import os
import numpy as np
from pyomo.environ import *
import csv
import matplotlib
import matplotlib.pyplot as plt


"""
Reads historical/forecast data of customer load and generation profiles and initialises dependent variables.
For investigation purposes,a scaling factor can also be applied to generation values.

Returns generation, consumption, ToU tariff, grid power, battery charge, and battery SOC numpy arrays.
"""
def read_data(file_names, generation_scale):

    lines = []
    base_path = os.path.dirname(__file__)
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
    generation = consumption = tou_tariff = grid_power = bat_charge = bat_soc = np.zeros((length, len(header) - 5))

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
        if category == "GG":
            generation[counter] = values
        elif category == "GC":
            consumption[counter] = values
            tou_tariff[counter] = daily_import_cost
            counter += 1
            date.append(line_split[4])

    # Scale generation values to suit currently available products
    generation *= generation_scale

    return generation, consumption, tou_tariff, grid_power, bat_charge, bat_soc


"""
Initialises linear program cost minimisation problem.

Returns model with configured variables.
"""
def init_model(generation, consumption, tou_tariff, x_bmin, x_bmax, e_bmin, e_bmax):

    model = ConcreteModel()

    model.d = RangeSet(len(generation))
    model.h = RangeSet(48)

    # Initialise independent variables

    def init_gen(model, i, j):
        return generation[i-1][j-1]
    model.g = Param(model.d, model.h, within=NonNegativeReals, initialize=init_gen)

    def init_con(model, i, j):
        return consumption[i-1][j-1]
    model.c = Param(model.d, model.h, within=NonNegativeReals, initialize=init_con)

    def init_cost(model, i, j):
        return tou_tariff[i-1][j-1]
    model.c_imp = Param(model.h, within=NonNegativeReals, initialize=init_cost)

    # Initialise dependent variables

    model.x = Var(model.d, model.h, initialize=0, within=Reals)
    model.x_imp = Var(model.d, model.h, within=NonNegativeReals)
    model.x_exp = Var(model.d, model.h, within=NegativeReals)

    model.x_b = Var(model.d, model.h, initialize=0, within=Reals)
    model.x_b_exp = Var(model.d, model.h, initialize=0, within=Reals,\
                        bounds=(x_bmin, 0))
    model.x_b_imp = Var(model.d, model.h, initialize=0, within=Reals,\
                        bounds=(0, x_bmax))
    model.e_b = Var(model.d, model.h, initialize=e_bmin, within=NonNegativeReals,\
                    bounds=(e_bmin, e_bmax))

    return model


"""
Adds constraints to linear program cost minimisation problem and solves considering decision horizon.

Returns solved model and results summary.
"""
def solve_model(model, e_bmin, eta, c_exp, decision_horizon):

    model.limits = ConstraintList()

    if decision_horizon != "daily":
        model.limits.add(model.e_b[1,1] == e_bmin)

    for d in model.d:
        if decision_horizon == "daily":
            model.limits.add(model.e_b[d,1] == e_bmin)
        for h in model.h:
            model.limits.add(model.x[d,h] - model.x_b[d,h] - model.c[d,h] + model.g[d,h] == 0)
            model.limits.add(model.x_exp[d,h] + model.x_imp[d,h] - model.x[d,h] == 0)
            model.limits.add(model.x_b_exp[d,h] + model.x_b_imp[d,h] - model.x_b[d,h] == 0)

            if h != 48 and d != len(model.d):
                if h != 48:
                    model.limits.add(eta*model.x_b_imp[d,h] + (1/eta)*model.x_b_exp[d,h] - \
                                    (model.e_b[d,h+1] - model.e_b[d,h]) == 0)
                else:
                    model.limits.add(eta*model.x_b_imp[d,h] + (1/eta)*model.x_b_exp[d,h] - \
                                    (model.e_b[d+1,1] - model.e_b[d,h]) == 0)
            
    # Optimization model - objective

    def HEMS_obj(model):
        return sum(sum((model.c_imp[h]*model.x_imp[d,h] - c_exp*model.x_exp[d,h]) for h in model.h) for d in model.d)
    model.obj = Objective(rule=HEMS_obj, sense=minimize)

    solver = SolverFactory("gurobi")

    results = solver.solve(model, tee=True)

    return model, results

"""
Get optimal grid import/export, battery charge/discharge, and battery SOC

Returns updated variables
"""
def get_solution(model, grid_power, bat_charge, bat_soc):

    counter = 0
    for d in model.d:
        grid_power[counter] = [value(i) for i in model.x[d,:]]
        bat_charge[counter] = [value(i) for i in model.x_b[d,:]]
        bat_soc[counter] = [value(i) for i in model.e_b[d,:]]
        counter += 1
    
    return grid_power, bat_charge, bat_soc

# Plot results (OPTIONAL)

# Retrieve optimal objective and add to plot

title = "Optimal SOC - Daily Decision Horizon"

start = 0 * 48
end = 7 * 48
x_axis = range(length*48)[start:end]

plt.figure(figsize=(20,10))

plt.plot(x_axis, generation.flatten()[start:end], label="generation")
plt.plot(x_axis, consumption.flatten()[start:end], label="consumption")
plt.fill_between(x_axis,
                 -2, 
                 10,
                 where=(tou_tariff.flatten()[start:end] == peak_tarrif),
                 alpha=0.5, label="peak tariff")

plt.plot(x_axis, grid_power.flatten()[start:end], label="grid power")
plt.plot(x_axis, bat_charge.flatten()[start:end], label="bat charge")
plt.plot(x_axis, bat_soc.flatten()[start:end], label="bat soc")

plt.xlabel("Time (0.5 hours)")
plt.ylabel("Power (kWh)")
plt.title(title)
plt.legend()
plt.savefig(os.path.join(base_path, "results", title+".png"))
plt.close()

# # Send results to .csv file

# fields = header[3:]

# with open("training_data/training_data.csv", "w") as f:

#     writer = csv.writer(f)

#     writer.writerow(fields)
    
#     for i in range(length):
#         writer.writerow(["GC"] + [date[i]] + list(consumption[i]))
#         writer.writerow(["GG"] + [date[i]] + list(generation[i]))
#         writer.writerow(["ToU"] + [date[i]] + list(c_imp))
#         writer.writerow(["GP"] + [date[i]] + list(grid_power[i]))