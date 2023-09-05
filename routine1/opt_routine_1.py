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
def read_data(file_names, generation_scale, base_path):

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
    consumption = np.zeros((length, len(header) - 5))
    generation = np.zeros((length, len(header) - 5))
    tou_tariff = np.zeros((length, len(header) - 5))
    grid_power = np.zeros((length, len(header) - 5))
    bat_charge = np.zeros((length, len(header) - 5))
    bat_soc = np.zeros((length, len(header) - 5))

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

 
    # Scale generation values to suit currently available products
    generation *= generation_scale

    return consumption, generation, tou_tariff, grid_power, bat_charge, bat_soc, date


"""
Initialises linear program cost minimisation problem.

Returns model with configured variables.
"""
def init_model(consumption, generation, tou_tariff, x_bmin, x_bmax, e_bmin, e_bmax):

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
    model.c_imp = Param(model.d, model.h, within=NonNegativeReals, initialize=init_cost)

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

    def initial_soc_constraint_rule(m):
        # if decision_horizon == "Daily":
        #     return m.e_b[i,1] == e_bmin
        # else:
        #     print("HELLO")
            return m.e_b[1,1] == e_bmin
    model.initial_soc_constraint = Constraint(rule=initial_soc_constraint_rule)

    def power_bal_constraint_rule(m, i, j):
        return m.x[i,j] - m.x_b[i,j] - m.c[i,j] + m.g[i,j] == 0
    model.power_bal_constraint = Constraint(model.d, model.h, rule=power_bal_constraint_rule)

    def grid_constraint_rule(m, i, j):
        return m.x_exp[i,j] + m.x_imp[i,j] - m.x[i,j] == 0
    model.grid_constraint = Constraint(model.d, model.h, rule=grid_constraint_rule)

    def bat_constraint_rule(m, i, j):
        return m.x_b_exp[i,j] + m.x_b_imp[i,j] - m.x_b[i,j] == 0
    model.bat_constraint = Constraint(model.d, model.h, rule=bat_constraint_rule)

    def soc_constraint_rule(m, i, j):
        if j != 48:
            return eta*m.x_b_imp[i,j] + (1/eta)*m.x_b_exp[i,j] - (m.e_b[i,j+1] - m.e_b[i,j]) == 0
        elif j == 48 and i != len(m.d) and decision_horizon != "Daily":
            return eta*m.x_b_imp[i,j] + (1/eta)*m.x_b_exp[i,j] - (m.e_b[i+1,1] - m.e_b[i,j]) == 0
        else:
            return eta*m.x_b_imp[i,j] + (1/eta)*m.x_b_exp[i,j] + m.e_b[i,j] >= e_bmin
    model.soc_constraint = Constraint(model.d, model.h, rule=soc_constraint_rule)
                
    # Optimization model - objective

    # Minimise for 96 timesteps, then take first 48 values.
    # Introduce cost variable per horizon scenario
    # Create an objective/model for each day. Sum independently at end.

    if decision_horizon != "Daily":
        def HEMS_obj(model):
            return sum(model.c_imp[d,h]*model.x_imp[d,h] + c_exp*model.x_exp[d,h] for h in model.h for d in model.d)
        model.obj = Objective(rule=HEMS_obj, sense=minimize)
        solver = SolverFactory("gurobi")
        results = solver.solve(model, tee=True)
        solution = value(model.obj)
    else:
        solution = 0
        model.obj = Objective(expr=sum((model.c_imp[1,h]*model.x_imp[1,h] - c_exp*model.x_exp[1,h]) for h in model.h), sense=minimize)
        solver = SolverFactory("gurobi_persistent")
        solver.set_instance(model)
        for j in model.d:
            print(j)
            model.obj = sum((model.c_imp[j,h]*model.x_imp[j,h] - c_exp*model.x_exp[j,h]) for h in model.h)
            solver.set_objective(model.obj)
            solver.solve(model, save_results=False)
            solution += value(model.obj)

    return model, solution


"""
Get optimal grid import/export, battery charge/discharge, and battery SOC

Returns updated variables
"""
def get_solution(model, grid_power, bat_charge, bat_soc):

    for i, d in enumerate(model.d):
        grid_power[i] = [value(j) for j in model.x[d,:]]
        bat_charge[i] = [value(j) for j in model.x_b[d,:]]
        bat_soc[i] = [value(j) for j in model.e_b[d,:]]
    
    return grid_power, bat_charge, bat_soc


"""
Plots optimal solution and saves as png.

Returns null.
"""
def plot_solution(consumption, generation, tou_tariff, grid_power, bat_charge, bat_soc, start, end, dec_horizon, base_path):

    title = "Optimal SOC - " + dec_horizon + " Decision Horizon"

    x_axis = range(len(consumption)*48)[start*48:end*48]
    consumption_sliced = consumption.flatten()[start*48:end*48]
    generation_sliced = generation.flatten()[start*48:end*48]
    tou_tariff_sliced = tou_tariff.flatten()[start*48:end*48]
    grid_power_sliced = grid_power.flatten()[start*48:end*48]
    bat_charge_sliced = bat_charge.flatten()[start*48:end*48]
    bat_soc_sliced = bat_soc.flatten()[start*48:end*48]

    plt.figure(figsize=(20,10))

    plt.plot(x_axis, consumption_sliced, label="consumption")
    plt.plot(x_axis, generation_sliced, label="generation")
    plt.fill_between(x_axis, -2, 10, where=(tou_tariff_sliced == max(tou_tariff_sliced)),
                     alpha=0.5, label="peak tariff")
    plt.plot(x_axis, grid_power_sliced, label="grid power")
    plt.plot(x_axis, bat_charge_sliced, label="bat charge")
    plt.plot(x_axis, bat_soc_sliced, label="bat soc")

    plt.xlabel("Time (0.5 hours)")
    plt.ylabel("Power (kWh)")
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(base_path, "results", title+".png"))
    plt.close()


"""
Builds training data and exports to csv file.

Returns null.
"""
def build_training_data(consumption, generation, tou_tariff, bat_soc, date, base_path):

    fields = ["Consumption Category","date","0:30","1:00","1:30","2:00","2:30","3:00","3:30","4:00","4:30","5:00",
              "5:30","6:00","6:30","7:00","7:30","8:00","8:30","9:00","9:30","10:00","10:30","11:00","11:30","12:00",
              "12:30","13:00","13:30","14:00","14:30","15:00","15:30","16:00","16:30","17:00","17:30","18:00","18:30",
              "19:00","19:30","20:00","20:30","21:00","21:30","22:00","22:30","23:00","23:30","0:00"]

    with open(os.path.join(base_path, "..", "training_data", "training_data.csv"), "w") as f:

        writer = csv.writer(f)

        writer.writerow(fields)
        
        for i in range(len(consumption)):
            writer.writerow(["GC"] + [date[i]] + [consumption[i]])
            writer.writerow(["GG"] + [date[i]] + [generation[i]])
            writer.writerow(["ToU"] + [date[i]] + [tou_tariff[i]])
            writer.writerow(["SOC"] + [date[i]] + [bat_soc[i]])