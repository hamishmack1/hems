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
def read_data(file_names, base_path):

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

    # return consumption[:30], generation[:30], tou_tariff[:30], grid_power[:30], bat_charge[:30], bat_soc[:30], date[:30]
    days = None
    return consumption[:days], generation[:days], tou_tariff[:days], grid_power[:days], bat_charge[:days], bat_soc[:days], date[:days]


"""
Initialises linear program cost minimisation problem.

Returns model with configured variables.
"""
def init_model(generation, consumption, tou_tariff, x_bmin, x_bmax, e_bmin, e_bmax, eta, c_exp, prev_soc):

    model = ConcreteModel()

    model.d = RangeSet(len(generation))

    # Initialise independent variables

    def init_gen(m, i):
        return generation[i-1]
    model.g = Param(model.d, within=NonNegativeReals, initialize=init_gen)

    def init_con(m, i):
        return consumption[i-1]
    model.c = Param(model.d, within=NonNegativeReals, initialize=init_con)

    def init_cost(m, i):
        return tou_tariff[i-1]
    model.c_imp = Param(model.d, within=NonNegativeReals, initialize=init_cost)

    # Dependent variables

    model.x = Var(model.d, initialize=0, within=Reals)
    model.x_imp = Var(model.d, within=NonNegativeReals)
    model.x_exp = Var(model.d, within=NegativeReals)

    model.x_b = Var(model.d, initialize=0, within=Reals, bounds=(x_bmin, x_bmax))
    model.x_b_exp = Var(model.d, initialize=0, within=Reals,\
                        bounds=(x_bmin, 0))
    model.x_b_imp = Var(model.d, initialize=0, within=Reals,\
                        bounds=(0, x_bmax))
    model.e_b = Var(model.d, initialize=e_bmin, within=NonNegativeReals,\
                    bounds=(e_bmin, e_bmax))

    def initial_soc_constraint_rule(m):
        return m.e_b[1] == prev_soc
    model.initial_soc_constraint = Constraint(rule=initial_soc_constraint_rule)

    def final_soc_constraint_rule(m):
        return m.e_b[len(m.d)] == e_bmin
    model.final_soc_constraint = Constraint(rule=final_soc_constraint_rule)

    def final_bat_constraint_rule(m):
        return m.x_b[len(m.d)] == 0
    model.final_bat_constraint = Constraint(rule=final_bat_constraint_rule)

    def power_bal_constraint_rule(m, i):
        return m.x[i] - m.x_b[i] - m.c[i] + m.g[i] == 0
    model.power_bal_constraint = Constraint(model.d, rule=power_bal_constraint_rule)

    def grid_constraint_rule(m, i):
        return m.x_exp[i] + m.x_imp[i] - m.x[i] == 0
    model.grid_constraint = Constraint(model.d, rule=grid_constraint_rule)

    def bat_constraint_rule(m, i):
        return m.x_b_exp[i] + m.x_b_imp[i] - m.x_b[i] == 0
    model.bat_constraint = Constraint(model.d, rule=bat_constraint_rule)

    def soc_constraint_rule(m, i):
        if i < len(m.d):
            return eta*m.x_b_imp[i] + (1/eta)*m.x_b_exp[i] - (m.e_b[i+1] - m.e_b[i]) == 0
        else:
            return Constraint.Skip
            # return eta*m.x_b_imp[i] + (1/eta)*m.x_b_exp[i] + m.e_b[i] >= e_bmin
    model.soc_constraint = Constraint(model.d, rule=soc_constraint_rule)

    def hems_obj(m):
        return sum(m.c_imp[d]*m.x_imp[d] - c_exp*m.x_exp[d] for d in m.d)
    model.obj = Objective(rule=hems_obj)

    return model


"""
Adds constraints to linear program cost minimisation problem and solves considering decision horizon.

Returns solved model and results summary.
"""
def solve_model(model):
     
    # Optimization model - objective

    solver = SolverFactory("gurobi")
    solver.solve(model)

    solution = 0
    sol = 0

    for i in range(48):
        solution += value(model.c_imp[i+1]*model.x_imp[i+1] - (-0.10)*model.x_exp[i+1])
        sol += value(model.x_imp[i+1])
    return model, solution, sol


"""
Get optimal grid import/export, battery charge/discharge, and battery SOC

Returns updated variables
"""
def get_solution(model, grid_power, bat_charge, bat_soc, index):

    grid_power[index,:] = [value(model.x[d]) for d in model.d][:48]
    bat_charge[index,:] = [value(model.x_b[d]) for d in model.d][:48]
    bat_soc[index,:] = [value(model.e_b[d]) for d in model.d][:48]

    prev_soc = value(model.e_b[48])
    
    return grid_power, bat_charge, bat_soc, prev_soc


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
    plt.ylabel("Power (kW)")
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(base_path, "results", title+".png"))
    plt.close()


"""
Builds training data and exports to csv file.

Returns null.
"""
def build_training_data(consumption, generation, tou_tariff, bat_soc, grid_power, bat_charge, date, base_path):

    fields = ["Consumption Category","date","0:30","1:00","1:30","2:00","2:30","3:00","3:30","4:00","4:30","5:00",
              "5:30","6:00","6:30","7:00","7:30","8:00","8:30","9:00","9:30","10:00","10:30","11:00","11:30","12:00",
              "12:30","13:00","13:30","14:00","14:30","15:00","15:30","16:00","16:30","17:00","17:30","18:00","18:30",
              "19:00","19:30","20:00","20:30","21:00","21:30","22:00","22:30","23:00","23:30","0:00"]

    with open(os.path.join(base_path, "..", "training_data", "rolling.csv"), "w") as f:

        writer = csv.writer(f)

        writer.writerow(fields)
        
        for i in range(len(consumption)):
            writer.writerow(["GC"] + [date[i]] + list(consumption[i]))
            writer.writerow(["GG"] + [date[i]] + list(generation[i]))
            writer.writerow(["GP"] + [date[i]] + list(grid_power[i]))
            writer.writerow(["BAT"] + [date[i]] + list(bat_charge[i]))
            # writer.writerow(["ToU"] + [date[i]] + [tou_tariff[i]])
            writer.writerow(["SOC"] + [date[i]] + list(bat_soc[i]))
