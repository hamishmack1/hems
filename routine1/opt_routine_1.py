"""HEMS optimisation routine 1 helper functions.

This script provides a collection of helper functions for HEMS optimisation. It
enables parsing of historical/forecast customer data, initialisaiton and solving
of linear program cost minimisaiton problem for different decision horizons,
visualisation of solutions, and the creation of training data.

This script is designed to streamline the optimisation process for managing
home energy consumption and assets efficiently.

Author:
    Hamish Mackinlay
"""

import os
import numpy as np
from pyomo.environ import *
import csv
import matplotlib.pyplot as plt
from memory_profiler import profile


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

    days = 3
    return consumption[:days], generation[:days], tou_tariff[:days], \
            grid_power[:days], bat_charge[:days], bat_soc[:days], date[:days]



def init_model(generation, consumption, tou_tariff, x_bmin, x_bmax, e_bmin,
               e_bmax, eta, c_exp, prev_soc):
    """Initialises linear program cost minimisation problem.

    Args:
        generation: 1-d array of customer generation data.
        consumption: 1-d array of customer consumption data.
        tou_tariff: 1-d array of ToU tariff.
        x_bmin: Float specifying minimum battery charge/discharge.
        x_bmax: Float specifying maximum battery charge/discharge.
        e_bmin: Float specifying minimum battery SOC.
        e_bmax: Float specifying maximum battery SOC.
        eta: Float specifying battery efficiency parameter.
        c_exp: Negative float specifying feed in tariff price.
        prev_soc: Float specifying SOC at end of previous day.

    Returns:
        Initialised concrete model.    
    """

    model = ConcreteModel()

    model.h = RangeSet(len(generation))

    # Initialise independent variables.

    def init_gen(m, i):
        return generation[i-1]
    model.g = Param(model.h, within=NonNegativeReals, initialize=init_gen)

    def init_con(m, i):
        return consumption[i-1]
    model.c = Param(model.h, within=NonNegativeReals, initialize=init_con)

    def init_cost(m, i):
        return tou_tariff[i-1]
    model.c_imp = Param(model.h, within=NonNegativeReals, initialize=init_cost)

    model.c_exp = Param(initialize=c_exp)

    # Initialise dependent variables.

    model.x = Var(model.h, initialize=0, within=Reals)
    model.x_imp = Var(model.h, within=NonNegativeReals)
    model.x_exp = Var(model.h, within=NegativeReals)

    model.x_b = Var(model.h, initialize=0, within=Reals, bounds=(x_bmin, x_bmax))
    model.x_b_exp = Var(model.h, initialize=0, within=Reals,\
                        bounds=(x_bmin, 0))
    model.x_b_imp = Var(model.h, initialize=0, within=Reals,\
                        bounds=(0, x_bmax))
    model.e_b = Var(model.h, within=NonNegativeReals,\
                    bounds=(e_bmin, e_bmax))

    # Initialise constraints.

    def initial_soc_constraint_rule(m):
        return m.e_b[1] == prev_soc
    model.initial_soc_constraint = Constraint(rule=initial_soc_constraint_rule)

    def final_soc_constraint_rule(m):
        return m.e_b[len(m.h)] == e_bmin
    model.final_soc_constraint = Constraint(rule=final_soc_constraint_rule)

    def final_bat_constraint_rule(m):
        return m.x_b[len(m.h)] == 0
    model.final_bat_constraint = Constraint(rule=final_bat_constraint_rule)

    def power_bal_constraint_rule(m, i):
        return m.x[i] - m.x_b[i] - m.c[i] + m.g[i] == 0
    model.power_bal_constraint = Constraint(model.h, rule=power_bal_constraint_rule)

    def grid_constraint_rule(m, i):
        return m.x_exp[i] + m.x_imp[i] - m.x[i] == 0
    model.grid_constraint = Constraint(model.h, rule=grid_constraint_rule)

    def bat_constraint_rule(m, i):
        return m.x_b_exp[i] + m.x_b_imp[i] - m.x_b[i] == 0
    model.bat_constraint = Constraint(model.h, rule=bat_constraint_rule)

    def soc_constraint_rule(m, i):
        if i < len(m.h):
            return eta*m.x_b_imp[i] + (1/eta)*m.x_b_exp[i] - (m.e_b[i+1] - m.e_b[i]) == 0
        else:
            return eta*m.x_b_imp[i] + (1/eta)*m.x_b_exp[i] + m.e_b[i] >= e_bmin
    model.soc_constraint = Constraint(model.h, rule=soc_constraint_rule)

    def hems_obj(m):
        return sum(m.c_imp[d]*m.x_imp[d] - m.c_exp*m.x_exp[d] for d in m.h)
    model.obj = Objective(rule=hems_obj)

    return model


def solve_model(m, dec_hor):
    """Solves model instance considering decision horizon.

    Args:
        m: Concrete model of initialised linear cost minimisation problem.
        dec_hor: String specifying decision horizon.

    Returns:
        m: Solved model.
        solution: Numeric value of optimal cost.
    """

    solver = SolverFactory("gurobi")
    solver.solve(m, tee=True)

    if dec_hor == "Global":
        solution = value(m.obj)
    else:
        solution = value(sum(m.c_imp[i+1]*m.x_imp[i+1] - m.c_exp*m.x_exp[i+1] for i in range(48)))

    return m, solution


def get_solution(model, grid_power, bat_charge, bat_soc, day):
    """Gets optimal dependent variables for rolling decision horizon.

    Args:
        model: Solved concrete model.
        grid_power: 2-d array specifying grid values over all time-steps.
        bat_charge: 2-d array specifying battery values over all time-steps.
        bat_soc: 2-d array specifying SOC values over all time-steps.
        day: Int indicating day index.

    Returns:
        grid_power: 2-d array with updated hourly grid values for particular day.
        bat_charge: 2-d array with updated hourly battery values for particular day.
        bat_soc: 2-d array with updated hourly SOC values for particular day.
        prev_soc: Numeric value specifying SOC at time-step 49 to be utilised by following day as initial SOC.
                    This is only relevant if decision horizon > 1 day.
    """

    grid_power[day,:] = [value(model.x[d]) for d in model.h][:48]
    bat_charge[day,:] = [value(model.x_b[d]) for d in model.h][:48]
    bat_soc[day,:] = [value(model.e_b[d]) for d in model.h][:48]

    prev_soc = 2
    if len(model.h) > 48: prev_soc = value(model.e_b[49])
    
    return grid_power, bat_charge, bat_soc, prev_soc


def get_solution_global(model, grid_power, bat_charge, bat_soc):
    """Gets optimal dependent variables for global decision horizon.

    Args:
        model: Solved concrete model.
        grid_power: 2-d array specifying grid values over all time-steps.
        bat_charge: 2-d array specifying battery values over all time-steps.
        bat_soc: 2-d array specifying SOC values over all time-steps.

    Returns:        
        grid_power: 2-d array with updated hourly grid values for particular day.
        bat_charge: 2-d array with updated hourly battery values for particular day.
        bat_soc: 2-d array with updated hourly SOC values for particular day.        
    """

    grid_power = np.reshape([value(model.x[d]) for d in model.h], (len(grid_power), 48))
    bat_charge = np.reshape([value(model.x_b[d]) for d in model.h], (len(grid_power), 48))
    bat_soc = np.reshape([value(model.e_b[d]) for d in model.h], (len(grid_power), 48))

    return grid_power, bat_charge, bat_soc


def plot_solution(consumption, generation, tou_tariff, grid_power, bat_charge,
                  bat_soc, start, end, dec_horizon, base_path):
    """Plots optimal solution and saves as png file.

    Args:
        data: Arrays for all variables respectively.
        start: Int specifying day to begin plot.
        end: Int specifying day to end plot.
        dec_horizon: String specifying decision horizon.
        base_path: String specifying path where script is executed.
    """

    title = "Optimal SOC - " + dec_horizon + " Decision Horizon"

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


def build_training_data(consumption, generation, tou_tariff, bat_soc,
                        grid_power, bat_charge, date, base_path):
    """Builds training data to be utilised by optimisation routine 2.

    Args:
        data: 2-d Arrays for all variables respectively.
        date: 1-d Array specifying customer data dates.
        base_path: String specifying path where script is executed.
    """

    fields = ["Consumption Category","date","0:30","1:00","1:30","2:00","2:30","3:00","3:30","4:00","4:30","5:00",
              "5:30","6:00","6:30","7:00","7:30","8:00","8:30","9:00","9:30","10:00","10:30","11:00","11:30","12:00",
              "12:30","13:00","13:30","14:00","14:30","15:00","15:30","16:00","16:30","17:00","17:30","18:00","18:30",
              "19:00","19:30","20:00","20:30","21:00","21:30","22:00","22:30","23:00","23:30","0:00"]

    with open(os.path.join(base_path, "..", "training_data", "global.csv"), "w") as f:

        writer = csv.writer(f)

        writer.writerow(fields)
        
        for i in range(len(consumption)):
            writer.writerow(["GC"] + [date[i]] + list(consumption[i]))
            writer.writerow(["GG"] + [date[i]] + list(generation[i]))
            writer.writerow(["ToU"] + [date[i]] + list(tou_tariff[i]))
            writer.writerow(["SOC"] + [date[i]] + list(bat_soc[i]))
            writer.writerow(["GP"] + [date[i]] + list(grid_power[i]))
            # writer.writerow(["BAT"] + [date[i]] + list(bat_charge[i]))
