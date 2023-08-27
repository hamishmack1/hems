# HEMS Optimization Routine 1
# Author: Hamish Mackinlay

import os
import numpy as np
from pyomo.environ import *
import csv

# Read Historical Data
lines = []

file_names = ["Customer2(2010-2011).csv", "Customer2(2011-2012).csv", "Customer2(2012-2013).csv"]

for file in file_names:
    fname = os.path.join("raw_data", file)

    with open(fname) as f:
        data = f.read()

    line_data = data.split("\n")
    header = line_data[0].split(",")
    line_data = line_data[1:]

    lines += line_data

# Parse historical data. Separate into generation and consumption.

length = int(len(lines) / 2)
generation = np.zeros((length, len(header) - 5))
consumption = np.zeros((length, len(header) - 5))
date = []

gen_count = con_count = 0
for i, line in enumerate(lines):
    line_split = line.split(",")
    values = [float(x) for x in line_split[5:]]
    if i % 2:
        generation[gen_count] = values
        gen_count += 1
    else:
        consumption[con_count] = values
        con_count += 1
        date.append(line_split[4])

# Linear program cost minimization variables

# ToU Pricing
peak_tarrif = 0.539         # $AUD/kWh (3pm - 9pm)
offpeak_tarrif = 0.1495     # $AUD/kWh (All other times)
c_imp = [offpeak_tarrif] * 29 + [peak_tarrif] * 13 + [offpeak_tarrif] * 6
c_exp = 0.1                 # $AUD/kWh (static for now...)

# Battery
x_bmin = -2
x_bmax = 2
e_bmin = 2
e_bmax = 10
eta = 0.9

# Optimization model

model = ConcreteModel()

model.d = RangeSet(length)
model.h = RangeSet(48)

def init_gen(model, i, j):
    return generation[i-1][j-1]
model.g = Param(model.d, model.h, within=NonNegativeReals, initialize=init_gen)

def init_con(model, i, j):
    return consumption[i-1][j-1]
model.c = Param(model.d, model.h, within=NonNegativeReals, initialize=init_con)

def init_cost(model, i):
    return c_imp[i-1]
model.c_imp = Param(model.h, within=NonNegativeReals, initialize=init_cost)


# Optimization model - variables

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

# Optimization model - constraints

model.limits = ConstraintList()

for d in model.d:
    model.limits.add(model.e_b[d,1] == e_bmin)
    for h in model.h:

        model.limits.add(model.x[d,h] - model.x_b[d,h] - model.c[d,h] + model.g[d,h] == 0)

        model.limits.add(model.x_exp[d,h] + model.x_imp[d,h] - model.x[d,h] == 0)

        model.limits.add(model.x_b_exp[d,h] + model.x_b_imp[d,h] - model.x_b[d,h] == 0)

        if h != 48:
            model.limits.add(eta*model.x_b_imp[d,h] + (1/eta)*model.x_b_exp[d,h] - \
                             (model.e_b[d,h+1] - model.e_b[d,h]) == 0)
        else:
            model.limits.add(eta*model.x_b_imp[d,h] + (1/eta)*model.x_b_exp[d,h] + \
                                model.e_b[d,h] >= e_bmin)
            

# Optimization model - objective

def HEMS_obj(model):
    return sum(sum((model.c_imp[h]*model.x_imp[d,h] - c_exp*model.x_exp[d,h]) for h in model.h) for d in model.d)
model.obj = Objective(rule=HEMS_obj, sense=minimize)

solver = SolverFactory("gurobi")

results = solver.solve(model, tee=True)

# Retrieve results

grid_power = []

for d in model.d:
    grid_power.append(list(value(i) for i in model.x[d,:]))

# Send results to .csv file

fields = header[3:]

with open("training_data.csv", "w") as f:

    writer = csv.writer(f)

    writer.writerow(fields)
    
    for i in range(length):
        writer.writerow(["GC"] + [date[i]] + list(consumption[i]))
        writer.writerow(["GG"] + [date[i]] + list(generation[i]))
        writer.writerow(["ToU"] + [date[i]] + list(c_imp))
        writer.writerow(["GP"] + [date[i]] + list(grid_power[i]))