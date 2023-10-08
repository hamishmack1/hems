from opt_routine_1 import *
# from memory_profiler import profile

base_path = os.path.dirname(__file__)

historical_data = ["Customer2(2010-2011).csv",
                "Customer2(2011-2012).csv",
                "Customer2(2012-2013).csv"]

consumption, generation, tou_tariff, grid_power, bat_charge, bat_soc, date = read_data(historical_data, base_path)

consumption = consumption.flatten()
generation = generation.flatten()
tou_tariff = tou_tariff.flatten()

@profile
def init_and_solve():
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

    model.c_exp = Param(initialize=0.10)

    model.x_imp = Var(model.h, within=Reals)
    model.y_imp = Var(model.h, within=Binary)
    model.x_exp = Var(model.h, within=Reals)
    model.y_exp = Var(model.h, within=Binary)

    model.x_b = Var(model.h, within=Reals, bounds=(-2,2))
    model.e_b = Var(model.h, within=NonNegativeReals, bounds=(2,10))

    def power_bal_constraint_rule(m, i):
        return (m.x_imp[i]*m.y_imp[i] - m.x_exp[i]*m.y_exp[i]) + m.x_b[i] - m.c[i] + m.g[i] == 0
    model.power_bal_constraint = Constraint(model.h, rule=power_bal_constraint_rule)

    def bin_constraint_rule(m, i):
        return m.y_imp[i] + m.y_exp[i] <= 1
    model.bin_constraint = Constraint(model.h, rule=bin_constraint_rule)

    def soc_constraint_rule(m, i):
        if i < len(m.h):
            return 0.9*m.x_b[i] - (m.e_b[i+1] - m.e_b[i]) == 0
        else:
            return 0.9*m.x_b[i] + m.e_b[i] >= 2
    model.soc_constraint = Constraint(model.h, rule=soc_constraint_rule)

    def hems_obj(m):
        return sum(m.c_imp[d]*m.x_imp[d]*m.y_imp[d] - m.c_exp*m.x_exp[d]*m.y_exp[d] for d in m.h)
    model.obj = Objective(rule=hems_obj)

    solver = SolverFactory("gurobi")
    solver.solve(model, tee=True)

    solution = value(sum(model.c_imp[i+1]*model.x_imp[i+1] - model.c_exp*model.x_exp[i+1] for i in range(48)))
    
init_and_solve()