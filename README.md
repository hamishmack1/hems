# hems

The aim of this project is to investigate the feasibility of deploying optimization routines on edge devices for HEMS scheduling applications.

## to-do:

1. Scale PV generation historical data to more typical values. 2x, 3x, 4x, etc. Do this while parsing.
2. Using optimisation routine 1, test quality of output for different decision horizons. Gather cost data and visualisations. Determine most suitable decsision horizon.
    1. All customer data (best case scenerio/benchmark)
    2. Single day
    3. Two day (rolling horizon)
    4. Four day (rolling horizon)
3. For all visualisations include battery flow and ToU tariffs.
3. Add battery SOC to training data.
4. Add a control loop to optimisation routine 2.
    1. Satisfy power balance equation.
    2. State of system needs to be carried between days. Feed current battery SOC to next time step.
