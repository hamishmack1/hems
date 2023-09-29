# hems

The aim of this project is to investigate the feasibility of deploying optimization routines on edge devices for HEMS scheduling applications.

## to-do:

1. Modify optimisation routine 2 to train models for EVERY time-step instead of entire day. This allows for fast real time execution and mitigates error introduced by poor forecasting.
2. Add a control loop/policy to optimisation routine 2.
    1. Satisfy power balance equation.
    2. State of system needs to be carried between days. Feed current battery SOC to next time step.
