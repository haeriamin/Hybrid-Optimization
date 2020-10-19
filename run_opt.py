# Copyright (C) 2020 Amin Haeri [ahaeri92@gmail.com]

# Update rate [Hz] Fixed at 60
# Parameters:
# 1. Internal friction coefficient
# 2. External friction coefficient
# 3. Grain radius [m]
# 4. Surcharge factor
# 5. Iteration number

import pickle

from hyb_obj_func import Excavation
from constr_nm import constrNM, printDict
from plot_opt import plot

# Load/Save last best solution? True/False
load = False  
save = True

# Simulation time
sim_times = [5, 20] # e.g. it can be [20]

# Lower bound
LB = [0.3, 0.1, 0.005, 0.01, 1.0]

# Upper bound
UB = [0.9, 0.7, 0.012, 1.00, 20.0]

for sim_time in sim_times:
    # Initial solution
    if load:
        with open('last_solution.pkl', 'r') as f:
            x0 = list(pickle.load(f))
            print ' Last best solution:', x0[0], ',', x0[1], ',', x0[2], ',', x0[3], ',', x0[4]
    else:
        x0 = [0.7, 0.364, 0.012, 1.00, 10.0]

        # 5cm case:
        # x0 = [6.75401407e-01, 3.35455415e-01, 1.01656163e-02, 2.41966281e-01, 1.96258919e+01]  # 21.4% in 5 sec
        # x0 = [6.78161282e-01, 3.35453257e-01, 1.11135779e-02, 2.09709838e-01, 1.96984516e+01]  # 12.6% in 10 sec 
        # x0 = [6.83741285e-01, 3.35455728e-01, 1.10511245e-02, 2.11047748e-01, 1.98022255e+01]  # 8.2% in 20 sec (34 sec runtime and 1.2% mpe)

        # 2cm case took 15 min (starting with best solution):
        # x0 = [7.04127367e-01, 3.35380587e-01, 1.09857695e-02, 3.47156506e-01, 1.98045868e+01]  # 6.2% in 20 sec (24 sec runtime and ?% mpe)

    # Optimizer
    vrtx = Excavation(sim_time)
    res = constrNM(vrtx.obj_func, x0, LB, UB, maxiter=1000, full_output=True, disp=True, save=save)
    del vrtx

    # Results
    printDict(res)

    # MAPE plot
    plot(sim_time)

    load = True  
