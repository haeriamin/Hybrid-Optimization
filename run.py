# Copyright (C) 2020 Amin Haeri [ahaeri92@gmail.com]

# Update rate [Hz] Fixed at 60
# Parameters:
# 1. Internal friction coefficient
# 2. External friction coefficient
# 3. Grain radius [m]
# 4. Surcharge factor
# 5. Iteration number

import pickle

from obj_func import Excavation
from constr_nm import constrNM, printDict
from plot import plot

# Load/Save last best solution? True/False
load = False  
save = True

# Excavator depths
depths = [0.02, 0.05]  # m

# Simulation time
sim_times = [5, 10, 20]  # sec

# Lower bound
LB = [0.6, 0.2, 0.008, 0.01, 5.0]

# Upper bound
UB = [0.8, 0.4, 0.012, 1.00, 20.0]

for sim_time in sim_times:
    # Initial solution
    if load:
        with open('last_solution.pkl', 'r') as f:
            x0 = list(pickle.load(f))
            print ' Last best solution:', x0[0], ',', x0[1], ',', x0[2], ',', x0[3], ',', x0[4]
    else:
        x0 = [0.7, 0.364, 0.012, 1.00, 20.0]

        # 5cm case -- 20 sec -- took 150 min -- ? fevals
        # x0 = [6.60321784e-01, 3.98142579e-01, 1.16217771e-02, 2.20756469e-01, 1.78680532e+01] # 8.9% in 20

        # 5cm case -- 5, 10, 20 sec -- took 30 min
        # x0 = [6.83741285e-01, 3.35455728e-01, 1.10511245e-02, 2.11047748e-01, 1.98022255e+01]  # 8.2% (34 sec runtime and 1.2% mpe)

        # 2cm case -- starting with 5cm best solution -- took 10 min
        # x0 = [7.04127367e-01, 3.35380587e-01, 1.09857695e-02, 3.47156506e-01, 1.98045868e+01]  # 6.2% (24 sec runtime and ?% mpe)

        # Both 2cm and 5cm cases -- 5, 10, 20 sec -- took 40 min
        # x0 = [7.11653299e-01, 3.97859022e-01, 1.19631175e-02, 2.47373440e-01, 1.97698647e+01]  # 18.6% 

    # Optimizer
    vrtx = Excavation(sim_time, depths)
    res = constrNM(vrtx.obj_func, x0, LB, UB, maxfun=200, full_output=True, disp=True, save=save)  # maxiter=200 
    del vrtx

    # Results
    printDict(res)

    # MAPE plot
    plot(sim_time)

    load = True  

# plot(5)
