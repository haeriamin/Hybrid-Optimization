# Optimizing Hybrid Model Parameters in Vortex Studio

The optimization is done via the gradient-free Nelder-Mead method (aka Downhill Simplex Algorithm). The code is specified for simulating soil cutting operations (e.g. excavation).


## Code structure

```
|- run.py
|   - constr_nm.py
|       - nelder_mead.py
|           - obj_func.py
|           - ref.py
|               - input/
|       - plot.py
|   - output/
|- test.py
```

* `run.py`: Script for running the optimization.

    * The initial, lower and upper bounds of optimization variables are defined here.
        ```python
        X0 = [,]
        LB = [,]
        UB = [,]
        ```

    * Some other settings including loading/saving optimal solution, and excavation depth and time can also be set.
        ```python
        load = True/False
        save = True/False
        depths = [,]  # [m]
        sim_times = [,]  # [sec]
        ```


* `constr_nm.py`: Implementation of the constrained Nelder-Mead method [(reference)](https://github.com/alexblaessle/constrNMPy). This file is called from `run.py`.

* `nelder_mead.py`: Implementation of the Nelder-Mead method [(reference)](https://github.com/scipy/scipy/blob/master/scipy/optimize/optimize.py). This file is called from `constr_nm.py`.

* `obj_func.py`: Visualization code for displaying rollouts such as the example animation.

* `ref.py`: example connecting the model to input dummy data.

* `input/`: example connecting the model to input dummy data.

* `plot.py`: example connecting the model to input dummy data.

* `output/`: example connecting the model to input dummy data.

* `test.py`: example connecting the model to input dummy data.


## Setup

### Requirements

* VxSim
* numpy
* pickle
