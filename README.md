# Optimizing Hybrid model parameters in Vortex Studio

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

* `run.py`: Script for running the optimization. Here,

    * The optimization variables and their initial, lower and upper bound can be defined.

    * Some other settings including loading/saving optimal solution, and excavation depth and time can be set.
        `load = True/False`
        `save = True/False`
        `depths = []  # [m]`
        `sim_times = []  # [sec]`


* `constr_nm.py`: Implementation of the learnable one-step model that returns the next position of the particles given inputs. It includes data preprocessing, Euler integration, and a helper method for building normalized training outputs and targets.

* `nelder_mead.py`: Implementation of the graph network used at the core of the learnable part of the model.

* `obj_func.py`: Visualization code for displaying rollouts such as the example animation.

* `ref.py`: example connecting the model to input dummy data.

* `input/`: example connecting the model to input dummy data.

* `plot.py`: example connecting the model to input dummy data.

* `output/`: example connecting the model to input dummy data.

* `test.py`: example connecting the model to input dummy data.


## Setup