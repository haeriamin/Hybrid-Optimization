# Optimizing Hybrid Model Parameters in Vortex Studio

The optimization is done via the gradient-free Nelder-Mead method (aka Downhill Simplex Algorithm). The code is specified for simulating soil cutting operations (e.g. excavation).


## Code structure

```
|- run.py
|   - constr_nm.py
|       - nelder_mead.py
|           - obj_func.py
|               - input/
|               - ref.py
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


* `constr_nm.py`: Implementation of the constrained Nelder-Mead method [(reference)](https://github.com/alexblaessle/constrNMPy).

* `nelder_mead.py`: Implementation of the Nelder-Mead method [(reference)](https://github.com/scipy/scipy/blob/master/scipy/optimize/optimize.py).

    * This is modified to terminate the optimization loop when no significant error changes happen (e.g. `<1`%) during the last specified iterations by setting e.g. `history = 10` as fallows:

        ```python
        if iterations > history+2:
            for i in range(2,history+2):
                fval_sum += abs(fval_history[-1] - fval_history[-i])
            if fval_sum/history < 1:
                break
        ```

* `obj_func.py`: Implementation of the objective function.

    * The Vortex (excavation) model is implemented and called here.

        ```python
        def run_vortex(self, x, depth):
            .
            .
            .
        ```

    * The mean absolute percentage error (MAPE) is calculated using the results from Vortex and experiment.

    * The Vortex files and reference (experimental) results should already be provided in folder `input/`.

* `ref.py`: Reading reference (experimental) results from the files provided in `input/`.

* `plot.py`: Visualization code for displaying rollouts such as the example animation.

    * `output/`: example connecting the model to input dummy data.

* `test.py`: example connecting the model to input dummy data.


## Setup

### Requirements

* VxSim
* numpy
* pickle
