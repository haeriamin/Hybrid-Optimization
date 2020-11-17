# Optimizing Hybrid Model Parameters in Vortex Studio

The optimization is done via the gradient-free Nelder-Mead method (aka Downhill Simplex Algorithm). The code is specified for simulating soil cutting operations (e.g. excavation).

<img src="https://github.com/haeriamin/Hybrid-Optimization/blob/main/output/excav.gif" alt="drawing" width="400"> <img src="https://github.com/haeriamin/Hybrid-Optimization/blob/main/output/mape5-10-20_2-5cm.png" alt="drawing" width="440"> 


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

* `run.py`: Run the optimization.

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


* `constr_nm.py`: Implement the constrained Nelder-Mead method [(reference)](https://github.com/alexblaessle/constrNMPy).

* `nelder_mead.py`: Implement the Nelder-Mead method [(reference)](https://github.com/scipy/scipy/blob/master/scipy/optimize/optimize.py).

    * This is modified to terminate the optimization loop when no significant error changes happen (e.g. `<1`%) during the last specified iterations by setting e.g. `history = 10` as fallows:

        ```python
        if iterations > history+2:
            for i in range(2,history+2):
                fval_sum += abs(fval_history[-1] - fval_history[-i])
            if fval_sum/history < 1:
                break
        ```

* `obj_func.py`: Implement the objective function.

    * The Vortex (excavation) model is called here and implemented in:

        ```python
        def run_vortex(self, x, depth):
            ...
        ```

    * The mean absolute percentage error (MAPE) is calculated using the results from Vortex and experiment.

    * The Vortex files and reference (experimental) results should already be provided in folder `input/`.

* `ref.py`: Read reference (experimental) results from the files provided in `input/`.

* `plot.py`: Plot MAPE versus number of function evaluations, and save in folder `output/`.

* `test.py`: Finally, test the Vortex (excavation) model via the optimal solution and see the results.


## Requirements

* VxSim
* numpy
* pickle
