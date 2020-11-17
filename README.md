# Optimizing Hybrid model parameters in Vortex Studio Developed by CM-Labs Inc.

This optimization or tuning is done via gradient-free Nelder-Mead (DownHill Simplex) method.

## Code structure

```
|- run.py
|   |- constr_nm.py
|       |- nelder_mead.py
|           |- obj_func.py
|           |- ref.py
|               |- input/
|   |- plot.py
|       |- output/

|-- test.py
```

* `run.py`: Script for training, evaluating and generating rollout trajectories.
* `constr_nm.py`: Implementation of the learnable one-step model that returns the next position of the particles given inputs. It includes data preprocessing, Euler integration, and a helper method for building normalized training outputs and targets.
* `nelder_mead.py`: Implementation of the graph network used at the core of the learnable part of the model.
* `obj_func.py`: Visualization code for displaying rollouts such as the example animation.
* `ref.py`: example connecting the model to input dummy data.
* `input/`: example connecting the model to input dummy data.

* `plot.py`: example connecting the model to input dummy data.
* `output/`: example connecting the model to input dummy data.

* `test.py`: example connecting the model to input dummy data.

* `{noise/connectivity/reading}_utils.py`: Util modules for adding noise to the inputs, computing graph connectivity and reading datasets form TFRecords.
