# Optimizing Hybrid model parameters in Vortex Studio Developed by CM-Labs Inc.

This optimization or tuning is done via gradient-free Nelder-Mead (DownHill Simplex) method.

## Code structure
The code package is consist of different files as follows:

```
# Hybrid-Optimization
|-- run.py
|    |-- constr_nm.py
|    |    |-- constr_nm.py
|    |    |    |-- nelder_mead.py
|    |    |    |    |-- obj_func.py
|    |    |    |    |-- ref.py
```

* `run.py`: Script for training, evaluating and generating rollout trajectories.
* `learned_simulator.py`: Implementation of the learnable one-step model that returns the next position of the particles given inputs. It includes data preprocessing, Euler integration, and a helper method for building normalized training outputs and targets.
* `graph_network.py`: Implementation of the graph network used at the core of the learnable part of the model.
* `render_rollout.py`: Visualization code for displaying rollouts such as the example animation.
* `{noise/connectivity/reading}_utils.py`: Util modules for adding noise to the inputs, computing graph connectivity and reading datasets form TFRecords.
*  `model_demo.py`: example connecting the model to input dummy data.

Note this is a reference implementation not designed to scale up to TPUs (unlike the one used for the paper). We have tested that the model can be trained with a batch size of 2 on a single NVIDIA V100 to reach similar qualitative performance (except for the XL and 3D datasets due to OOM).

