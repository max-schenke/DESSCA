# DESSCA
**D**ensity **E**stimation-based **S**tate-**S**pace **C**overage **A**cceleration

The provided DESSCA algorithm was designed to aid the state space exploration in reinforcement learning applications.
In many cases where standard exploring starts may be used, 
the degree of freedom that is provided by the initial state can be utilized to a better extent when using DESSCA instead.
Suggestions or experiences concerning applications of DESSCA outside reinforcement learning are welcome!

## Citing
An in-depth explanation of the principle, realization and improvement capabilities of DESSCA can be found in the article 
"Improved Exploring Starts by Kernel Density Estimation-Based State-Space Coverage Acceleration in Reinforcement Learning".
Please cite it when using the provided code:

```
PLACEHOLDER FOR BIBTEX SOURCE
```

## Usage

This code snippet serves as a minimal usage example to DESSCA.
Firstly, import the dessca_model from DESSCA.py and create a corresponding object.
Make Sure to have DESSCA.py in the same folder as its application file

```
import numpy as np
from DESSCA import dessca_model
my_dessca_instance0 = dessca_model(box_constraints=[[-1, 1],
                                                   [-1, 1]],
                                   state_names=["x1", "x2"],
                                   bandwidth=0.5)
```

This model instance can be used on a two-dimensional state space.
Now let's make use of its functionality by viewing the state-space coverage of a dataset.
Here are some samples:

```
samples_2d = np.array([[-0.8, -0.8],
                       [0.8, -0.8],
                       [-0.8, 0.8],
                       [0, 0]])

my_dessca_instance0.update_coverage_pdf(data=np.transpose(samples_2d))
my_dessca_instance0.plot_scatter()
```

