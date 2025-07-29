import numpy as np

# This code snippet serves as a minimal usage example to DESSCA.
# Firstly, import the dessca_model from DESSCA.py and create a corresponding object.
# Make Sure to have DESSCA.py in the same folder as its application file

from dessca import dessca_model
my_dessca_instance0 = dessca_model(box_constraints=[[-1, 1],
                                                    [-1, 1]],
                                   state_names=["x1", "x2"],
                                   bandwidth=0.5)

# This model instance can be used on a two-dimensional state space.
# Now let's make use of its functionality by viewing the state-space coverage of a dataset.
# Here are some samples:

samples_2d = np.array([[-0.8, -0.8],
                       [0.8, -0.8],
                       [-0.8, 0.8],
                       [0, 0]])

my_dessca_instance0.update_coverage_pdf(data=np.transpose(samples_2d))
my_dessca_instance0.plot_scatter()

# And a corresponding coverage heatmap

my_dessca_instance0.plot_heatmap()

# The coverage pdf is now updated with the given distribution.
# DESSCA can now suggest where to place the next sample.

next_sample_suggest = my_dessca_instance0.sample_optimally()
print(next_sample_suggest)

# As was to be expected, the suggestion is in the upper right corner of the state space.
# Update the coverage density and view the new distribution:

my_dessca_instance0.update_coverage_pdf(data=np.transpose([next_sample_suggest]))
my_dessca_instance0.plot_scatter()

# Let's have a look at the density:

my_dessca_instance0.plot_heatmap()

# The scatter plots can also be rendered in an online fashion (100 samples):

my_dessca_instance1 = dessca_model(box_constraints=[[-1, 1],
                                                    [-1, 1]],
                                  state_names=["x1", "x2"],
                                  bandwidth=0.1,
                                  render_online=True)

next_sample_suggest = my_dessca_instance1.update_and_sample()
for _ in range(100):
    next_sample_suggest = my_dessca_instance1.update_and_sample(np.transpose([next_sample_suggest]))

# Further, we can parametrize a memory buffer to only memorize a limited number of past samples:

my_dessca_instance2 = dessca_model(box_constraints=[[-1, 1],
                                                    [-1, 1]],
                                  state_names=["x1", "x2"],
                                  bandwidth=0.1,
                                  render_online=True,
                                  buffer_size=25)

next_sample_suggest = my_dessca_instance2.update_and_sample()
for _ in range(100):
    next_sample_suggest = my_dessca_instance2.update_and_sample(np.transpose([next_sample_suggest]))

# See how forgetting past samples leads to a group of samples in a similar area?
# Lastly, we can also choose to use a specific reference coverage density:

def reference_coverage(X):
    # for uniform distribution on a given shape the value range of the reference coverage is not important
    x0 = X[0]
    x1 = X[1]
    return np.less(x0**2 + x1**2, 1).astype(float)

my_dessca_instance3 = dessca_model(box_constraints=[[-1, 1],
                                                    [-1, 1]],
                                  state_names=["x1", "x2"],
                                  bandwidth=0.1,
                                  render_online=True,
                                  reference_pdf=reference_coverage)

next_sample_suggest = my_dessca_instance3.update_and_sample()
for _ in range(100):
    next_sample_suggest = my_dessca_instance3.update_and_sample(np.transpose([next_sample_suggest]))

    
# DESSCA can also be used for downsampling. Let's firstly find a large dataset that we would like to reduce in size:

x1_samples = np.random.triangular(-1, 0.75, 1, size=(1000, 1))
x2_samples = np.random.triangular(-1, -0.75, 1, size=(1000, 1))
samples_2d = np.append(x1_samples, x2_samples, axis=1)

my_dessca_instance4 = dessca_model(box_constraints=[[-1, 1],
                                                    [-1, 1]],
                                   state_names=["x1", "x2"],
                                   bandwidth=0.5)
my_dessca_instance4.update_coverage_pdf(data=np.transpose(samples_2d))
my_dessca_instance4.plot_scatter()

#Now, DESSCA can be used to reduce the set down to a specified number of remaining samples while trying to preserve the original distribution:

my_dessca_instance5 = dessca_model(box_constraints=[[-1, 1],
                                                    [-1, 1]],
                                   state_names=["x1", "x2"],
                                   bandwidth=0.5)

my_dessca_instance5.downsample(data=np.transpose(samples_2d), target_size=100)
my_dessca_instance5.plot_scatter()

# Comparing the edge distributions, it can be seen that they are still almost the same despite removing 90 % of the dataset's content.
    
