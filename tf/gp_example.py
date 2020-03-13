## Example from 
# https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/GaussianProcessRegressionModel

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels

# Generate noisy observations from a known function at some random points.
observation_noise_variance = .5
f = lambda x: np.sin(10*x[..., 0]) * np.exp(-x[..., 0]**2)
observation_index_points = np.random.uniform(-1., 1., 50)[..., np.newaxis]
observations = (f(observation_index_points) +
                np.random.normal(0., np.sqrt(observation_noise_variance)))

index_points = np.linspace(-1., 1., 100)[..., np.newaxis]

kernel = psd_kernels.MaternFiveHalves()

gprm = tfd.GaussianProcessRegressionModel(
    kernel=kernel,
    index_points=index_points,
    observation_index_points=observation_index_points,
    observations=observations,
    observation_noise_variance=observation_noise_variance)

samples = gprm.sample(10)
# ==> 10 independently drawn, joint samples at `index_points`.

## show some visuals of the generated data
import matplotlib
%matplotlib inline

plt = matplotlib.pyplot

plt.plot(index_points, samples[7])
plt.scatter(observation_index_points, observations)


