## from https://peterroelants.github.io/posts/gaussian-process-kernel-fitting/

# Imports
import os
import sys
import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

import bokeh
import bokeh.io
import bokeh.plotting
import bokeh.models
from IPython.display import display, HTML

bokeh.io.output_notebook(hide_banner=True)

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

np.random.seed(42)
tf.random.set_seed(42)

# Load the data and clean it up
# we've already downloaded this data (see gaussian_processes_co2.R)
co2_df = pd.read_csv('~/bayesian_machine_learning_r/data/monthly_in_situ_co2_mlo.csv', \
    na_values='-99.99', \
    dtype=np.float64)
# drop missing values
co2_df.dropna(inplace=True)
# Remove whitespace from column names
co2_df.rename(columns=lambda x: x.strip(), inplace=True)

## rename the date and co2 column to fit with the rest of the code
co2_df.rename(columns={"date_num": "Date", "co2": "CO2"}, inplace = True)

## plot ------------------------------------------------------------------------
fig = bokeh.plotting.figure(
    width=600, height=300, 
    x_range=(1958, 2020), y_range=(310, 420))
fig.xaxis.axis_label = 'Date'
fig.yaxis.axis_label = 'CO₂ (ppm)'
fig.add_layout(bokeh.models.Title(
    text='In situ air measurements at Mauna Loa, Observatory, Hawaii',
    text_font_style="italic"), 'above')
fig.add_layout(bokeh.models.Title(
    text='Atmospheric CO₂ concentrations', 
    text_font_size="14pt"), 'above')
fig.line(
    co2_df.Date, co2_df.CO2, legend_label='All data',
    line_width=2, line_color='midnightblue')
fig.legend.location = 'top_left'
fig.toolbar.autohide = True
bokeh.plotting.show(fig)

# Split the data into observed and to predict
date_split_predict = 2008
df_observed = co2_df[co2_df.Date < date_split_predict]
print('{} measurements in the observed set'.format(len(df_observed)))
df_predict = co2_df[co2_df.Date >= date_split_predict]
print('{} measurements in the test set'.format(len(df_predict)))


# Define mean function which is the means of observations
observations_mean = tf.constant(
    [np.mean(df_observed.CO2.values)], dtype=tf.float64)
mean_fn = lambda _: observations_mean

## KERNEL ----------------------------------------------------------------------
def cov_kernel(smooth_amplitude, smooth_length_scale, 
               periodic_amplitude, periodic_length_scale, 
               periodic_period, periodic_local_length_scale,
               irregular_amplitude, irregular_length_scale, 
               irregular_scale_mixture, 
               observation_noise_variance):

    ## Define the kernel with trainable parameters.
    ## Long term smooth rising trend (Rasmussen eq 5.15)
    # Smooth kernel
    smooth_kernel = tfk.ExponentiatedQuadratic(
        amplitude=smooth_amplitude,
        length_scale=smooth_length_scale)

    ## Allows for decay away from exact periodicity (Rasmussen eq. 5.16)
    # Local periodic kernel
    local_periodic_kernel = (
        tfk.ExpSinSquared(
            amplitude=periodic_amplitude,
            length_scale=periodic_length_scale,
            period=periodic_period) *
        tfk.ExponentiatedQuadratic(
            length_scale=periodic_local_length_scale))

    # Short-medium term irregularities kernel (Rasmussen eq. 5.17)
    irregular_kernel = tfk.RationalQuadratic(
        amplitude=irregular_amplitude,
        length_scale=irregular_length_scale,
        scale_mixture_rate=irregular_scale_mixture)

    # Noise variance of observations
    # Start out with a medium-to high noise
    # observation_noise_variance = tf.exp(
    #     tf.Variable(np.float64(1)), name='observation_noise_variance')

    # Sum all kernels to single kernel containing all characteristics
    kernel = (smooth_kernel + local_periodic_kernel + irregular_kernel)

    return(kernel)

## using code from https://colab.research.google.com/github/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Gaussian_Process_Regression_In_TFP.ipynb#scrollTo=JiJukqfWuXUq
## put prior distributions on each of the hyperparameters to tune
def build_gp(smooth_amplitude, smooth_length_scale, 
             periodic_amplitude, periodic_length_scale, 
             periodic_period, periodic_local_length_scale,
             irregular_amplitude, irregular_length_scale, 
             irregular_scale_mixture, 
             observation_noise_variance):
    """Defines the conditional dist. of GP outputs, given kernel parameters."""

    ## define the kernel given the inputs
    kernel = cov_kernel(smooth_amplitude, smooth_length_scale, 
             periodic_amplitude, periodic_length_scale, 
             periodic_period, periodic_local_length_scale,
             irregular_amplitude, irregular_length_scale, 
             irregular_scale_mixture, 
             observation_noise_variance)

    # Create the GP prior distribution, which we will use to train the model
    # parameters.
    return tfd.GaussianProcess(
        kernel=kernel,
        index_points=batch_date,
        observation_noise_variance=observation_noise_variance)

## sampling from training data
batch_size = 128
batch_co2_dataset = tf.data.Dataset.from_tensor_slices( \
    (df_observed.Date.values.reshape(-1, 1), df_observed.CO2.values)) \
        .shuffle(buffer_size=len(df_observed)) \
            .repeat(count=None).batch(batch_size) 

iterator = iter(batch_co2_dataset)
batch_date, batch_co2 = iterator.get_next()

gp_joint_model = tfd.JointDistributionNamed({
    "smooth_amplitude": tfd.LogNormal(loc=0., scale=np.float64(1.)), 
    "smooth_length_scale": tfd.LogNormal(loc=0., scale=np.float64(1.)),
    "periodic_amplitude": tfd.LogNormal(loc=0., scale=np.float64(1.)), 
    "periodic_length_scale": tfd.LogNormal(loc=0., scale=np.float64(1.)), 
    "periodic_period": tfd.LogNormal(loc=0., scale=np.float64(1.)), 
    "periodic_local_length_scale": tfd.LogNormal(loc=0., scale=np.float64(1.)),
    "irregular_amplitude": tfd.LogNormal(loc=0., scale=np.float64(1.)), 
    "irregular_length_scale": tfd.LogNormal(loc=0., scale=np.float64(1.)), 
    "irregular_scale_mixture": tfd.LogNormal(loc=0., scale=np.float64(1.)), 
    "observation_noise_variance": tfd.LogNormal(loc=0., scale=np.float64(1.)),
    "observations": build_gp,
})

## test
x = gp_joint_model.sample()
lp = gp_joint_model.log_prob(x)

print("sampled {}".format(x))
print("log_prob of sample: {}".format(lp))


## Setup the optimisation ------------------------------------------------------
# Define mini-batch data iterator
# batch_size = 128

# Create the trainable model parameters, which we'll subsequently optimize.
# Note that we constrain them to be strictly positive.
constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

# Smooth kernel hyperparameters
smooth_amplitude_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='smooth_amplitude',
    dtype=np.float64)

smooth_length_scale_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='smooth_length_scale',
    dtype=np.float64)

# Local periodic kernel hyperparameters
periodic_amplitude_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='periodic_amplitude',
    dtype=np.float64)

periodic_length_scale_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='periodic_length_scale',
    dtype=np.float64)

periodic_period_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='periodic_period',
    dtype=np.float64)

periodic_local_length_scale_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='periodic_local_length_scale',
    dtype=np.float64)

# Short-medium term irregularities kernel hyperparameters
irregular_amplitude_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='irregular_amplitude',
    dtype=np.float64)

irregular_length_scale_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='irregular_length_scale',
    dtype=np.float64)

irregular_scale_mixture_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='irregular_scale_mixture',
    dtype=np.float64)

## observation noise
observation_noise_variance_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='observation_noise_variance',
    dtype=np.float64)

## combine trainable variables into a list which are constrained to be positive
trainable_variables = [v.trainable_variables[0] for v in 
                       [smooth_amplitude_var,
                       smooth_length_scale_var,
                       periodic_amplitude_var,
                       periodic_length_scale_var,
                       periodic_period_var,
                       periodic_local_length_scale_var,
                       irregular_amplitude_var,
                       irregular_length_scale_var,
                       irregular_scale_mixture_var,
                       observation_noise_variance_var]]

## want to condition model on training data so define the target_log_prob func
## which we want to minimize
# Use `tf.function` to trace the loss for more efficient evaluation.
@tf.function(autograph=False, experimental_compile=False)
def target_log_prob(smooth_amplitude, smooth_length_scale, 
                    periodic_amplitude, periodic_length_scale, 
                    periodic_period, periodic_local_length_scale,
                    irregular_amplitude, irregular_length_scale, 
                    irregular_scale_mixture, 
                    observation_noise_variance):
  return gp_joint_model.log_prob({
      "smooth_amplitude": smooth_amplitude, 
      "smooth_length_scale": smooth_length_scale,
      "periodic_amplitude": periodic_amplitude, 
      "periodic_length_scale": periodic_length_scale, 
      "periodic_period": periodic_period, 
      "periodic_local_length_scale": periodic_local_length_scale,
      "irregular_amplitude": irregular_amplitude, 
      "irregular_length_scale": irregular_length_scale, 
      "irregular_scale_mixture": irregular_scale_mixture, 
      "observation_noise_variance": observation_noise_variance,
      "observations": batch_co2
  })


## optimise
num_iters = 10000
optimizer = tf.optimizers.Adam(learning_rate=0.002) 

# Store the likelihood values during training, so we can plot the progress
lls_ = np.zeros(num_iters, np.float64)
for i in range(num_iters):
    with tf.GradientTape() as tape:
        loss = -target_log_prob(smooth_amplitude_var, 
                                smooth_length_scale_var, 
                                periodic_amplitude_var,
                                periodic_length_scale_var, 
                                periodic_period_var, 
                                periodic_local_length_scale_var,
                                irregular_amplitude_var,
                                irregular_length_scale_var, 
                                irregular_scale_mixture_var, 
                                observation_noise_variance_var)
        grads = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(grads, trainable_variables))
        lls_[i] = loss

print('Trained parameters:')
print('smooth_amplitude: {}'.format(smooth_amplitude_var._value().numpy()))
print('smooth_length_scale: {}'.format(smooth_length_scale_var._value().numpy()))
print('periodic_amplitude: {}'.format(periodic_amplitude_var._value().numpy()))
print('periodic_length_scale: {}'.format(periodic_length_scale_var._value().numpy()))
print('periodic_period: {}'.format(periodic_period_var._value().numpy()))
print('periodic_local_length_scale: {}'.format(periodic_local_length_scale_var._value().numpy()))
print('irregular_amplitude: {}'.format(irregular_amplitude_var._value().numpy()))
print('irregular_length_scale: {}'.format(irregular_length_scale_var._value().numpy()))
print('irregular_scale_mixture_var: {}'.format(irregular_scale_mixture_var._value().numpy()))
print('observation_noise_variance: {}'.format(observation_noise_variance_var._value().numpy()))
print('Min LL: {}'.format(min(lls_)))

%pylab inline
plt.figure(figsize=(12, 4))
plt.plot(lls_)
plt.xlabel("Training iteration")
plt.ylabel("Log marginal likelihood")
plt.show()

# Having trained the model, we'd like to sample from the posterior conditioned
# on observations. We'd like the samples to be at points other than the training
# inputs.
# predictive_index_points_ = np.linspace(-1.2, 1.2, 200, dtype=np.float64)
predictive_index_points_ = df_predict.Date.values.reshape(-1, 1)
# Reshape to [200, 1] -- 1 is the dimensionality of the feature space.
# predictive_index_points_ = predictive_index_points_[..., np.newaxis]

optimized_kernel = tfk.ExponentiatedQuadratic(amplitude_var, length_scale_var)
gprm = tfd.GaussianProcessRegressionModel(
    kernel=optimized_kernel,
    index_points=predictive_index_points_,
    observation_index_points=observation_index_points_,
    observations=observations_,
    observation_noise_variance=observation_noise_variance_var,
    predictive_noise_variance=0.)

# Create op to draw  50 independent samples, each of which is a *joint* draw
# from the posterior at the predictive_index_points_. Since we have 200 input
# locations as defined above, this posterior distribution over corresponding
# function values is a 200-dimensional multivariate Gaussian distribution!
num_samples = 50
samples = gprm.sample(num_samples)

### v1 -------------------------------------------------------------------------
# Plot NLL over iterations
fig = bokeh.plotting.figure(
    width=600, height=400, 
    x_range=(0, nb_iterations), y_range=(50, 200))
fig.add_layout(bokeh.models.Title(
    text='Negative Log-Likelihood (NLL) during training', 
    text_font_size="14pt"), 'above')
fig.xaxis.axis_label = 'iteration'
fig.yaxis.axis_label = 'NLL batch'
# First plot
fig.line(
    *zip(*batch_nlls), legend_label='Batch data',
    line_width=2, line_color='midnightblue')
# Seoncd plot
# Setting the second y axis range name and range
fig.extra_y_ranges = {
    'fig1ax2': bokeh.models.Range1d(start=130, end=250)}
fig.line(
    *zip(*full_ll), legend_label='All observed data',
    line_width=2, line_color='red', y_range_name='fig1ax2')
# Adding the second axis to the plot.  
fig.add_layout(bokeh.models.LinearAxis(
    y_range_name='fig1ax2', axis_label='NLL all'), 'right')

fig.legend.location = 'top_right'
fig.toolbar.autohide = True
bokeh.plotting.show(fig)

# Show values of parameters found
variables = [
    smooth_amplitude,
    smooth_length_scale,
    periodic_amplitude,
    periodic_length_scale,
    periodic_period,
    periodic_local_length_scale,
    irregular_amplitude,
    irregular_length_scale,
    irregular_scale_mixture,
    observation_noise_variance
]
variables_eval = session.run(variables)

data = list([
    (var.name[:-2], var_eval) 
    for var, var_eval in zip(variables, variables_eval)])
df_variables = pd.DataFrame(
    data, columns=['Hyperparameters', 'Value'])
display(HTML(df_variables.to_html(
    index=False, float_format=lambda x: f'{x:.4f}')))

# Posterior GP using fitted kernel and observed data
gp_posterior_predict = tfd.GaussianProcessRegressionModel(
    mean_fn=mean_fn,
    kernel=kernel,
    index_points=df_predict.Date.values.reshape(-1, 1),
    observation_index_points=df_observed.Date.values.reshape(-1, 1),
    observations=df_observed.CO2.values,
    observation_noise_variance=observation_noise_variance)

# Posterior mean and standard deviation
posterior_mean_predict = gp_posterior_predict.mean()
posterior_std_predict = gp_posterior_predict.stddev()

# Plot posterior predictions

# Get posterior predictions
μ = session.run(posterior_mean_predict)
σ = session.run(posterior_std_predict)

# Plot
fig = bokeh.plotting.figure(
    width=600, height=400,
    x_range=(2008, 2019), y_range=(380, 415))
fig.xaxis.axis_label = 'Date'
fig.yaxis.axis_label = 'CO₂ (ppm)'
fig.add_layout(bokeh.models.Title(
    text='Posterior predictions conditioned on observations before 2008.',
    text_font_style="italic"), 'above')
fig.add_layout(bokeh.models.Title(
    text='Atmospheric CO₂ concentrations', 
    text_font_size="14pt"), 'above')
fig.circle(
    co2_df.Date, co2_df.CO2, legend_label='True data',
    size=2, line_color='midnightblue')
fig.line(
    df_predict.Date.values, μ, legend_label='μ (predictions)',
    line_width=2, line_color='firebrick')
# Prediction interval
band_x = np.append(
    df_predict.Date.values, df_predict.Date.values[::-1])
band_y = np.append(
    (μ + 2*σ), (μ - 2*σ)[::-1])
fig.patch(
    band_x, band_y, color='firebrick', alpha=0.4, 
    line_color='firebrick', legend_label='2σ')

fig.legend.location = 'top_left'
fig.toolbar.autohide = True
bokeh.plotting.show(fig)
#
