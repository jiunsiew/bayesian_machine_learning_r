## from https://peterroelants.github.io/posts/gaussian-process-kernel-fitting/

# Imports
import os
import sys
import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_probability as tfp

tf.logging.set_verbosity(tf.logging.ERROR)

import bokeh
import bokeh.io
import bokeh.plotting
import bokeh.models
from IPython.display import display, HTML

bokeh.io.output_notebook(hide_banner=True)

tfd = tfp.distributions
tfk = tfp.math.psd_kernels

np.random.seed(42)
tf.set_random_seed(42)
# tf.random.set_seed(42)  # tf2

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
# Define the kernel with trainable parameters. 
# Note we transform some of the trainable variables to ensure
#  they stay positive.

# Use float64 because this means that the kernel matrix will have 
#  less numerical issues when computing the Cholesky decomposition

## Long term smooth rising trend (Rasmussen eq 5.15)
# Smooth kernel hyperparameters 
smooth_amplitude = tf.exp(
    tf.Variable(np.float64(3)), name='smooth_amplitude')
smooth_length_scale = tf.exp(
    tf.Variable(np.float64(3)), name='smooth_length_scale')
# Smooth kernel
smooth_kernel = tfk.ExponentiatedQuadratic(
    amplitude=smooth_amplitude, 
    length_scale=smooth_length_scale)

## Allows for decay away from exact periodicity (Rasmussen eq. 5.16)
# Local periodic kernel hyperparameters
periodic_amplitude = tf.exp(
    tf.Variable(np.float64(0)), name='periodic_amplitude')
periodic_length_scale = tf.exp(
    tf.Variable(np.float64(1)), name='periodic_length_scale')
periodic_period = tf.exp(
    tf.Variable(np.float64(0)), name='periodic_period')
periodic_local_length_scale = tf.exp(
    tf.Variable(np.float64(2)), name='periodic_local_length_scale')
# Local periodic kernel
local_periodic_kernel = (
    tfk.ExpSinSquared(
        amplitude=periodic_amplitude, 
        length_scale=periodic_length_scale,
        period=periodic_period) * 
    tfk.ExponentiatedQuadratic(
        length_scale=periodic_local_length_scale))

# Short-medium term irregularities kernel hyperparameters
# (Rasmussen eq. 5.17)
irregular_amplitude = tf.exp(
    tf.Variable(np.float64(0)), name='irregular_amplitude')
irregular_length_scale = tf.exp(
    tf.Variable(np.float64(0)), name='irregular_length_scale')
irregular_scale_mixture = tf.exp(
    tf.Variable(np.float64(0)), name='irregular_scale_mixture')
# Short-medium term irregularities kernel
irregular_kernel = tfk.RationalQuadratic(
    amplitude=irregular_amplitude,
    length_scale=irregular_length_scale,
    scale_mixture_rate=irregular_scale_mixture)

## does not include a noise model (eq. 5.18) --> try adding this later

# Noise variance of observations
# Start out with a medium-to high noise
observation_noise_variance = tf.exp(
    tf.Variable(np.float64(1)), name='observation_noise_variance')

# Sum all kernels to single kernel containing all characteristics
kernel = (smooth_kernel + local_periodic_kernel + irregular_kernel)


## Setup the optimisation ------------------------------------------------------
# Define mini-batch data iterator
batch_size = 128

## tf1 version
batch_date, batch_co2 = (
    tf.data.Dataset.from_tensor_slices(
        (df_observed.Date.values.reshape(-1, 1), df_observed.CO2.values))
    .shuffle(buffer_size=len(df_observed))
    .repeat(count=None)
    .batch(batch_size)
    .make_one_shot_iterator()
    .get_next()
)

# Gaussian process with batch data to fit the kernel parameters
gp_batched = tfd.GaussianProcess(
    mean_fn=mean_fn,
    kernel=kernel,
    index_points=batch_date,
    observation_noise_variance=observation_noise_variance)

## tf2 version
# batch_co2_dataset = tf.data.Dataset.from_tensor_slices( \
#     (df_observed.Date.values.reshape(-1, 1), df_observed.CO2.values)) \
#         .shuffle(buffer_size=len(df_observed)) \
#             .repeat(count=None).batch(batch_size) 

# iterator = iter(batch_co2_dataset)
# batch_date, batch_co2 = iterator.get_next()

# Gaussian process with batch data to fit the kernel parameters
# gp_batched = tfd.GaussianProcess(
#     mean_fn=mean_fn,
#     kernel=kernel,
#     index_points=batch_date,
#     observation_noise_variance=observation_noise_variance)


# NLL to minimize
neg_log_likelihood_batch = -gp_batched.log_prob(batch_co2)

# Adam optimizer to minimize NLL --> tf v1
optimize = tf.train.AdamOptimizer(
    learning_rate=0.002).minimize(neg_log_likelihood_batch)

# tf2
# optimize = tf.keras.optimizers.Adam(learning_rate=0.002) 
# optimize.minimize(neg_log_likelihood_batch, var_list=variables)

# Gaussian process on all training data to monitor NLL
#  on full observed dataset.
gp_all_observed = tfd.GaussianProcess(
    mean_fn=mean_fn,
    kernel=kernel,
    index_points=df_observed.Date.values.reshape(-1, 1),
    observation_noise_variance=observation_noise_variance)

log_likelihood_all = -gp_all_observed.log_prob(
    df_observed.CO2.values)

# Fit hyperparameters

# Start session
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

# Training loop
batch_nlls = []  # Batch NLL for plotting
full_ll = []  # Full data NLL for plotting
nb_iterations = 10001
for i in range(nb_iterations):
    # Run optimization for single batch
    _, nlls = session.run([optimize, neg_log_likelihood_batch])
    batch_nlls.append((i, nlls))
    # Evaluate on all observations
    if i % 100 == 0:
        # Evaluate on all observed data
        ll = session.run(log_likelihood_all)
        full_ll.append((i, ll))


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