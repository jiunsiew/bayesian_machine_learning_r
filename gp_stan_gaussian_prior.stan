data {
  int<lower=1> N;
  real x[N];

  real<lower=0> rho;    // characteristic length param for sq exp cov function
  real<lower=0> alpha;  // scale param for sq exp cov function
  real<lower=0> sigma;  // noise variance
}

transformed data {
  matrix[N, N] cov =   cov_exp_quad(x, alpha, rho)
                     + diag_matrix(rep_vector(1e-10, N));
  matrix[N, N] L_cov = cholesky_decompose(cov);
}

parameters {}
model {}

generated quantities {
  vector[N] f = multi_normal_cholesky_rng(rep_vector(0, N), L_cov);
  vector[N] y;
  for (n in 1:N)
    y[n] = normal_rng(f[n], sigma);
}
