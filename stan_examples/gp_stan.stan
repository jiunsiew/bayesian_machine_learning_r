// simple stan file to generate gaussian process samples 
// not conditioned on any input data

data {
  int<lower=1> N;         // Number of samples to generate
  
  // Input 1
  int<lower=1> nr_1;      // number of samples input 1
  int<lower=1> nc_1;      // number covariates input 1
  matrix[nr_1,nc_1] x1;   // covariate input 1
  
  // Input 2
  int<lower=1> nr_2;      // number of samples input 2
  int<lower=1> nc_2;      // number covariates input 2 -- should be equal nc_1
  matrix[nr_2,nc_2] x2;  // covariate input 2

  // Kernal parameters
  real<lower=0> theta_1;  // scale param for sq exp cov function
  real<lower=0> theta_2;  // characteristic length param for sq exp cov function
  
  // Noise
  real<lower=0> sigma_n;  // noise variance
}

transformed data {
  // define the covariance matrix --> manually calculate it here
  // see Rasmussen eq. 5.15
  matrix[nr_1,nr_2] cov; 
  row_vector[nc_1] dist;
  real sqDistSum;
  for (i in 1:nr_1){
    for (j in 1:nr_2){
      dist = x1[i] - x2[j];
      sqDistSum = dot_self(dist);
      cov[i,j] = square(theta_1)*exp(-sqDistSum/(2*square(theta_2)));
    }
  }
  // add some noise for numerical stability
  cov = cov + diag_matrix(rep_vector(1e-10, nr_1));
}

parameters {}
model {}

generated quantities {
  matrix[nr_1,nr_2] L_cov = cholesky_decompose(cov);
  vector[nr_1] multiVarNorm;
  matrix[N,nr_1] f;
  matrix[N,nr_1] y;
  
  for (n in 1:N){
    multiVarNorm = multi_normal_cholesky_rng(rep_vector(0, nr_1), L_cov);
    
    // generate multivariate normal samples
    f[n] = multiVarNorm';
    
    // sample outputs using mean from f
    for (i in 1:nr_1){
      y[n,i] = normal_rng(multiVarNorm[i], sigma_n);
    }
  }
}
