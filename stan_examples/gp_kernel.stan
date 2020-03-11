//  GP with own kernels 
//

// The input data is a vector 'y' of length 'N'.
data {
  int<lower=1> nr_1;  // number of samples of input 1
  int<lower=1> nc_1;   // number of columns of input 1 (parameters)
  int<lower=1> nr_2;  // number of samples of input 2
  int<lower=1> nc_2;   // number of columns of input 2 (parameters)
  
  // input 1
  matrix[nr_1,nc_1] X1;
  //input 2
  matrix[nr_2,nc_2] X2;
  
  // some fixed parameters
  // characteristic length param for sq exp cov function
  real<lower=0> l;    
  // scale param for sq exp cov function
  real<lower=0> sigma_f;  
  // noise variance
  real<lower=0> sigma_n;  
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
//parameters {
//  real mu;
//  real<lower=0> sigma;
//}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  y ~ normal(mu, sigma);
}

