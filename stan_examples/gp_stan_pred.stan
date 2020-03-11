// from https://github.com/betanalpha/knitr_case_studies/blob/master/gaussian_processes/gp_part1/predict_gauss.stan
// gaussian process conditioned on observed data
// creating own kernel function here based on Rasmussen co2 example

functions{
  // covariance kernel
  matrix cov_kernel(matrix x1, matrix x2, real theta_1, real theta_2){
    int nr_1 = rows(x1);
    int nc_1 = cols(x1);
    int nr_2 = rows(x2);
    int nc_2 = cols(x2);
    
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
    // cov = cov + diag_matrix(rep_vector(1e-10, nr_1));
    return(cov);
  }


  // prediction
  vector gp_pred_rng(matrix xNew, vector yTrain, matrix xTrain, 
                     real theta_1, real theta_2, real sigma, real delta){
    int nTrain = rows(yTrain);
    int nCovariate = cols(xTrain);
    int nPred = rows(xNew);
    
    vector[nPred] f2;
    matrix[nTrain, nTrain] K;
    matrix[nTrain, nTrain] L_K;
    vector[nTrain] L_K_div_y1;
    vector[nTrain] K_div_y1;
    matrix[nTrain, nPred] k_x1_x2;
    vector[nPred] f2_mu;
    matrix[nTrain, nPred] v_pred;
    matrix[nPred, nPred] cov_f2;
    
    K = cov_kernel(xTrain, xTrain, theta_1, theta_2) 
         + diag_matrix(rep_vector(square(sigma), nCovariate));
    L_K = cholesky_decompose(K);
    L_K_div_y1 = mdivide_left_tri_low(L_K, yTrain);
    K_div_y1 = mdivide_right_tri_low(L_K_div_y1', L_K)';
    
    
    k_x1_x2 = cov_kernel(xTrain, xNew, theta_1, theta_2);
    f2_mu = (k_x1_x2' * K_div_y1);
    v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
    cov_f2 =   cov_kernel(xNew, xNew, theta_1, theta_2) - v_pred' * v_pred 
               + diag_matrix(rep_vector(delta, nPred));
               
    f2 = multi_normal_rng(f2_mu, cov_f2);
    return f2;
  }
  
  // vector gp_pred_rng(real[] x2, vector y1, real[] x1, 
  //                    real alpha, real rho, real sigma, real delta) {
  //   int N1 = rows(y1);
  //   int N2 = size(x2);
  //   vector[N2] f2;
  //   {
  //     matrix[N1, N1] K =   cov_exp_quad(x1, alpha, rho)
  //     + diag_matrix(rep_vector(square(sigma), N1));
  //     matrix[N1, N1] L_K = cholesky_decompose(K);
  //     
  //     vector[N1] L_K_div_y1 = mdivide_left_tri_low(L_K, y1);
  //     vector[N1] K_div_y1 = mdivide_right_tri_low(L_K_div_y1', L_K)';
  //     matrix[N1, N2] k_x1_x2 = cov_exp_quad(x1, x2, alpha, rho);
  //     vector[N2] f2_mu = (k_x1_x2' * K_div_y1);
  //     matrix[N1, N2] v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
  //     matrix[N2, N2] cov_f2 =   cov_exp_quad(x2, alpha, rho) - v_pred' * v_pred
  //     + diag_matrix(rep_vector(delta, N2));
  //     f2 = multi_normal_rng(f2_mu, cov_f2);
  //   }
  //   return f2;
  // }
}


// data {
//   int<lower=1> N;
//   real x[N];
//   vector[N] y;
// 
//   int<lower=1> N_predict;
//   real x_predict[N_predict];
// 
//   real<lower=0> theta_2;
//   real<lower=0> theta_1;
//   real<lower=0> sigma;
// }

data {
  // int<lower=1> N;         // Number of samples to generate
  
  // Input 1 --> train data
  int<lower=1> nr_1;      // number of samples input 1
  int<lower=1> nc_1;      // number covariates input 1
  matrix[nr_1,nc_1] x1;   // covariate input 1
  
  vector[nr_1] y;         // observations training data
  
  // Input 2 --> new data
  int<lower=1> nr_2;      // number of samples input 2
  int<lower=1> nc_2;      // number covariates input 2 -- should be equal nc_1
  matrix[nr_2,nc_2] x2;  // covariate input 2

  // Kernal parameters
  real<lower=0> theta_1;  // scale param for sq exp cov function
  real<lower=0> theta_2;  // characteristic length param for sq exp cov function
  
  // Noise
  real<lower=0> sigma_n;  // noise variance
}

 
// transformed data {
//   matrix[nr_1,nr_2] cov  = cov_kernel(x1, x1, theta_1, theta_2) + 
//                            diag_matrix(rep_vector(1e-10, nr_1)); 
//   // matrix[N, N] cov =   cov_exp_quad(x, alpha, rho)
//   //                    + diag_matrix(rep_vector(1e-10, N));
//   matrix[nr_1,nr_2] L_cov = cholesky_decompose(cov);
// }

parameters {}
model {}

// stopped here:
// issues with sampling.
// Chain 1: Exception: Exception: add: Rows of m1 (545) and rows of m2 (1) must match in size  (in 'model95e25bdd736_gp_stan_pred' at line 46)
// (in 'model95e25bdd736_gp_stan_pred' at line 136)

generated quantities {
  vector[nr_2] f_predict = gp_pred_rng(x2, y, x1, theta_1, theta_2, sigma_n, 1e-10);
  vector[nr_2] y_predict;

  for (n in 1:nr_2)
    y_predict[n] = normal_rng(f_predict[n], sigma_n);
}

