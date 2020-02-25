## latent variable models 1
## see https://github.com/krasserm/bayesian-machine-learning
## Latent variable models - part 1: Gaussian mixture models and the EM alg
## 
## re-create the functions/code.  for text and wonderful explanations, 
## see the notebook 

## source the util file
source("latent_variable_models_util.R")

## generate the data and unpack outputs
dat <- generate_data(n_true, mu_true, sigma_true)

X <- dat$x    # the covariate data
y <- dat$y    # the target data
T_ <- dat$T_  # latent variables 

## visualise with contours and colour coded by groups
plot_data(X, y, add_contour = T)


## expectation step
#' @param x (N,D) matrix observed data
#' @param pi (C) vector priors of latent variable T  p(T|Theta)
#' @param mu (C,D) matrix mean of mixture components 
#' @param sigma list of size C, with matrices of size (D,D) 
#' covariances of mixture components
#' @return q (N, C) matrix posterior p(T|X,Theta)
e_step <- function(x, pi, mu, sigma){
  N <- nrow(x)          # number of samples
  C <- nrow(mu)         # number of latent variables
  q <- matrix(0, N, C)  # posterior probabilities
  
  for (iC in 1:C){
    ## get the density of data points --> conditional distribution of data 
    ## p(x|t_c = 1, Theta), where Theta = c(mu, sigma)
    mvnDensity <- mvtnorm::dmvt(x = X, 
                                delta = mu[iC,], 
                                sigma = sigma[[iC]], 
                                df = Inf, 
                                log = FALSE)
    
    ## calculate the posterior p(T|X,Theta) 
    ## by scaling density with prior 
    q[,iC] <- mvnDensity*pi[iC]
    
  }
  
  ## scale s.t. sum of columns = 1
  q <- q/rowSums(q)
  
  return(q)
}


## 'M' step
#' @param x (N,D) matrix observed data
#' @param q (N,C) matrix posterior distbn
#' @return output list containing updated mu (C,D) matrix, pi (C) vector 
#' and sigma list of size (C) with matrix (D,D) in each element
m_step <- function(x, q){
  N <- nrow(x)   # number of data samples 
  D <- ncol(x)   # number of covariates/dimension of multivariate distbn
  C <- ncol(q)   # number of latent variables
  
  
  ## "new"/updated prior probabilities (Eq. 16)
  pi <- colSums(q)/N
  
  ## "new"/updated mixture component means (Eq. 17)
  mu <- (t(q) %*% x)/rowSums(t(q))
  
  ## mixture component covariances (Eq. 18)
  sigma <- lapply(1:C, function(x) matrix(0, nrow = D, ncol = D))
  for (iC in 1:C){
    delta <- x - matrix(mu[iC,], nrow = N, ncol = D, byrow = TRUE)
    sigma[[iC]] <- t(q[, iC]*delta) %*% delta /sum(q[, iC])
  }
  
  output <- list(pi = pi, 
                 mu = mu, 
                 sigma = sigma)
  return(output)
}



## lower bound
lower_bound <- function(x, pi, mu, sigma, q){
  N <- nrow(q)   # number of data samples
  C <- ncol(q)   # number of latent variables
  
  ## lower bound  (Eq. 19)
  ll <- matrix(0, nrow = N, ncol = C)
  for (iC in 1:C){
    ll[, iC] <- mvtnorm::dmvt(x, 
                              delta = mu[iC, ], 
                              sigma = sigma[[iC]], 
                              df = Inf, 
                              log = TRUE)
  }
  
  sum(q* (ll + 
            matrix(log(pi), nrow = N, ncol = C, byrow = TRUE) - 
            log(pmax(q, 1e-8))))
  
  
}


## random init 
## generate some random parameters 
random_init_params <- function(x, C){
  D <- ncol(x)
  
  pi <- rep(1, C)/C   # uniform priors
  
  # means --> multivariate norm distbn
  mu <- mvtnorm::rmvt(C, 
                      delta = colMeans(x), 
                      sigma = matrix(c(var(x[, 1]), 0, 
                                       0, var(x[, 2])), nrow = 2, ncol = 2), 
                      df = Inf)
  
  # covariances identity matrices
  sigma <- lapply(1:C, function(x) diag(D))
  
  output <- list(pi = pi, 
                 mu = mu, 
                 sigma = sigma)
  return(output)
}




## train function
train <- function(X, C, n_restarts=10, max_iter=50, rtol=1e-3){
  # setup
  q_best     <- NA
  pi_best    <- NA
  mu_best    <- NA
  sigma_best <- NA
  lb_best    <- -Inf
  
  for (iR in 1:n_restarts){
    init_params <- random_init_params(X, C)
    pi <- init_params$pi
    mu <- init_params$mu
    sigma <- init_params$sigma
    
    prev_lb <- NA
    
    try({
      for (iI in 1:max_iter){
        print(paste0("iter", iI))
        
        q <- e_step(X, pi, mu, sigma)
        
        mOutput <- m_step(X, q)
        pi    <- mOutput$pi
        mu    <- mOutput$mu
        sigma <- mOutput$sigma
        lb    <- lower_bound(X, pi, mu, sigma, q)
        
        print(paste0("pi: ", paste(round(pi, 6), collapse = ", ")))
        print(paste0("mu: ", paste(round(mu, 6), collapse = ", ")))
        print(paste0("lb: ", round(lb, 6)))
        
        # store
        if (lb > lb_best){
          q_best <- q
          pi_best <- pi
          mu_best <- mu
          sigma_best <- sigma
          lb_best <- lb
        }
        
        if (!is.na(prev_lb)){
          if (abs((lb - prev_lb)/prev_lb) < rtol){
            print(paste0("rtol limit reached...early stopping"))
            break
          }
        }
        
        prev_lb <- lb
      }
      
      output <- list(pi_best = pi_best,
                     mu_best = mu_best,
                     sigma_best = sigma_best,
                     q_best = q_best,
                     lb_best = lb_best)
      return(output)
    })
  }
  
}


em <- train(X, C = 3, n_restarts = 10, max_iter = 1000, rtol = 1e-23)

em$mu_best
mu_true


em$sigma_best
sigma_true


## try different C's
lbs <- NULL
for (iCC in 1:8){
  emC <- train(X, iCC, n_restarts = 10, max_iter = 100, rtol = 1e-23)
  lbs <- rbind(lbs, emC$lb_best)
}
plot(lbs)
