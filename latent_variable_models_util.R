## ported to R from
## https://github.com/krasserm/bayesian-machine-learning/blob/master/latent_variable_models_util.py


n_true = c(320, 170, 510)

mu_true = matrix(c(5.0, 5.0, 6.5, 8.0, 9.5, 7.5), 
                 nrow = 3, ncol = 2,
                 byrow = TRUE)

sigma_true = list(matrix(c(1.0,  0.0,  0.0, 0.7), nrow = 2, ncol = 2), 
                  matrix(c(2.0, -0.7, -0.7, 1.0), nrow = 2, ncol = 2), 
                  matrix(c(0.7,  0.9,  0.9, 5.0), nrow = 2, ncol = 2))

generate_data <- function(n, mu, sigma){
  library(mvtnorm)
  # x <- matrix(0, nrow = sum(n), ncol = length(mu[[1]]))
  
  ## one hot encode the latent variable
  T_ <- matrix(0, sum(n), length(n))
  ## the random data
  x <- NULL
  y <- NULL
  for (iC in 1:length(n)){
    x <- rbind(x,
               rmvt(n[iC], delta = mu[[iC]], sigma = sigma[[iC]], df = Inf))
               # MASS::mvrnorm(n[iC], mu = mu[[iC]], Sigma = sigma[[iC]]))
    
    y <- rbind(y, matrix(rep(iC, n[iC]), nrow = n[iC]))
    if (iC == 1) {
      startIdx = 1
      endIdx = n[iC]
    } else {
      startIdx = n[iC - 1] + 1
      endIdx = n[iC - 1] + n[iC]
    }
    
    T_[startIdx:endIdx, iC] <- 1
    
  }
  ## return output, return y as well
  output <- list(x = x, y = y, T_ = T_)
  return(output)
}



plot_data <- function(x, y = NULL, alpha = 0.7, add_contour = FALSE){
  library(ggplot2)
  df <- as.data.frame(cbind(x, y))
  p <- ggplot(df, aes(x = V1, y = V2), alpha = alpha) 
  
  if (is.null(y)){
    p <- p + geom_point(colour = "grey50")
  } else {
    p <- p + geom_point(aes(colour = as.factor(V3)))
    
    ## add contour lines --> only when y is not NUll
    if (add_contour){
      p <- p + geom_density_2d(aes(colour = as.factor(V3)))
    }
    
    p <- p + labs(colour = "Latent Variables")
  }
  
  p <- p + theme_bw()
  print(p)
}


