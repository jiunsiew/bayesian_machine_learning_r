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





# def generate_data(n, mu, sigma):
#   X = []
# y = []
# for i, n_c in enumerate(n):
#   X.append(mvn(mu[i], sigma[i]).rvs(n_c))
# y.append([i] * n_c)
# 
# X = np.vstack(X)
# y = np.concatenate(y).reshape(-1, 1)
# 
# return X, OneHotEncoder(categories='auto', sparse=False).fit_transform(y)
# 
# 
# def plot_data(X, color='grey', alpha=0.7):
#   plt.scatter(X[:,0], X[:,1], c=color, alpha=alpha)
# 
# 
# def plot_densities(X, mu, sigma, alpha=0.5):
#   grid_x, grid_y = np.mgrid[X[:,0].min():X[:,0].max():200j,
#                             X[:,1].min():X[:,1].max():200j]
# grid = np.stack([grid_x, grid_y], axis=-1)
# 
# for mu_c, sigma_c in zip(mu, sigma):
#   plt.contour(grid_x, grid_y, mvn(mu_c, sigma_c).pdf(grid), colors='grey', alpha=alpha)
# 
# 
# def plot_gmm_plate(filename="gmm.png", dpi=100):
#   pgm = daft.PGM([3.0, 2.5], origin=(0, 0))
# pgm.add_node(daft.Node("theta", r"$\mathbf{\theta}$", 1, 2, fixed=True))
# pgm.add_node(daft.Node("ti", r"$\mathbf{t}_i$", 1, 1))
# pgm.add_node(daft.Node("xi", r"$\mathbf{x}_i$", 2, 1, observed=True))
# pgm.add_edge("theta", "ti")
# pgm.add_edge("theta", "xi")
# pgm.add_edge("ti", "xi")
# pgm.add_plate(daft.Plate([0.4, 0.5, 2.2, 1.0], label=r"$N$"))
# ax = pgm.render()
# ax.text(0.8, 0.5, 'Gaussian mixture model')
# pgm.savefig(filename, dpi=dpi)