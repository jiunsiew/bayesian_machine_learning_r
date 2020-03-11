library(data.table)
library(magrittr)
library(rstan)
library(stringr)

rm(list = ls())
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores()-1)

## run the stan file
nSamples <- 25
nNew <- 1
x.star <- as.matrix(seq(-5, 5, len=nSamples))
stanData <- list(N=nNew,
                 nr_1=nSamples,
                 nc_1=1,
                 x1=x.star, 
                 nr_2=nSamples,
                 nc_2=1,
                 x2=x.star, 
                 theta_1=1,  
                 theta_2=1,  
                 sigma_n=0.5)

## sample --> just generating some random samples 
simu_fit <- stan(file='gp_stan.stan',
                 data=stanData, iter=1,
                 chains=1, seed=494838, algorithm="Fixed_param")

f_total <- matrix(extract(simu_fit)$f, nrow = nNew)
y_total <- matrix(extract(simu_fit)$y, nrow = nNew)

priorSamples <- data.table(t(f_total)) %>% 
  .[, x := x.star] %>% 
  cbind(data.table(t(y_total))) %>% 
  setnames(., c(paste0("f_", 1:nNew), "x", paste0("y_", 1:nNew))) %>% 
  melt.data.table(., id.vars = c("x")) %>% 
  .[, type := str_sub(variable, 1,1)] %>% 
  .[, sample_idx := str_sub(variable, 3,-1)] %>% 
  .[, variable := NULL] %>% 
  dcast(x+sample_idx~type, value.var = "value")

ggplot(priorSamples, aes(x=x, y=f)) +
  geom_line(aes(group=sample_idx, colour = sample_idx), lty = 2) +
  geom_point(aes(x = x, y = y, colour = sample_idx), alpha = 0.5) +
  theme_bw() +
  xlab("input, x")


