## GP with co2 concentration data 
## example in Rassmussen and code derived from 
## https://docs.pymc.io/notebooks/GP-MaunaLoa.html
## 
## using rstan instead of pymc3
rm(list=ls())
library(data.table)
library(magrittr)
library(lubridate)
library(stringr)
library(ggplot2)
theme_set(theme_bw())

## to download data
# co2Fid <- "https://scrippsco2.ucsd.edu/assets/data/atmospheric/stations/in_situ_co2/monthly/monthly_in_situ_co2_mlo.csv"
# 
# ## skip the first 58 lines as these are all fluff
# rawDat <- fread(co2Fid, skip = 58) %>% 
#   setnames(., c("yr", "mth", "date_excel", "date_num",
#                       "co2", "seasonally_adjusted", "fit", 
#                       "seasonally_adjusted_fit",
#                       "co2_filled", "seasonally_adjusted_filled")) 
# fwrite(rawDat, "./data/monthly_in_situ_co2_mlo.csv")

## read data from previously downloaded file
co2Fid <- "./data/monthly_in_situ_co2_mlo.csv"
rawDat <- fread(co2Fid)

## clean up 
cleanDat <- rawDat %>% 
  copy() %>% 
  ## drop the weird date cols
  .[, c("date_excel", "date_num") := NULL] %>% 
  ## replace -99.99 with NA
  .[, lapply(.SD, function(x) ifelse(x == -99.99, NA, x))] %>% 
  ## drop the NA rows
  na.omit(.) %>% 
  ## add date, make each month the 15th
  .[, dt := ymd(paste0(yr, "/", str_pad(mth, width = 2, "left", "0"), "/15"))] %>% 
  ## index -- divide by 365 days which is not exactly what you get in the 
  ## https://docs.pymc.io/notebooks/GP-MaunaLoa.html but close enough
  .[, dt_idx := as.numeric((dt - ymd('1958-03-15'))/365)] %>% 
  ## normalise co2
  .[, co2_norm := (co2 - co2[1])/sd(co2)] %>% 
  ## lastly add the test/train index --> use data up to end 2003 for training
  .[, isTrain := dt < ymd("2004-01-01")]
  
## plot
ggplot(cleanDat, aes(x = dt, y = co2)) + 
  geom_line()

ggplot(cleanDat, aes(x = dt_idx, y = co2_norm)) + 
  geom_line()


## STAN (by me) ----------------------------------------------------------------
## see examples https://betanalpha.github.io/assets/case_studies/gp_part1/part1.html#2_gaussian_process_regression
## other references:
## https://betanalpha.github.io/writing/
## https://github.com/betanalpha/knitr_case_studies/tree/master/gaussian_processes
## https://discourse.mc-stan.org/t/implementig-periodic-covariance-function/1787
## https://github.com/stan-dev/stan/wiki/Adding-a-Gaussian-Process-Covariance-Function
## https://github.com/stan-dev/stan/wiki/Adding-a-Gaussian-Process-Covariance-Function
## https://mc-stan.org/docs/2_19/functions-reference/covariance.html
## https://docs.pymc.io/notebooks/GP-MaunaLoa.html
## https://gitlab.com/hnickisch/gpml-matlab/-/tree/master/doc
## 
library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores() - 2)

## run the stan file
stanData <- list(alpha=3, 
                 rho=5.5,
                 sigma=2,
                 N=sum(cleanDat$isTrain), 
                 x=cleanDat$co2[cleanDat$isTrain==TRUE])

simu_fit <- stan(file='gp_stan_gaussian_prior.stan',
                 data=stanData, iter=1,
                 chains=1, seed=494838, algorithm="Fixed_param")

f_total <- extract(simu_fit)$f[1,]
y_total <- extract(simu_fit)$y[1,]
true_realization <- data.table(f_total, stanData$x)
setnames(true_realization, c("f_total", "x_total"))

## stopped here:
## - need to work out how to use the output of simu_fit 

## paper on using bayesian for item response theory
## https://arxiv.org/pdf/1905.09501.pdf