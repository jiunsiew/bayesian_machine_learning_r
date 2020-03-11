library(data.table)
library(magrittr)
library(rstan)
library(stringr)
library(lubridate)

rm(list = ls())
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores()-1)


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


trainDat <- cleanDat[isTrain==TRUE]
testDat <- cleanDat[isTrain==FALSE]


pred_data <- list(nr_1=nrow(trainDat),
                  nc_1=1,
                  x1=as.matrix(trainDat$dt_idx, nrow=nrow(trainDat)),
                  y=trainDat$co2_norm,
                  nr_2=nrow(testDat),
                  nc_2=1,
                  x2=as.matrix(testDat$dt_idx, nrow=nrow(testDat)),
                  theta_1=1, 
                  theta_2=1, 
                  sigma_n=0.1)

pred_fit <- stan(file='gp_stan_pred.stan', 
                 data=pred_data, 
                 iter=1000,
                 warmup=0,
                 chains=1, 
                 seed=5838298, 
                 refresh=1000, 
                 algorithm="Fixed_param")
