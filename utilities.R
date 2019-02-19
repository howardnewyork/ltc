###  Utilities for Long Term Care Incidence Rate Analysis

library(dplyr)
library(readr)
library(tidyr)
library(ggplot2)

library(keras)

library(xgboost)
library(purrr)
library(glmnet)


## Required shortcut functions
p0=paste0


## Directories

working_dir = p0(getwd(),"/")
data_dir = p0(working_dir, "data/")
output_dir = p0(working_dir, "output/")
stan_dir = p0(working_dir, "stan/")
csv_dir = p0(data_dir,"csv/")


#' Loads in the Individual csv files and combines them in a single incidence.RData file

combine_csv  = function(){
  csv_files = list.files(path = csv_dir)
  
  for (f in csv_files){
    if (is.null(data)){
      incidence=read_csv(p0(csv_dir, f))
    } else {
      incidence=rbind(incidence, read_csv(p0(csv_dir, f)))
    }
    
  }
  incidence %>% print(width= Inf)
  object.size(incidence) / 2^20
  str(incidence)
  save(incidence, file = p0(data_dir, "incidence.RData"))
  
  "done"
  
}

#' The mean of the vector x excuding zero values

mean_ex_0 = function(x){
  mean(x[x!=0])
}

#' loads incidence rate database and cleans file
clean_incidence = function(){
  load(file = p0(data_dir, "incidence.RData"))
  #incidence = incidence[1:10000,]
  
  
  incidence = incidence %>%
    mutate(Gender = recode(Gender, M = "m", `FALSE` = "f"),
           IssueAgeBucket = as.numeric(recode(IssueAgeBucket, `0-49` = "45", `90+` = "93")),
           IncurredAgeBucket = as.numeric(recode(IncurredAgeBucket, `0-49` = "45", `90+` = "93")),
           IssueYear = as.numeric(unlist(lapply(strsplit(incidence$IssueYear, "-"), FUN = function(x) mean(as.numeric(x))))),
           NH_Unk = NH_Orig_Daily_Ben_Bucket == "Unk",
           ALF_Unk = ALF_Orig_Daily_Ben_Bucket == "Unk",
           HHC_Unk = HHC_Orig_Daily_Ben_Bucket == "Unk",
           NH_Orig_Daily_Ben_Bucket = as.numeric(recode(NH_Orig_Daily_Ben_Bucket, `1-99` = "75", `200+` = "225", `100-199` = "150", Unk = "0")),
           ALF_Orig_Daily_Ben_Bucket = as.numeric(recode(ALF_Orig_Daily_Ben_Bucket, `1-99` = "75", `200+` = "225", `100-199` = "150", Unk = "0")),
           HHC_Orig_Daily_Ben_Bucket = as.numeric(recode(HHC_Orig_Daily_Ben_Bucket, `1-99` = "75", `200+` = "225", `100-199` = "150", Unk = "0")),
           NH_Ben_Period_Bucket = as.numeric(recode(NH_Ben_Period_Bucket, `< 1` = "1", `1-2` = "1.5", `3-4` = "3.5", `5+` = "7", Unk = "0", Unlm = "10")),
           ALF_Ben_Period_Bucket = as.numeric(recode(ALF_Ben_Period_Bucket, `< 1` = "1", `1-2` = "1.5", `3-4` = "3.5", `5+` = "7", Unk = "0", Unlm = "10")),
           HHC_Ben_Period_Bucket = as.numeric(recode(HHC_Ben_Period_Bucket, `< 1` = "1", `1-2` = "1.5", `3-4` = "3.5", `5+` = "7", Unk = "0", Unlm = "10")),
           NH_EP_Bucket = as.numeric(recode(NH_EP_Bucket, `90/100` = "95", `> 100` = "120", Unk = "0")),
           ALF_EP_Bucket = as.numeric(recode(ALF_EP_Bucket, `90/100` = "95", `> 100` = "120", Unk = "0")),
           HHC_EP_Bucket = as.numeric(recode(HHC_EP_Bucket, `90/100` = "95", `> 100` = "120", Unk = "0"))
    ) %>%
    mutate(NH_Orig_Daily_Ben_Bucket = if_else(NH_Orig_Daily_Ben_Bucket == 0, mean_ex_0(NH_Orig_Daily_Ben_Bucket), NH_Orig_Daily_Ben_Bucket),
           ALF_Orig_Daily_Ben_Bucket = if_else(ALF_Orig_Daily_Ben_Bucket == 0, mean_ex_0(ALF_Orig_Daily_Ben_Bucket), ALF_Orig_Daily_Ben_Bucket),
           HHC_Orig_Daily_Ben_Bucket = if_else(HHC_Orig_Daily_Ben_Bucket == 0, mean_ex_0(HHC_Orig_Daily_Ben_Bucket), HHC_Orig_Daily_Ben_Bucket),
           
           NH_Ben_Period_Bucket = if_else(NH_Ben_Period_Bucket == 0, mean_ex_0(NH_Ben_Period_Bucket), NH_Ben_Period_Bucket),
           ALF_Ben_Period_Bucket = if_else(ALF_Ben_Period_Bucket == 0, mean_ex_0(ALF_Ben_Period_Bucket), ALF_Ben_Period_Bucket),
           HHC_Ben_Period_Bucket = if_else(HHC_Ben_Period_Bucket == 0, mean_ex_0(HHC_Ben_Period_Bucket), HHC_Ben_Period_Bucket),
           
           NH_EP_Bucket = if_else(NH_EP_Bucket == 0, mean_ex_0(NH_EP_Bucket), NH_EP_Bucket),
           ALF_EP_Bucket = if_else(ALF_EP_Bucket == 0, mean_ex_0(ALF_EP_Bucket), ALF_EP_Bucket),
           HHC_EP_Bucket = if_else(HHC_EP_Bucket == 0, mean_ex_0(HHC_EP_Bucket), HHC_EP_Bucket),
           AdjustedExposure = pmax(ActiveExposure, TotalExposure - 0.5 * (Count_ALF + Count_HHC)),
           AdjustedExposure = ActiveExposure + 0.5 * Count_NH,
           Duration = PolicyYear, 
           Age = IncurredAgeBucket
           ) %>%
    filter(TotalExposure >0, ActiveExposure >0) %>%
    rename(TQ_Status = "TQ Status")
  
  # Convert Strings to Factors
  incidence = incidence %>%
    mutate_if(sapply(incidence, is.character), as.factor)
  
  for (i in 1:ncol(incidence)){
    if (is.factor(incidence[,i][[1]])){
      print(colnames(incidence)[i])
      print(levels(incidence[,i][[1]]))
    } else {
      cat("Values........................")
      cat(colnames(incidence)[i])
      cat(p0("  ",anyNA(incidence[,i][[1]]),"\n"))
    } 
      
  }

  save(incidence, file = p0(data_dir, "incidence_clean.RData"))
  
  "done"
  
  return(incidence)
}

#' loads the cleaned incidence rate database
load_incidence = function(){
  load(file = p0(data_dir, "incidence_clean.RData"))
  return(incidence)
}



#' Loads the incidence tables and splits them into training, validation and test sets.
get_incidence = function(train_val_test = c(0.8, 0.1, 0.1), seed = 1111){
  # Load Database
  incidence = load_incidence()
  N= nrow(incidence)
  
  # Shuffle
  if (!is.null(seed))
    set.seed(1112)
  
  incidence = incidence[sample(1:N, N, replace =F),]
  
  # Split the data into train / validation / test subsets
  train_perc = train_val_test[1]
  val_perc = train_val_test[2]
  test_perc = train_val_test[3]
  
  N_train=floor(train_perc * N)
  N_val=floor(val_perc * N)
  N_test= N - N_train - N_val
  
  index = 1:N
  index_train = sample(x = index, size = N_train, replace = FALSE)
  index_val = sample(x = setdiff(index, index_train), size = N_val, replace = FALSE)
  index_test = setdiff(setdiff(index, index_train), index_val)
  length(index_train) == N_train
  length(index_val) == N_val
  length(index_test) == N_test
  
  # Select sets and exclude unknown files
  train = incidence[index_train, ] %>% filter(!NH_Unk)
  val = incidence[index_val , ] %>% filter(!NH_Unk)
  test = incidence[index_test , ] %>% filter(!NH_Unk)
  rm(incidence)
  
  return(list(train = train, val=val, test=test))
  
}

rmse = function(x){
  sqrt(mean(x^2))
}

calc_expected = function(x, exposure) {
  return(c(exposure * (1- exp(-x))))
}


#' Scale all values except exposure and counts
#' 
scale_data = function(incidence){
  scale_mean = rep(0,23); names(scale_mean) = colnames(incidence$train[1:23])
  scale_sd = rep(1,23); names(scale_sd) = colnames(incidence$train[1:23])
  incidence = map(incidence, as.data.frame)
  for (i in 1:23){
    if (!is.factor(incidence$train[,i])){
      scale_mean[i] = mean(incidence$train[,i])
      scale_sd[i] = sd(incidence$train[,i])
      incidence$train[,i] = (incidence$train[,i] - scale_mean[i]) / scale_sd[i]
      incidence$val[,i] = (incidence$val[,i] - scale_mean[i]) / scale_sd[i]
      incidence$test[,i] = (incidence$test[,i] - scale_mean[i]) / scale_sd[i]
    }
  }
  return(list(incidence = incidence, scale_mean =scale_mean, scale_sd =scale_sd))
  
}



#  Poisson Negative Log Likelihood
poisson_neg_log_lik = function(y_pred, y_true, eps=0){
  mean(y_pred - y_true * log(y_pred+eps))
}

K <- backend()
k_poisson_neg_log_lik = function(y_true, y_pred){
  
  eps=1e-8
  
  return(K$mean(y_pred - y_true * K$log(y_pred+eps)))
}
ExpLayer <- R6::R6Class("CustomLayer",
                        inherit = KerasLayer,
                        public = list(
                          output_dim = NULL,
                          kernel = NULL,
                          initialize = 
                            function(output_dim) {
                              self$output_dim <- output_dim
                            },
                          call = function(x, mask = NULL) {
                            k_exp(x)
                          }
                        ))
layer_exp <- function(object, name = "exponent", trainable = FALSE) {
  create_layer(ExpLayer, object, list(
    output_dim = as.integer(1),
    name = name,
    trainable = trainable
  ))
}


if (F){
  print(incidence, width=Inf)
  
  unique(incidence$NH_Ben_Period_Bucket)
  unique(incidence$IncurredAgeBucket)
  unique(incidence$HHC_Orig_Daily_Ben_Bucket)
  unique(incidence$NH_Ben_Period_Bucket)
  unique(incidence$ALF_Ben_Period_Bucket)
  unique(incidence$HHC_Ben_Period_Bucket)
  unique(incidence$NH_EP_Bucket)
  unique(incidence$ALF_EP_Bucket)
  unique(incidence$HHC_EP_Bucket)
  print()
  
  unique(incidence2$NH_Orig_Daily_Ben_Bucket)
  unique(incidence2$HHC_Orig_Daily_Ben_Bucket)
  unique(incidence2$NH_Ben_Period_Bucket)
  unique(incidence2$ALF_Ben_Period_Bucket)
  unique(incidence2$HHC_Ben_Period_Bucket)
  unique(incidence2$NH_EP_Bucket)
  unique(incidence2$ALF_EP_Bucket)
  unique(incidence2$HHC_EP_Bucket)
}  


