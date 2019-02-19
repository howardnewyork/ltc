## Forward Stepwise Regression

#source('models_01.R')

# Prepare Code
p0=paste0
working_dir = p0(getwd(),"/")
source(p0(working_dir, 'utilities.R'))


library(stringr)

#' Utility Function to Sort the Interaction Terms derived from a formula
#' 
#' @param vars A Vector of variables derived from a formula
#' @return The vector sorted in alphabetical order
sortVars = function(vars){
  
  vars=str_split(vars, pattern = ":") %>%
    map(sort) %>%
    map(function(x) paste(x, collapse = ":")) %>%
    unlist() %>%
    sort()
  vars
}

#' Forward Stepwise Regression
#'
#' @param data A data matrix
#' @param FUN A function which runs the model fitting routine.  The FUN has the following inputs:  data, formula, startResults, ...
#' The FUN must return a list where one of the slots contains a real-valued field with the name defined in the \code{criterion} parameter.
#' @param startFormula The starting formula
#' @param endFormula The ending formula
#' @param criterion The name of the criterion used to compare model fits
#' @param A vector of starting results
#' @param ... any additional parameters required by FUN
#' @return  A list containing all models tested in the forward stepwise algorithm including the best fit model.  The components of the list are "all" and "best"
#' @export
forwardStepwise <- function(data, FUN, startFormula, endFormula, criterion, startResults = NULL, ...){
  
  allResults <- list()
  firstRun = TRUE
  startVars <-  attributes(terms(startFormula))$term.labels
  endVars <- attributes(terms(endFormula))$term.labels

  startVars = sortVars(startVars)
  endVars = sortVars(endVars)
  
  if (any(!startVars %in% endVars)){
    print(startVars[!startVars %in% endVars])
    stop("startFormula is not contained in endFormula")
  }
  if (length(startVars) == length(endVars)){
    warning("No Steps Required")
    results <- FUN(data, startFormula, startResults)
    return(list(all = results, best = results))
  }
  
  cat(" ####****  FORWARD STEPWISE REGRESSION ****####\n")
  
  results <- FUN(data, startFormula, startResults, ...)
  
  allResults <- list(all = list(results), best = results)
  bestCriterion <- results[[criterion]]
  print(format(startFormula))
  print(paste0("Starting Criterion: ", bestCriterion))
  
  testVars <- setdiff(endVars, startVars)
  while (length(testVars) >0) {
    testVarsCurrent <- testVars
    runningFormula <- startFormula
    collectedCriteria = NULL
    runningResults <- list()
    newPar = NULL
    while (length(testVarsCurrent)>0){
      runningFormula <- update(startFormula, as.formula(paste("~ . + ",testVarsCurrent[1])))
      cat(paste("Testing Formula:"))
      cat(paste(format(runningFormula), collapse = ""), "\n")
      results <- NULL
      try(
        results <- FUN(data, runningFormula, startResults, ...)
      )
      if (!is.null(results)){
        runningResults[[length(runningResults) + 1]] <- results
        collectedCriteria <- c(collectedCriteria, results[[criterion]])
        cat(" Criterion: ", results[[criterion]], "\n")
        if (results[[criterion]] < bestCriterion){
          newPar <- testVarsCurrent[1]
          bestCriterion <- results[[criterion]]
          best <- results
          save(best, file =paste0(output_dir,"best_stepwise.RData"))
        }
      } else {
        cat("formula skipped!!\n")
      }
      testVarsCurrent <- testVarsCurrent[-1]
    }
    
    if (is.null(newPar)){
      print("Early Termination")
      return(allResults)
    }
    
    startFormula <- update(startFormula, as.formula(paste("~ . + ",newPar)))
    testVars <- setdiff(testVars, newPar)
    allResults$all[[length(allResults$all)+ 1]] <- best
    allResults$best <- best
    startResults <- best
    bestCriterion <- best[[criterion]]
  }
  allResults
}





#' Neural network model
nn_model = function(data, formula, startResults, ...){
  
  
  
  design = model.matrix(formula, data)
  offset =log (data$ActiveExposure)
  y = data$Count_NH
  
  
  N_design= nrow(design)
  N_predictors = ncol(design)
  
  
  ##########################
  ###  Single Node / Single Layer Neural Network
  ##########################
  
  # The input layers are specified (blue circles in chart) 
  input_offset <- layer_input(shape = c(1), name = "input_offset")
  input_predictors <- layer_input(shape = c(N_predictors), 
                                  name = "input_predictors")
  
  # Single node, single layer (green circle in chart)
  rate_mean = input_predictors %>%
    layer_dense(units =1, activation = 'linear') 
  
  # Prediction is made (purple and orange circles in chart)
  predictions = layer_add(c(input_offset, rate_mean)) %>%
    layer_exp()
  
  # The model is defined linking the inputs to the outputs
  model <- keras_model(inputs = c(input_offset, input_predictors), 
                       outputs = predictions)
  
  # Model is compiled
  model %>% compile(
    loss = 'poisson',
    optimizer = optimizer_adadelta()
  )
  
  #Model is run
  a=Sys.time()
  history <- model %>% fit(
    x = list(input_offset = offset, input_predictors = design),
    y= y,
    epochs = 6, batch_size = 512, 
    validation_split = 0.2,
    verbose = 0
  )
  b=Sys.time(); b-a
  #plot(history) ; history$metrics
  print(history$metrics)
  
  ## Val Log Likelihood:  Score
  preds = model %>% predict(
    x=list(input_offset = offset, 
           input_predictors=design),
    verbose=0, batch_size = 2048*16
  ) %>% c()
  
  pnll_mean = poisson_neg_log_lik(y_pred = preds, y_true = y, eps=1e-8) 
  pnll = pnll_mean * nrow(design)
  K= ncol(design)
  aic = 2* pnll + 2* K
  
  list(formula = formula, pnll = pnll, pnll_mean = pnll_mean, K=K, aic=aic, model = model)
  
}




if (F){
  # Load Database
  incidence = get_incidence()
  incidence$train = incidence$train #[1:200000,]
  
  # Scale all values except exposure and counts
  incidence_scaled = scale_data(incidence)
  incidence = incidence_scaled$incidence; 
  scale_mean = incidence_scaled$scale_mean; 
  scale_sd = incidence_scaled$scale_sd
  rm(incidence_scaled)
  
  formula_soa = Count_NH ~ -1+ offset(log(ActiveExposure)) +
    NH_Ben_Period_Bucket + NH_Orig_Daily_Ben_Bucket:Region + Gender:IncurredAgeBucket+ 
    PolicyYear:Prem_Class+ PolicyYear:Underwriting_Type + PolicyYear:TQ_Status +
    IncurredAgeBucket:Cov_Type_Bucket + IncurredAgeBucket:NH_EP_Bucket  +
    IncurredAgeBucket:Marital_Status
  
  formula_end = Count_NH ~ -1+ offset(log(ActiveExposure)) +
    NH_Ben_Period_Bucket + NH_Orig_Daily_Ben_Bucket:Region + Gender:IncurredAgeBucket+ 
    PolicyYear:Prem_Class+ PolicyYear:Underwriting_Type + PolicyYear:TQ_Status +
    IncurredAgeBucket:Cov_Type_Bucket + IncurredAgeBucket:NH_EP_Bucket  +
    IncurredAgeBucket:Marital_Status + 
    
    (Gender + IssueYear + IncurredAgeBucket  + PolicyYear + Marital_Status + Prem_Class + 
     Underwriting_Type + Cov_Type_Bucket + TQ_Status + NH_Orig_Daily_Ben_Bucket +         
     NH_EP_Bucket  + Region + IssueAgeBucket)   + I(IncurredAgeBucket^2)
  
  formula_winning =   Count_NH ~ NH_Ben_Period_Bucket + Underwriting_Type + I(IncurredAgeBucket^2) + 
    Region + IssueYear + IssueAgeBucket + Marital_Status + Cov_Type_Bucket + 
    Prem_Class + NH_Orig_Daily_Ben_Bucket + IncurredAgeBucket + 
    NH_EP_Bucket + Region:NH_Orig_Daily_Ben_Bucket + IncurredAgeBucket:Gender + 
    PolicyYear:(Prem_Class + TQ_Status) + Underwriting_Type:PolicyYear  + 
    Cov_Type_Bucket:IncurredAgeBucket + IncurredAgeBucket:NH_EP_Bucket + 
    Marital_Status:IncurredAgeBucket + offset(log(ActiveExposure)) - 
    1
  
  formula_winning2 = Count_NH ~ NH_Ben_Period_Bucket + Underwriting_Type + I(IncurredAgeBucket^2) + 
    Region + IssueYear + IssueAgeBucket + Marital_Status + Cov_Type_Bucket + 
    Prem_Class + NH_Orig_Daily_Ben_Bucket + IncurredAgeBucket + 
    NH_EP_Bucket + Region:NH_Orig_Daily_Ben_Bucket + IncurredAgeBucket:Gender + 
    Prem_Class:PolicyYear + PolicyYear:TQ_Status + Underwriting_Type:PolicyYear + 
    Cov_Type_Bucket:IncurredAgeBucket + IncurredAgeBucket:NH_EP_Bucket + 
    Marital_Status:IncurredAgeBucket + IssueYear:NH_EP_Bucket + 
    IssueYear:PolicyYear + NH_Ben_Period_Bucket:Underwriting_Type + 
    Cov_Type_Bucket:TQ_Status + Cov_Type_Bucket:NH_Orig_Daily_Ben_Bucket + 
    NH_Ben_Period_Bucket:Cov_Type_Bucket + Marital_Status:Cov_Type_Bucket + 
    offset(log(ActiveExposure)) - 1
  
  formula_winning3 =   Count_NH ~ NH_Ben_Period_Bucket + Underwriting_Type + I(IncurredAgeBucket^2) + 
    Region + IssueYear + IssueAgeBucket + Marital_Status + Cov_Type_Bucket + 
    Prem_Class + NH_Orig_Daily_Ben_Bucket + IncurredAgeBucket + 
    NH_EP_Bucket + Region:NH_Orig_Daily_Ben_Bucket + IncurredAgeBucket:Gender + 
    Prem_Class:PolicyYear + PolicyYear:TQ_Status + Underwriting_Type:PolicyYear + 
    Cov_Type_Bucket:IncurredAgeBucket + IncurredAgeBucket:NH_EP_Bucket + 
    Marital_Status:IncurredAgeBucket + IssueYear:NH_EP_Bucket + 
    IssueYear:PolicyYear + NH_Ben_Period_Bucket:Underwriting_Type + 
    Cov_Type_Bucket:TQ_Status + Cov_Type_Bucket:NH_Orig_Daily_Ben_Bucket + 
    NH_Ben_Period_Bucket:Cov_Type_Bucket + Marital_Status:Cov_Type_Bucket + 
    NH_Ben_Period_Bucket:PolicyYear + IncurredAgeBucket:PolicyYear + 
    NH_Ben_Period_Bucket:IncurredAgeBucket + offset(log(ActiveExposure)) - 
    1
  
  formula_winning4 =  Count_NH ~ NH_Ben_Period_Bucket + Underwriting_Type + I(IncurredAgeBucket^2) + 
    Region + IssueYear + IssueAgeBucket + Marital_Status + Cov_Type_Bucket + 
    Prem_Class + NH_Orig_Daily_Ben_Bucket + IncurredAgeBucket + 
    NH_EP_Bucket + Region:NH_Orig_Daily_Ben_Bucket + IncurredAgeBucket:Gender + 
    Prem_Class:PolicyYear + PolicyYear:TQ_Status + Underwriting_Type:PolicyYear + 
    Cov_Type_Bucket:IncurredAgeBucket + IncurredAgeBucket:NH_EP_Bucket + 
    Marital_Status:IncurredAgeBucket + IssueYear:NH_EP_Bucket + 
    IssueYear:PolicyYear + NH_Ben_Period_Bucket:Underwriting_Type + 
    Cov_Type_Bucket:TQ_Status + Cov_Type_Bucket:NH_Orig_Daily_Ben_Bucket + 
    NH_Ben_Period_Bucket:Cov_Type_Bucket + Marital_Status:Cov_Type_Bucket + 
    Region:Prem_Class + Underwriting_Type:IncurredAgeBucket + 
    Underwriting_Type:Region + offset(log(ActiveExposure)) - 
    1
  
  formula_ultimate =   Count_NH ~      ( PolicyYear + NH_Ben_Period_Bucket + Underwriting_Type +
    Region + IssueYear  + Marital_Status + Cov_Type_Bucket + 
    Prem_Class + NH_Orig_Daily_Ben_Bucket + IncurredAgeBucket + 
    NH_EP_Bucket + Gender + TQ_Status+ 
    Marital_Status) ^2  + IssueAgeBucket + I(IncurredAgeBucket^2) + offset(log(ActiveExposure)) -
    1
  

  # formula_end = Count_NH ~ -1 + offset(log(ActiveExposure)) + 
  #   (Gender + IssueYear + IncurredAgeBucket  + PolicyYear + Marital_Status + Prem_Class + 
  #      Underwriting_Type + Cov_Type_Bucket + TQ_Status + NH_Orig_Daily_Ben_Bucket + NH_Ben_Period_Bucket +        
  #      NH_EP_Bucket  + Region + IssueAgeBucket)^2   + I(IncurredAgeBucket^2) + Infl_Rider 
  # 
  # formula_end = Count_NH ~ -1+ offset(log(ActiveExposure)) +
  #   NH_Ben_Period_Bucket + NH_Orig_Daily_Ben_Bucket:Region + 
  #   PolicyYear:Prem_Class+ PolicyYear:Underwriting_Type + PolicyYear:TQ_Status +
  #   IncurredAgeBucket:Cov_Type_Bucket + IncurredAgeBucket:NH_EP_Bucket + IncurredAgeBucket:Gender +
  #   IncurredAgeBucket:Marital_Status + IssueYear + Marital_Status + Prem_Class
  
  ##################################################333
  ################33  First Round
  a=Sys.time()
  ans = forwardStepwise(data = incidence$train, startFormula = formula_soa, endFormula = formula_end, criterion = "aic", FUN = nn_model)
  a=Sys.time(); b-a

  
  ans$best
  model_soa = nn_model(incidence$train, formula = formula_soa)
  model_soa$model
  
  y_true = incidence$val$Count_NH
  
  preds = model_soa$model %>% predict(
    x=list(input_offset = log (incidence$val$ActiveExposure), 
           input_predictors=model.matrix(formula_soa, incidence$val)),
    verbose=0, batch_size = 2048*16
  ) %>% c()
  
  pnll_mean = poisson_neg_log_lik(y_pred = preds, y_true = y_true, eps=1e-8) 
  pnll_mean
  
  #model_best = nn_model(incidence$train, formula = best$formula)
  model_best = nn_model(incidence$train, formula = ans$best$formula)
  model_best
  #save(model_best, file = paste0(output_dir, "stepwise_model_best.RData"))
  
  # PNLL Validation Set
  preds = model_best$model %>% predict(
    x=list(input_offset = log (incidence$val$ActiveExposure), 
           input_predictors=model.matrix(model_best$formula, incidence$val)),
    verbose=0, batch_size = 2048*16
  ) %>% c()
  
  pnll_mean = poisson_neg_log_lik(y_pred = preds, y_true = y_true, eps=1e-8) 
  pnll_mean

  
  ##################################################333
  ################ Second  Round
  a=Sys.time()
  ans = forwardStepwise(data = incidence$train, startFormula = formula_winning, endFormula = formula_ultimate, criterion = "aic", FUN = nn_model)
  a=Sys.time(); b-a
  
  ################ Third  Round
  a=Sys.time()
  ans = forwardStepwise(data = incidence$train, startFormula = formula_winning2, endFormula = formula_ultimate, criterion = "aic", FUN = nn_model)
  a=Sys.time(); b-a
  
  ################ Fourth  Round
  a=Sys.time()
  ans = forwardStepwise(data = incidence$train, startFormula = formula_winning3, endFormula = formula_ultimate, criterion = "aic", FUN = nn_model)
  a=Sys.time(); b-a
  
  ################ Fifth  Round:  Redundant?
  a=Sys.time()
  ans = forwardStepwise(data = incidence$train, startFormula = formula_winning4, endFormula = formula_ultimate, criterion = "aic", FUN = nn_model)
  a=Sys.time(); b-a
  
  save(ans, file = paste0(output_dir,"forwardStepwise_results.RData"))
  #load(file = paste0(output_dir,"forwardStepwise_results.RData"))

  ans$best
  
  model_soa = nn_model(incidence$train, formula = formula_soa)
  model_soa$model
  
  y_true = incidence$val$Count_NH
  
  preds = model_soa$model %>% predict(
    x=list(input_offset = log (incidence$val$ActiveExposure), 
           input_predictors=model.matrix(formula_soa, incidence$val)),
    verbose=0, batch_size = 2048*16
  ) %>% c()
  
  pnll_mean = poisson_neg_log_lik(y_pred = preds, y_true = y_true, eps=1e-8) 
  pnll_mean
  
  #model_best = nn_model(incidence$train, formula = best$formula)
  model_best = nn_model(incidence$train, formula = ans$best$formula)
  model_best
  #save(model_best, file = paste0(output_dir, "stepwise_model_best.RData"))
  
  preds = model_best$model %>% predict(
    x=list(input_offset = log (incidence$val$ActiveExposure), 
           input_predictors=model.matrix(model_best$formula, incidence$val)),
    verbose=0, batch_size = 2048*16
  ) %>% c()
  
  pnll_mean = poisson_neg_log_lik(y_pred = preds, y_true = y_true, eps=1e-8) 
  pnll_mean
  
  # PNLL Test Set
  preds_test = model_best$model %>% predict(
    x=list(input_offset = log (incidence$test$ActiveExposure), 
           input_predictors=model.matrix(model_best$formula, incidence$test)),
    verbose=0, batch_size = 2048*16
  ) %>% c()
  y_true_test = incidence$test$Count_NH
  
  pnll_mean = poisson_neg_log_lik(y_pred = preds_test, y_true = y_true_test, eps=1e-8) 
  pnll_mean
  
  #plots
  
  incidence$val$preds=c(preds)
  
  # Produce Plots
  res = incidence$val %>%
    group_by(Gender,Age) %>%
    summarize(AdjustedExposure = sum(AdjustedExposure), Count_NH = sum(Count_NH),preds = sum(preds)) %>%
    mutate(ae = Count_NH / preds, actual = Count_NH / AdjustedExposure, 
           expected = preds / AdjustedExposure)
  
  rate_plot = ggplot(res) + geom_line(aes(Age , actual, color="Actual"))+
    geom_line(aes(Age , expected, color="Predicted")) + 
    facet_wrap(~Gender) + labs(title = "GLM Model: Actual vs. Expected Incidence Rates", colour="Source")
  
  res = incidence$val %>%
    group_by(Duration) %>%
    summarize(AdjustedExposure = sum(AdjustedExposure), Count_NH = sum(Count_NH), 
              preds =sum(preds)) %>%
    mutate(Actual = Count_NH / AdjustedExposure, Predicted = preds / AdjustedExposure, 
           AE = Actual/Predicted)
  total_ae = sum(res$Count_NH) / sum(res$preds)
  res$mean_ae = total_ae
  duration_plot = ggplot(res) + geom_line(aes(Duration , Actual, color="Actual"))+ 
    geom_line(aes(Duration , Predicted, color="Predicted")) +
    labs(title = "Actual vs. Predicted Incidence Rates by Duration", y= "Incidence Rate", x="Duration", colour="Source") +
    coord_cartesian(xlim = c(0,20), ylim=c(0,.05))
  ae_plot = ggplot(res) + geom_line(aes(Duration , AE, color="Actual/Predicted"))+
    geom_line(aes(Duration,mean_ae, color="Mean AE")) + 
    labs(title = "Ratio of Actual / Predicted Incidence Rates by Duration", x= "Duration") + 
    coord_cartesian(xlim = c(0,20), ylim = c(0.5,1.2))
  
  rate_plot
  
  duration_plot
  ae_plot
    
}




