# Clear existing data and graphics
rm(list=ls())
#graphics.off()


# 1. Load Libraries ----
library(tidyverse)
library(speff2trial)  # for 1996 HIV RCT data set
library(rvinecopulib)  # for r-vine copula implementation
library(e1071)  # for r-vine copula implementation
library(caret)  # for r-vine copula implementation and to train ML models using CV
library(EnvStats)  # for r-vine copula implementation
library(truncnorm)  # for r-vine copula implementation
library(dgof)  # for Kolmogorov-Smirnov tests
library(caTools)  # to easily split data into train/test sets
library(xgboost)  # for XGBoost model
library(RANN)  # to impute missing values using KNN
library(class)  # to train KNN model
library(arf)  # for adversarial random forest (arf) implementation


# 2. Import Data ----
data(ACTG175)

# Make sure each column is correct type (numeric, factor)
real <- ACTG175 %>%
  dplyr::select(c(pidnum, age, wtkg, hemo, homo, drugs, karnof, oprior, z30, preanti, 
                  race, gender, strat, symptom, cd40, arms, cd420, cd496, cens)) %>%
  mutate(strat = strat - 1) %>%
  mutate(age = as.numeric(age),
         wtkg = as.numeric(wtkg),
         hemo = as.factor(hemo),
         homo = as.factor(homo),
         drugs = as.factor(drugs),
         karnof = as.factor(karnof),
         oprior = as.factor(oprior),
         z30 = as.factor(z30),
         preanti = as.numeric(preanti),
         race = as.factor(race),
         gender = as.factor(gender),
         strat = as.factor(strat),
         symptom = as.factor(symptom),
         cd40 = as.numeric(cd40),
         cd420 = as.numeric(cd420),
         cd496 = as.numeric(cd496),
         arms = as.factor(arms),
         cens = as.factor(cens)) %>%
  dplyr::select(c(pidnum, age, wtkg, hemo, homo, drugs, karnof, oprior, z30, preanti, 
                  race, gender, strat, symptom, cd40, arms, cd420, cd496, cens))


# 3. Simulation Setup ----

# Load the following functions.

### R-Vine Copula Baseline: ####

# Function to transform uniform distribution to original scale according to empirical distribution
PseudoObsInverse <- function(DataBaseline, UniformData) {
  # DataBaseline is the matrix of covariates (original data)
  # UniformData is the matrix of uniform data to be transformed to original scale
  PsInverse <- list()
  for (j in 1:ncol(DataBaseline)){ 
    ecdfj <- ecdf(DataBaseline[, j])  # empirical cdf
    ECDFvar <- get("x", environment(ecdfj))
    ECDFjump <- get("y", environment(ecdfj))
    PsInverse[[j]] <- stepfun(ECDFjump[-length(ECDFjump)], ECDFvar)  # define step function
  } 
  ScaledData <- matrix(0, nrow(UniformData), ncol(UniformData))
  for (j in 1:ncol(UniformData)){ ScaledData[, j] <- PsInverse[[j]](UniformData[, j]) }
  ScaledData <- as.data.frame(ScaledData)
  # output
  return(ScaledData)
}

# Function to estimate the 'R-vine copula' model of the baseline data (covariates)
Estimation_Copule <- function(DataBaseline)   {
  # DataBaseline is the matrix of covariates (original data)
  
  ## Data  preparation
  # Transformation of continuous variables (in original scale) into uniform distribution variables  
  # Pseudo-observations compute using rvinecopulib package   
  U_cont <- pseudo_obs(DataBaseline[, 11:14])  # columns 11:15 are the continuous variables
  
  # Distribution of the discrete variables
  disc_1 <- as.integer(DataBaseline[, 1])  # binary variable should have levels 0, 1
  disc_2 <- as.integer(DataBaseline[, 2])  # categorical variable should have levels 0, 1, 2, etc.
  disc_3 <- as.integer(DataBaseline[, 3])
  disc_4 <- as.integer(DataBaseline[, 4])
  disc_5 <- as.integer(DataBaseline[, 5])
  disc_6 <- as.integer(DataBaseline[, 6])
  disc_7 <- as.integer(DataBaseline[, 7])
  disc_8 <- as.integer(DataBaseline[, 8])
  disc_9 <- as.integer(DataBaseline[, 9])
  disc_10 <- as.integer(DataBaseline[, 10]) # karnofksy score
  freq_disc1 <- prop.table(table(DataBaseline[, 1]))
  freq_disc2 <- prop.table(table(DataBaseline[, 2]))
  freq_disc3 <- prop.table(table(DataBaseline[, 3]))
  freq_disc4 <- prop.table(table(DataBaseline[, 4]))
  freq_disc5 <- prop.table(table(DataBaseline[, 5]))
  freq_disc6 <- prop.table(table(DataBaseline[, 6]))
  freq_disc7 <- prop.table(table(DataBaseline[, 7]))
  freq_disc8 <- prop.table(table(DataBaseline[, 8]))
  freq_disc9 <- prop.table(table(DataBaseline[, 9]))
  freq_disc10 <- prop.table(table(DataBaseline[, 10]))
  
  # Preparation of the discrete variables needed to use 'vinecop' function for mixed data (package rvinecopulib)
  Freq_disc_t1 <- cbind(pdiscrete(disc_1 + 1, freq_disc1), pdiscrete(disc_2 + 1, freq_disc2),
                        pdiscrete(disc_3 + 1, freq_disc3), pdiscrete(disc_4 + 1, freq_disc4),
                        pdiscrete(disc_5 + 1, freq_disc5), pdiscrete(disc_6 + 1, freq_disc6),
                        pdiscrete(disc_7 + 1, freq_disc7), pdiscrete(disc_8 + 1, freq_disc8),
                        pdiscrete(disc_9 + 1, freq_disc9), pdiscrete(disc_10 + 1, freq_disc10))
  Freq_disc_t0 <- cbind(pdiscrete(disc_1, freq_disc1), pdiscrete(disc_2, freq_disc2),
                        pdiscrete(disc_3, freq_disc3), pdiscrete(disc_4, freq_disc4),
                        pdiscrete(disc_5, freq_disc5), pdiscrete(disc_6, freq_disc6),
                        pdiscrete(disc_7, freq_disc7), pdiscrete(disc_8, freq_disc8),
                        pdiscrete(disc_9, freq_disc9), pdiscrete(disc_10, freq_disc10))
  U_mixte <- cbind(Freq_disc_t1, U_cont, Freq_disc_t0) # need Freq_disc_t0 to handle discrete obs (check details of rdocumentation)
  #density: ddiscrete(x+1, freq)
  #distribution function: pdiscrete(x+1, freq)
  #quantile function: qdiscrete(u[, 1], freq) - 1
  
  ## Estimation of the R-vine model for mixed data using rvinecopulib package
  fit_DataDriven <- vinecop(U_mixte, var_types = c(rep("d", 10), rep("c", 4)))
  #summary(fit_DataDriven)
  #plot(fit_DataDriven)
  #contour(fit_DataDriven)
  
  # Definition of the R-vine distribution 
  Fit_dist <- vinecop_dist(fit_DataDriven$pair_copulas, fit_DataDriven$structure, fit_DataDriven$var_types)
  
  ## Output
  return(Fit_dist)
} 

# Function to generate a sample according to the estimated R-vine model and baseline data (covariates)
Simulation_Copule <- function(N, Fit_dist, DataBaseline)   {
  # N is number of observations to be generated (sample size)
  # Fit_dist is the R-vine model estimated on original data  
  # DataBaseline is the matrix of covariates (original data)
  
  # Generation of a uniform sample using the estimated R-vine copula distribution
  U_Simu <- rvinecop(N, Fit_dist)
  # Transform uniform distribution to original scale according to empirical distribution  
  # (reverse function for 'pseudo_obs' one)
  # This function is defined above
  VGenCop <- PseudoObsInverse(DataBaseline, U_Simu)
  # Data preparation
  for (i in 1:10){ VGenCop[,i] = as.factor(VGenCop[, i]) }  # discrete vars
  for (i in 11:14){ VGenCop[,i] = as.numeric(as.character(VGenCop[, i])) }  # continuous vars
  colnames(VGenCop) <- colnames(DataBaseline)
  levels(VGenCop[, 1]) <- c('0', '1')  # levels in original data 
  levels(VGenCop[, 2]) <- c('0', '1') 
  levels(VGenCop[, 3]) <- c('0', '1') 
  levels(VGenCop[, 4]) <- c('0', '1') 
  levels(VGenCop[, 5]) <- c('0', '1') 
  levels(VGenCop[, 6]) <- c('0', '1') 
  levels(VGenCop[, 7]) <- c('0', '1') 
  levels(VGenCop[, 8]) <- c('0', '1', '2') 
  levels(VGenCop[, 9]) <- c('0', '1') 
  levels(VGenCop[, 10]) <- c('70', '80', '90', '100')  # karnofsky
  
  ## Output
  return(VGenCop)
}   

### Execution Models: ####

# Function to generate random treatment assignment
Simulation_Treatment <- function(N, Treatment_Arms, Probabilities) {
  # N = number of trials = number of patients = number of rows in virtual baseline cohort
  # Treatment_Arms = all possible treatment arms = vector of possible treatments e.g., c(0, 1, 2, 3)
  # Probabilities = probabilities for each outcome = equal chance between all treatment options
  
  sample_txsums = rmultinom(n = N, size = 1, prob = Probabilities) %>% t() %>% colSums()
  tx_assign = rep(Treatment_Arms, sample_txsums %>% t())
  tx_assign_random = sample(tx_assign)
  
  return(tx_assign_random)
}

# Function to generate cd420 based on previous learned model and new data (simulated covariates) 
Simulation_PostRandom_Wk20 <- function(Model, Covariates_synth, Tx_synth) {
  # Model is the prediction model used to predict the post-randomization variable at week 20 (cd420)
  # Covariates_synth are the synthetic covariates used to generate synthetic cd420
  # Tx_synth are the synthetic treatment assignments used to generate synthetic cd420
  
  # Prediction
  cd420_predict <- predict(Model, newdata = cbind(Covariates_synth, arms = Tx_synth))
  
  # Residuals
  cd420_resid <- residuals(Model)
  
  # Initialize vector to store synthetic cd420 values
  cd420_synthetic = rep(NA, nrow(Covariates_synth))
  
  for (i in 1:length(cd420_predict)) {
    # Get all possible values of pred + resid for the i'th observation
    pred_resid_sums <- cd420_predict[i] + cd420_resid
    
    # All pred + resid >= 0
    pred_resid_sums_pos <- pred_resid_sums[pred_resid_sums >= 0]
    
    # Randomly sample from non-negative sum values
    sample_val = sample(pred_resid_sums_pos, size = 1)
    
    # Save sample
    cd420_synthetic[i] <- sample_val
  }
  return(cd420_synthetic)
}

# Function to generate cd496 based on previous learned model and new data (simulated covariates)
Simulation_PostRandom_Wk96 <- function(Model, Covariates_synth, Tx_synth, PostRandom_Wk20) {
  # Model is the prediction model used to predict the post-randomization variable at week 96 (cd496)
  # Covariates_synth are the synthetic covariates used to generate synthetic cd496
  # Tx_synth are the synthetic treatment assignments used to generate synthetic cd496
  # PostRandom_Wk20 are the values for synthetic post randomization variable at week 20 (cd420)
  
  # Prediction
  cd496_predict <- predict(Model, newdata = cbind(Covariates_synth, 
                                                  arms = Tx_synth, 
                                                  cd420 = PostRandom_Wk20))
  
  # Residuals
  cd496_resid <- residuals(Model)
  
  # Initialize vector to store synthetic cd496 values
  cd496_synthetic = rep(NA, nrow(Covariates_synth))
  
  for (i in 1:length(cd496_predict)) {
    # Get all possible values of pred + resid for the i'th observation
    pred_resid_sums <- cd496_predict[i] + cd496_resid
    
    # All pred + resid >= 0
    pred_resid_sums_pos <- pred_resid_sums[pred_resid_sums >= 0]
    
    # Randomly sample from non-negative sum values
    sample_val = sample(pred_resid_sums_pos, size = 1)
    
    # Save sample
    cd496_synthetic[i] <- sample_val
  }
  return(cd496_synthetic)
}

# Function to predict the outcome variable based on previous learned model and new data (simulated covariates)
Simulation_DataOutcome <- function(Model, Covariates_synth, Tx_synth, PostRandom_Wk20, PostRandom_Wk96) {
  # Model is the prediction model used to predict the outcome
  # Covariates_synth are the synthetically-generated covariates used to predict the outcome
  outcome_synthetic_prob <- predict.glm(Model,
                                        newdata = cbind(Covariates_synth,
                                                        arms = Tx_synth,
                                                        cd420 = PostRandom_Wk20,
                                                        cd496 = PostRandom_Wk96),
                                        type = "response")
  
  # Random sample of Unif(0,1)
  unif_samples <- runif(n = nrow(Covariates_synth), min = 0, max = 1)
  
  # Initialize vector to store final synthetic outcome values
  outcome_synthetic <- rep(NA, nrow(Covariates_synth))
  
  for (i in 1:length(outcome_synthetic)) {
    if (unif_samples[i] < outcome_synthetic_prob[i]) {
      outcome_synthetic[i] <- 1
    }
    else {
      outcome_synthetic[i] <- 0
    }
  }
  return(outcome_synthetic)
}

### Generate synthetic data: ####

# Function to generate entire data set using ARF Sequential framework
ARF_data_generation <- function(real_data, random_seed, n_obs) {
  # Input: real_data is the real data dataframe, 
  #        random_seed is a number for the random seed,
  #        n_obs is the number of observations (i.e., rows) to generate
  # Ouput: dataframe of synthetic data
  
  # Definition of the outcome vector 
  Outcome <- real_data %>% dplyr::select(cens) %>% unlist() %>% as.factor()
  
  # Definition of the matrix of discrete covariates (at baseline)
  Cov_Discrete <- real_data %>% 
    dplyr::select(c(hemo, homo, drugs, oprior, z30, race, gender, strat, symptom, karnof)) %>%
    # mutate(strat = strat - 1) %>%
    lapply(., as.factor) %>%
    as.data.frame()
  
  # Definition of the matrix of continuous covariates (at baseline)
  Cov_Cont <- real_data %>%
    dplyr::select(c(age, wtkg, preanti, cd40))
  
  # Definition of the matrix of covariates (at baseline)
  Cov <- c(Cov_Discrete, Cov_Cont) %>% as.data.frame()
  
  # Definition of treatment assignment vector 
  Tx <- real_data %>% dplyr::select(arms) %>% unlist() %>% as.factor()
  
  # Definition of the matrix of post-randomization variables
  Post_Random <- real_data %>%
    dplyr::select(c(cd420, cd496))
  
  # Definition of the considered data
  data_allvar <- cbind(Cov, Tx, Post_Random, Outcome)
  
  # Definition of baseline data and outcome
  db <- cbind(Cov, Outcome)
  
  # Set random seed
  set.seed(random_seed)
  
  # Train ARF
  arf <- adversarial_rf(Cov)
  
  # Estimate leaf parameters
  psi <- forde(arf, Cov)
  
  # Generate synthetic samples 
  DataSimu <- forge(psi, n_synth = n_obs)
  
  # Simulation of synthetic treatment allocation
  TxSimu <- Simulation_Treatment(N = n_obs, Treatment_Arms = c(0, 1, 2, 3), 
                                 Probabilities = c(0.25, 0.25, 0.25, 0.25))
  
  # Simulation of post-randomization variable at week 20 (cd420)
  # Model to predict cd420
  # The prediction model is learned based on original data using a linear regression model
  cd4wk20_data_learn <- cbind(db, arms = Tx, cd420 = Post_Random$cd420)
  
  cd420_model <- lm(cd420 ~ age + wtkg + as.factor(hemo) + as.factor(homo) + as.factor(drugs) +
                      as.factor(karnof) + as.factor(oprior) + as.factor(z30) + preanti + as.factor(race) +
                      as.factor(gender) + as.factor(strat) + as.factor(symptom) + cd40 + 
                      as.factor(arms),
                    data = cd4wk20_data_learn)
  
  cd420Simu <- Simulation_PostRandom_Wk20(Model = cd420_model, Covariates_synth = DataSimu, 
                                          Tx_synth = TxSimu)
  
  # Simulation of post-randomization variable at week 96 (cd496)
  # Model to predict cd496
  # The prediction model is learned based on original data using a linear regression model
  cd4wk96_data_learn <- cbind(db, arms = Tx, cd420 = Post_Random$cd420, cd496 = Post_Random$cd496)
  
  cd496_model <- lm(cd496 ~ age + wtkg + as.factor(hemo) + as.factor(homo) + as.factor(drugs) +
                      as.factor(karnof) + as.factor(oprior) + as.factor(z30) + preanti + as.factor(race) +
                      as.factor(gender) + as.factor(strat) + as.factor(symptom) + cd40 + cd420 +
                      as.factor(arms),
                    data = cd4wk96_data_learn)
  
  cd496Simu <- Simulation_PostRandom_Wk96(Model = cd496_model, Covariates_synth = DataSimu,
                                          Tx_synth = TxSimu, PostRandom_Wk20 = cd420Simu)
  
  # Simulation of the outcome for each virtual patients (using the RF prediction model)
  # Model to predict the outcome variable
  # The prediction model is learned based on original data using a logistic regression model 
  outcome_data_learn <- cbind(db[, !(names(db) %in% c("Outcome"))], 
                              arms = Tx, 
                              cd420 = Post_Random$cd420, 
                              cd496 = Post_Random$cd496,
                              cens = as.factor(Outcome))
  
  outcome_model <- glm(cens ~ age + wtkg + as.factor(hemo) + as.factor(homo) + 
                         as.factor(drugs) + as.factor(karnof) + oprior + as.factor(z30) + preanti + 
                         as.factor(race) + as.factor(gender) + as.factor(strat) + 
                         as.factor(symptom) + cd40 + cd420 + cd496 + as.factor(arms), 
                       data = outcome_data_learn, family = binomial())
  
  OutcomeSimu <- Simulation_DataOutcome(Model = outcome_model, Covariates_synth = DataSimu,
                                        Tx_synth = TxSimu, PostRandom_Wk20 = cd420Simu,
                                        PostRandom_Wk96 = cd496Simu)
  
  # Final table of synthetic data generated via R-vine copula + execution models
  data_synthetic_arf_execmod <- cbind(pidnum = c(1:nrow(DataSimu)),
                                      DataSimu,
                                      arms = as.factor(TxSimu),
                                      cd420 = cd420Simu,
                                      cd496 = cd496Simu,
                                      cens = as.factor(OutcomeSimu))
  
  return(data_synthetic_arf_execmod)
}

# Function to generate entire data set using R-Vine Copula Sequential framework
RVine_data_generation <- function(real_data, random_seed, n_obs) {
  # Input: real_data is the real data dataframe, 
  #        random_seed is a number for the random seed,
  #        n_obs is the number of observations (i.e., rows) to generate
  # Ouput: dataframe of synthetic data
  
  # Definition of the outcome vector 
  Outcome <- real_data %>% dplyr::select(cens) %>% unlist() %>% as.factor()
  
  # Definition of the matrix of discrete covariates (at baseline)
  Cov_Discrete <- real_data %>% 
    dplyr::select(c(hemo, homo, drugs, oprior, z30, race, gender, strat, symptom, karnof)) %>%
    # mutate(strat = strat - 1) %>%
    lapply(., as.factor) %>%
    as.data.frame()
  
  # Definition of the matrix of continuous covariates (at baseline)
  Cov_Cont <- real_data %>%
    dplyr::select(c(age, wtkg, preanti, cd40))
  
  # Definition of the matrix of covariates (at baseline)
  Cov <- c(Cov_Discrete, Cov_Cont) %>% as.data.frame()
  
  # Definition of treatment assignment vector 
  Tx <- real_data %>% dplyr::select(arms) %>% unlist() %>% as.factor()
  
  # Definition of the matrix of post-randomization variables
  Post_Random <- real_data %>%
    dplyr::select(c(cd420, cd496))
  
  # Definition of the considered data
  data_allvar <- cbind(Cov, Tx, Post_Random, Outcome)
  
  # Definition of baseline data and outcome
  db <- cbind(Cov, Outcome)
  
  ## Marginal distribution of the covariates are estimated using empirical estimator 
  set.seed(random_seed)
  
  # Estimation of the R-vine model based on original data
  Rvine_dist <- Estimation_Copule(Cov)
  
  # Simulation of virtual patients based on the R-vine model and empirical distribution (of the original data)
  DataSimu <- Simulation_Copule(n_obs, Rvine_dist, Cov)  # n_obs = number of rows in original data
  
  # Simulation of synthetic treatment allocation
  TxSimu <- Simulation_Treatment(N = n_obs, Treatment_Arms = c(0, 1, 2, 3), 
                                 Probabilities = c(0.25, 0.25, 0.25, 0.25))
  
  # Simulation of post-randomization variable at week 20 (cd420)
  # Model to predict cd420
  # The prediction model is learned based on original data using a linear regression model
  cd4wk20_data_learn <- cbind(db, arms = Tx, cd420 = Post_Random$cd420)
  
  cd420_model <- lm(cd420 ~ age + wtkg + as.factor(hemo) + as.factor(homo) + as.factor(drugs) +
                      as.factor(karnof) + as.factor(oprior) + as.factor(z30) + preanti + as.factor(race) +
                      as.factor(gender) + as.factor(strat) + as.factor(symptom) + cd40 + 
                      as.factor(arms),
                    data = cd4wk20_data_learn)
  
  cd420Simu <- Simulation_PostRandom_Wk20(Model = cd420_model, Covariates_synth = DataSimu, 
                                          Tx_synth = TxSimu)
  
  # Simulation of post-randomization variable at week 96 (cd496)
  # Model to predict cd496
  # The prediction model is learned based on original data using a linear regression model
  cd4wk96_data_learn <- cbind(db, arms = Tx, cd420 = Post_Random$cd420, cd496 = Post_Random$cd496)
  
  cd496_model <- lm(cd496 ~ age + wtkg + as.factor(hemo) + as.factor(homo) + as.factor(drugs) +
                      as.factor(karnof) + as.factor(oprior) + as.factor(z30) + preanti + as.factor(race) +
                      as.factor(gender) + as.factor(strat) + as.factor(symptom) + cd40 + cd420 +
                      as.factor(arms),
                    data = cd4wk96_data_learn)
  
  cd496Simu <- Simulation_PostRandom_Wk96(Model = cd496_model, Covariates_synth = DataSimu,
                                          Tx_synth = TxSimu, PostRandom_Wk20 = cd420Simu)
  
  # Simulation of the outcome for each virtual patients (using the RF prediction model)
  # Model to predict the outcome variable
  # The prediction model is learned based on original data using a logistic regression model 
  outcome_data_learn <- cbind(db[, !(names(db) %in% c("Outcome"))], 
                              arms = Tx, 
                              cd420 = Post_Random$cd420, 
                              cd496 = Post_Random$cd496,
                              cens = as.factor(Outcome))
  
  outcome_model <- glm(cens ~ age + wtkg + as.factor(hemo) + as.factor(homo) + 
                         as.factor(drugs) + as.factor(karnof) + oprior + as.factor(z30) + preanti + 
                         as.factor(race) + as.factor(gender) + as.factor(strat) + 
                         as.factor(symptom) + cd40 + cd420 + cd496 + as.factor(arms), 
                       data = outcome_data_learn, family = binomial())
  
  OutcomeSimu <- Simulation_DataOutcome(Model = outcome_model, Covariates_synth = DataSimu,
                                        Tx_synth = TxSimu, PostRandom_Wk20 = cd420Simu,
                                        PostRandom_Wk96 = cd496Simu)
  
  # Final table of synthetic data generated via R-vine copula + execution models
  data_synthetic_rvine_execmod <- cbind(pidnum = c(1:nrow(DataSimu)), 
                                        DataSimu, 
                                        arms = as.factor(TxSimu),
                                        cd420 = cd420Simu, 
                                        cd496 = cd496Simu, 
                                        cens = as.factor(OutcomeSimu))
  
  return(data_synthetic_rvine_execmod)
}

### Evaluate synthetic data (i.e., calculate metrics): ####

# Function for calculating 1-KS for all variables, for a given dataset (i.e., method)
ks.stat <- function(Synthetic_Data, Real_Data) {
  # This function takes as input a synthetic data set and the real data set
  # and returns a dataframe with the 1-KS value and the variable name.
  # This function uses the ks.test() function from the dgof package
  
  # Throw error if column names of Synthetic_Data and Real_Data are not the same
  if(length(intersect(colnames(Synthetic_Data), colnames(Real_Data))) != ncol(Synthetic_Data)) {
    stop("Column names should be the same in synthetic and real data frames.")
  }
  
  # Initialize dataframe to store 1-KS values
  vals <- matrix(data = NA, nrow = ncol(Synthetic_Data), ncol = 2) %>% as.data.frame()
  colnames(vals) <- c("Variable", "KS Statistic")
  
  # Perform KS test, then store 1 - KS statistic for each variable
  for(i in 1:ncol(Synthetic_Data)) {
    # i'th variable
    col_name <- names(Synthetic_Data)[i]
    
    # Save variable name
    vals[i, 1] <- col_name
    
    # Perform KS test for i'th variable
    ks_test <- ks.test(Synthetic_Data[, col_name], Real_Data[, col_name], alternative = "two.sided")
    
    # KS statistic
    ks_stat <- ks_test$statistic %>% unname()
    
    # Save 1-KS statistic
    vals[i, 2] <- 1 - ks_stat
  }
  
  # Return df of 1 - KS statistics
  return(vals)
  
}

# Calculate this by hand:
# calculate difference between synthetic and real proportions of each category of a given variable,
# sum the absolute value of all differences,
# then divide the sum by 2.
# Note: this function works for both binary variables and categorical variables with more than 2 levels.
tvd.calc <- function(Synthetic_Data, Real_Data, Var) {
  # Synthetic_Data is a dataframe containing synthetic data
  # Real_Data is a dataframe containing real data
  # Var is a character string of the variable name
  
  synthetic_prop <- table(Synthetic_Data[, Var])/nrow(Synthetic_Data)
  real_prop <- table(Real_Data[, Var])/nrow(Real_Data)
  tvd <- 0.5 * sum(abs(synthetic_prop - real_prop))
  
  return(tvd)
}

# Function to return all tvd stats for all discrete variables in a given data set
tvd.stat <- function(Synthetic_Data, Real_Data) {
  # This function takes as input a synthetic data set and the real data set
  # and returns a dataframe with the 1 - TVD value and the variable name.
  # This function uses the previously-declared function, tvd.calc
  
  # Throw error if column names of Synthetic_Data and Real_Data are not the same
  if(length(intersect(colnames(Synthetic_Data), colnames(Real_Data))) != ncol(Synthetic_Data)) {
    stop("Column names should be the same in synthetic and real data frames.")
  }
  
  # Initialize dataframe to store 1 - TVD values
  vals <- matrix(data = NA, nrow = ncol(Synthetic_Data), ncol = 2) %>% as.data.frame()
  colnames(vals) <- c("Variable", "TVD Statistic")
  
  # Calculate TVD, then store 1 - TVD statistic for each variable
  for(i in 1:ncol(Synthetic_Data)) {
    # i'th variable
    col_name <- names(Synthetic_Data)[i]
    
    # Save variable name
    vals[i, 1] <- col_name
    
    # Save 1 - TVD statistic
    vals[i, 2] <- 1 - tvd.calc(Synthetic_Data, Real_Data, col_name)
  }
  
  # Return df of 1 - KS statistics
  return(vals)
  
}

# Normalized difference between correlation of two given continuous variables in 
# the real v. synthetic data
# Score = 1 - (0.5 * |Corr_synth - Corr_real|) --> "similarity score" due to  1 - (value)
# We will use Spearman correlation, not Pearson
Corr.Sim.Score.Spearman <- function(Synthetic_Data, Real_Data) {
  # Calculate correlation for all combinations of pairs of continuous variables
  # Returns dataframe of pairs of variables, correlation in synthetic and real data, and similarity score
  
  # Throw error if Synthetic_Data and Real_Data have different number of columns
  if(ncol(Synthetic_Data) != ncol(Real_Data)) {
    stop("Number of columns in synthetic and real data frames should be equal.")
  }
  
  # Throw error if column names of Synthetic_Data and Real_Data are not the same
  if(length(intersect(colnames(Synthetic_Data), colnames(Real_Data))) != ncol(Synthetic_Data)) {
    stop("Column names should be the same in synthetic and real data frames.")
  }
  
  # Calculate correlations in synthetic data
  Corr_synth <- Synthetic_Data %>%
    as.matrix() %>%
    cor(use = "everything", method = "spearman") %>%
    as.data.frame %>%
    rownames_to_column(var = 'var1') %>%
    gather(var2, value, -var1) %>%
    rename(corr_synth = value)
  
  # Deal with missing values by using only complete data
  for(i in 1:nrow(Corr_synth)) {
    if(is.na(Corr_synth[i, "corr_synth"])) {
      Corr_synth[i, "corr_synth"] <- cor(Synthetic_Data[, Corr_synth[i, "var1"]],
                                         Synthetic_Data[, Corr_synth[i, "var2"]],
                                         use = "complete.obs",
                                         method = "spearman")
    }
  }
  
  # Calculate correlations in real data
  Corr_real <- Real_Data %>%
    as.matrix() %>%
    cor(use = "everything", method = "spearman") %>%
    as.data.frame %>%
    rownames_to_column(var = 'var1') %>%
    gather(var2, value, -var1) %>%
    rename(corr_real = value)
  
  # Deal with missing values by using only complete data
  for(i in 1:nrow(Corr_real)) {
    if(is.na(Corr_real[i, "corr_real"])) {
      Corr_real[i, "corr_real"] <- cor(Real_Data[, Corr_real[i, "var1"]],
                                       Real_Data[, Corr_real[i, "var2"]],
                                       use = "complete.obs",
                                       method = "spearman")
    }
  }
  
  # Join tables so all correlation values are in same table
  Corr_all <- left_join(Corr_synth, Corr_real, by = c("var1", "var2")) %>%
    # Remove variances (i.e., corr between a variable and itself)
    filter(corr_real != 1 & corr_synth != 1) %>%
    # Remove duplicate pairs
    mutate(var_order = paste(var1, var2) %>%
             strsplit(split = ' ') %>%
             map_chr( ~ sort(.x) %>% 
                        paste(collapse = ' '))) %>%
    mutate(cnt = 1) %>%
    group_by(var_order) %>%
    mutate(cumsum = cumsum(cnt)) %>%
    filter(cumsum != 2) %>%
    ungroup %>%
    select(-var_order, -cnt, -cumsum) %>%
    # Calculate normalized similarity score
    mutate(score = 1 - 0.5*(abs(corr_synth - corr_real)))
  
  # Return final table of results
  return(Corr_all)
}

# Calculate this by hand:
# calculate difference between synthetic and real proportions of each combination of categories
# of two given variables,
# sum the absolute value of all differences,
# then divide the sum by 2.
bivariatetvd.calc <- function(Synthetic_Data, Real_Data, Var1, Var2) {
  # Synthetic_Data is a dataframe containing synthetic data
  # Real_Data is a dataframe containing real data
  # Var1 and Var2 are character strings of variable names
  
  synthetic_prop <- table(Synthetic_Data[, Var1] %>% unlist(), 
                          Synthetic_Data[, Var2] %>% unlist())/nrow(Synthetic_Data)
  real_prop <- table(Real_Data[, Var1] %>% unlist(), 
                     Real_Data[, Var2] %>% unlist())/nrow(Real_Data)
  bivariatetvd <- 0.5 * sum(abs(synthetic_prop - real_prop))
  
  return(bivariatetvd)
}

# Function to capture all bivariate comparisons
# Takes as input Synthetic_Data, Real_Data
# If var 1 is continuous and var 2 is continuous, calculate correlations (Spearman and Pearson)
# If var 1 is continuous and var 2 is discrete, or vice versa, 
# bin the continous variable, then calculate contingency score
# If var 1 is discrete and var 2 is discrete, calculate contingency score
# This function calls the previously-defined function: bivariatetvd.calc
bivar.metrics <- function(Synthetic_Data, Real_Data) {
  
  # Throw error if Synthetic_Data and Real_Data have different number of columns
  if(ncol(Synthetic_Data) != ncol(Real_Data)) {
    stop("Number of columns in synthetic and real data frames should be equal.")
  }
  
  # Throw error if column names of Synthetic_Data and Real_Data are not the same
  if(length(intersect(colnames(Synthetic_Data), colnames(Real_Data))) != ncol(Synthetic_Data)) {
    stop("Column names should be the same in synthetic and real data frames.")
  }
  
  # Initialize dataframe to store similarity score values
  # Note: number of rows (i.e., values) = n(n-1)/2
  vals <- matrix(data = NA, 
                 nrow = ncol(Synthetic_Data)*(ncol(Synthetic_Data) - 1)/2, 
                 ncol = 5) %>% as.data.frame()
  colnames(vals) <- c("Var1", "Var2", "Corr_Score_Spearman", "Corr_Score_Pearson", "Contin_Score")
  
  # Fill in values for Var1 and Var2 columns
  varlist <- names(Synthetic_Data)
  df <- expand.grid(varlist, varlist) %>% 
    mutate(combo = paste(Var1, Var2)) %>% 
    group_by(combo) %>% 
    unique()
  # Unique combos of var 1 and var 2
  unique_combos <- df %>% 
    group_by(combo) %>%
    dplyr::select(combo) %>% 
    apply(1, function(x) paste(sort(unlist(strsplit(x, " "))), collapse = " ")) %>% 
    unique() %>%
    as.data.frame()
  colnames(unique_combos) <- c("combo")
  
  # Var 1 and Var 2 as separate columns, and as combo in 1 column
  df <- df %>% 
    right_join(unique_combos, by = "combo") %>% 
    filter(Var1 != Var2)
  
  vals[, 1] <- df[, 1]
  vals[, 2] <- df[, 2]
  
  # Calculate scores
  for(i in 1:nrow(vals)) {
    var1 <- vals[i, 1] %>% as.character()
    var2 <- vals[i, 2] %>% as.character()
    
    # If var1 and var2 are both continuous, calculate Spearman and Pearson correlation
    if((is.numeric(unlist(Synthetic_Data[, var1])) & is.numeric(unlist(Synthetic_Data[, var2])) & 
        is.numeric(unlist(Real_Data[, var1]))& is.numeric(unlist(Real_Data[, var2])))) {
      
      # Spearman
      corr_spearman_synth <- cor(x = Synthetic_Data[, var1], 
                                 y = Synthetic_Data[, var2],
                                 use = "complete.obs",
                                 method = "spearman")
      corr_spearman_real <- cor(x = Real_Data[, var1], 
                                y = Real_Data[, var2],
                                use = "complete.obs",
                                method = "spearman")
      vals[i, 3] = 1 - 0.5*(abs(corr_spearman_synth - corr_spearman_real))
      
      # Pearson
      corr_pearson_synth <- cor(x = Synthetic_Data[, var1], 
                                y = Synthetic_Data[, var2],
                                use = "complete.obs",
                                method = "pearson")
      corr_pearson_real <- cor(x = Real_Data[, var1], 
                               y = Real_Data[, var2],
                               use = "complete.obs",
                               method = "pearson")
      vals[i, 4] = 1 - 0.5*(abs(corr_pearson_synth - corr_pearson_real))
    }
    
    # If var1 and var2 are both discrete, calculate contingency score
    else if((is.factor(unlist(Synthetic_Data[, var1])) & is.factor(unlist(Synthetic_Data[, var2])) & 
             is.factor(unlist(Real_Data[, var1]))& is.factor(unlist(Real_Data[, var2])))) {
      
      # Calculate and store contingency score
      vals[i, 5] <- 1 - bivariatetvd.calc(Synthetic_Data = Synthetic_Data,
                                          Real_Data = Real_Data,
                                          Var1 = var1, 
                                          Var2 = var2)
    }
    
    # If one of var1 and var2 is continuous and the other is discrete,
    # bin the continuous variable and treat as categorical,
    # then calculate contingency score.
    else if((is.numeric(unlist(Synthetic_Data[, var1])) & is.factor(unlist(Synthetic_Data[, var2])) & 
             is.numeric(unlist(Real_Data[, var1]))& is.factor(unlist(Real_Data[, var2]))) |
            (is.factor(unlist(Synthetic_Data[, var1])) & is.numeric(unlist(Synthetic_Data[, var2])) & 
             is.factor(unlist(Real_Data[, var1])) & is.numeric(unlist(Real_Data[, var2])))) {
      
      # Identify the continuous variable
      # var1 is continuous
      if(is.numeric(unlist(Synthetic_Data[, var1])) & is.numeric(unlist(Real_Data[, var1]))) {
        
        # Bin this variable (using quantiles)
        Synthetic_Data_bin <- Synthetic_Data %>%
          mutate(cont_binned = cut(Synthetic_Data[, var1] %>% unlist(), 4))
        
        Real_Data_bin <- Real_Data %>%
          mutate(cont_binned = cut(Real_Data[, var1] %>% unlist(), 4))
        
        vals[i, 5] <- 1 - bivariatetvd.calc(Synthetic_Data = Synthetic_Data_bin,
                                            Real_Data = Real_Data_bin,
                                            Var1 = "cont_binned", 
                                            Var2 = var2)
      }
      # var2 is continuous
      else {
        # Bin this variable (using quantiles)
        Synthetic_Data_bin <- Synthetic_Data %>%
          mutate(cont_binned = cut(Synthetic_Data[, var2] %>% unlist(), 4))
        
        Real_Data_bin <- Real_Data %>%
          mutate(cont_binned = cut(Real_Data[, var2] %>% unlist(), 4))
        
        vals[i, 5] <- 1 - bivariatetvd.calc(Synthetic_Data = Synthetic_Data_bin,
                                            Real_Data = Real_Data_bin,
                                            Var1 = var1, 
                                            Var2 = "cont_binned")
      }
    }
  }
  
  return(vals)
}

### Generate and evaluate synthetic data: ####

# ARF Sequential
arf.sim.eval <- function(n, n_obs, real_data, col_order, var_cont, var_disc, random_seed) {
  
  # This function takes as input:
  # n: number of data sets to generate (i.e., number of replications)
  # n_obs: number of observations to generate per data set
  # real_data: the real data set
  # col_order: the order of columns in the real and synthetic data sets
  # var_cont: the names of the continuous columns in the data set
  # var_disc: the names of the discrete columns in the data set
  # random_seed: an integer for the random seed
  
  # First, make sure columns are in the correct order
  data_real <- real_data[, col_order]
  
  # Initialize objects to store all metrics
  
  # Univariate continuous
  univar_cont_all <- matrix(data = NA, nrow = n, ncol = length(var_cont)) %>% as.data.frame()
  colnames(univar_cont_all) <- var_cont
  
  # Univariate discrete
  univar_disc_all <- matrix(data = NA, nrow = n, ncol = length(var_disc)) %>% as.data.frame()
  colnames(univar_disc_all) <- var_disc
  
  # Bivariate (continuous x continuous, discrete x continuous, discrete x discrete)
  bivar_all <- matrix(data = NA, nrow = 0, ncol = 5)
  colnames(bivar_all) <- c("Var1", "Var2", "Corr_Score_Spearman", "Corr_Score_Pearson", "Contin_Score")
  
  # ML efficacy
  MLeff_xgb <- matrix(data = NA, nrow = 0, ncol = 4) 
  colnames(MLeff_xgb) <- c("Data", "Precision", "Recall", "F1")
  
  MLeff_knn <- matrix(data = NA, nrow = 0, ncol = 4)
  colnames(MLeff_knn) <- c("Data", "Precision", "Recall", "F1")
  
  # Trial inference
  inference_all <- matrix(data = NA, nrow = 0, ncol = 3)
  colnames(inference_all) <- c("OR", "Lower CI", "Upper CI")
  
  for(i in 1:n) {
    
    # Generate data set 
    data_arf <- ARF_data_generation(real_data = data_real,
                                    random_seed = random_seed + i,
                                    n_obs = n_obs)
    
    # Make sure columns are in the same order as real data set
    # This is not actually necessary for R, but to make things consistent with python
    data_arf <- data_arf[, col_order]
    
    # Compute univariate metrics
    
    # Calculate univariate continuous metrics for current data set
    univar_cont <- ks.stat(Synthetic_Data = data_arf[, var_cont], 
                           Real_Data = data_real[, var_cont]) %>% t()
    
    # Store in univar_cont_all
    univar_cont_all[i, ] <- univar_cont[2, ]
    
    # Calculate univariate discrete metrics for current data set
    univar_disc <- tvd.stat(Synthetic_Data = data_arf[, var_disc],
                            Real_Data = data_real[, var_disc]) %>% t()
    
    # Store in univar_disc_all
    univar_disc_all[i, ] <- univar_disc[2, ]
    
    # Calculate all bivariate metrics for current data set
    
    # First, remove id column
    data_arf_noid <- data_arf[, !(names(data_arf) %in% c("pidnum"))]
    data_real_noid <- data_real[, !(names(data_real) %in% c("pidnum"))]
    
    bivar <- bivar.metrics(Synthetic_Data = data_arf_noid, Real_Data = data_real_noid)
    
    # Store in bivar_all
    bivar_all <- rbind(bivar_all, bivar)
    
    # Calculate ML efficacy metrics
    
    # XGBoost
    
    # Split real data into training set and test set, and separate labels (outcome) from covariates
    # 70:30 split
    train_split <- 0.7
    
    # Set random seed
    set.seed(random_seed + i)
    
    # Sample rows for training set (70%)
    sample_indices <- sample.split(data_real[, 1], SplitRatio = 0.70)
    
    # Make sure data is all type numeric and stored in a matrix
    real_matrix <- apply(data_real, 2, as.numeric) %>% as.matrix()
    
    data_train_real <- real_matrix[sample_indices, ]
    data_test_real <- real_matrix[!sample_indices, ]
    
    # Train the prediction model on training set (real data)
    xgb_mod_real <- xgboost(data = data_train_real[, !(names(data_real) %in% c("cens", "pidnum"))], 
                            label = data_train_real[, (names(data_real) %in% c("cens"))], 
                            max.depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")
    
    # Predict the outcome on the real data test set based on trained model
    y_prob_real <- predict(xgb_mod_real, data_test_real[, !(names(data_real) %in% c("pidnum", "cens"))])
    y_pred_real <- as.numeric(y_prob_real > 0.5)
    
    # Confusion matrix
    cm_real <- confusionMatrix(as.factor(data_test_real[, "cens"]), as.factor(y_pred_real))
    
    # Metrics (Precision, Recall, F1-Score)
    MLmetrics_xgb_real <- cm_real$byClass[c("Precision", "Recall", "F1")]
    
    # Store in MLeff_xgb
    tostore <- c("Real", MLmetrics_xgb_real)
    MLeff_xgb <- rbind(MLeff_xgb, tostore)
    
    # Train another prediction model on synthetic data as the training set
    
    # Make sure data is all type numeric and stored in a matrix
    arf_matrix <- apply(data_arf, 2, as.numeric) %>% as.matrix()
    
    data_train_arf <- arf_matrix[sample_indices, ]
    data_test_arf <- arf_matrix[!sample_indices, ]
    
    # Train the prediction model on training set (synthetic data)
    xgb_mod_arf <- xgboost(data = data_train_arf[, !(names(data_arf) %in% c("cens", "pidnum"))],
                           label = data_train_arf[, (names(data_arf) %in% c("cens"))],
                           max.depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")
    
    # Predict the outcome on the real data test set based on trained model
    y_prob_arf <- predict(xgb_mod_arf, data_test_real[, !(names(data_real) %in% c("pidnum", "cens"))])
    y_pred_arf <- as.numeric(y_prob_arf > 0.5)
    
    # Confusion matrix
    cm_arf <- confusionMatrix(as.factor(data_test_real[, "cens"]), as.factor(y_pred_arf))
    
    # Metrics (Precision, Recall, F1-Score)
    MLmetrics_xgb_arf <- cm_arf$byClass[c("Precision", "Recall", "F1")]
    
    # Store in MLeff_xgb
    tostore <- c("Synthetic", MLmetrics_xgb_arf)
    MLeff_xgb <- rbind(MLeff_xgb, tostore)
    
    # KNN
    
    set.seed(random_seed + i)
    
    # Need to impute missing values first (for real data)
    preProcValues_real <- caret::preProcess(data_real, method = c("knnImpute"),
                                            k = 5, knnSummary = mean)
    
    knn_imp_real <- predict(preProcValues_real, data_real, na.action = na.pass)
    
    procNames_real <- data.frame(col = names(preProcValues_real$mean), 
                                 mean = preProcValues_real$mean, 
                                 sd = preProcValues_real$std)
    for(i in procNames_real$col){
      knn_imp_real[i] <- knn_imp_real[i]*preProcValues_real$std[i] + preProcValues_real$mean[i] 
    }
    
    # Split real data into training set and test set, and separate labels (outcome) from covariates
    # 70:30 split
    train_split <- 0.7
    sample_indices <- sample.split(knn_imp_real[, 1], SplitRatio = 0.70)
    
    data_train_real <- knn_imp_real[sample_indices, ]
    data_test_real <- knn_imp_real[!sample_indices, ]
    
    # Train the prediction model on training set (real data) and predict outcome 
    # KNN algorithm for prediction of the outcome based on real data
    knn_pred_real <- knn(train = data_train_real[, !(names(data_train_real) %in% "cens")],
                         test = data_test_real[, !(names(data_test_real) %in% "cens")],
                         cl = data_train_real[, names(data_train_real) %in% "cens"], 
                         k = 5)
    
    # Confusion matrix
    cm_knn_real <- confusionMatrix(data_test_real[, "cens"], knn_pred_real)
    
    # Metrics (Precision, Recall, F1-Score)
    MLmetrics_knn_real <- cm_knn_real$byClass[c("Precision", "Recall", "F1")]
    
    # Store in MLeff_knn
    tostore <- c("Real", MLmetrics_knn_real)
    MLeff_knn <- rbind(MLeff_knn, tostore)
    
    # Method: ARF vbc + exec
    
    # Split real data into training set and test set, and separate labels (outcome) from covariates
    # 70:30 split
    arf_df <- data_arf %>% as.data.frame()
    
    data_train_arf <- arf_df[sample_indices, ]
    data_test_arf <- arf_df[!sample_indices, ]
    
    knn_pred_arf <- knn(train = data_train_arf[, !(names(data_train_arf) %in% "cens")],
                        test = data_test_real[, !(names(data_test_real) %in% "cens")],
                        cl = data_train_arf[, names(data_train_arf) %in% "cens"],
                        k = 5)
    
    # Confusion matrix
    cm_knn_arf <- confusionMatrix(data_test_real[, "cens"], knn_pred_arf)
    
    # Metrics (Precision, Recall, F1-Score)
    MLmetrics_knn_arf <- cm_knn_arf$byClass[c("Precision", "Recall", "F1")]
    
    # Store in MLeff_knn
    tostore <- c("Synthetic", MLmetrics_knn_arf)
    MLeff_knn <- rbind(MLeff_knn, tostore)
    
    # Trial inference metrics
    
    # Create dichotomous tx variable for simplicity
    data_arf <- data_arf %>% 
      mutate(tx_bin = case_when(arms == 0 ~ 0,
                                arms == 1 ~ 1,
                                arms == 2 ~ 1,
                                arms == 3 ~ 1))
    
    # Fit logistic regression model (no confounders)
    mod <- glm(formula = cens ~ as.factor(tx_bin), family = "binomial", data = data_arf)
    
    # OR, CI for treatment
    OR_est <- exp(mod$coefficients)[2] %>% as.numeric()
    CI_est <- exp(confint(mod))[2,] %>% as.numeric()
    
    # Store in inference_all
    tostore <- c(OR_est, CI_est)
    inference_all <- rbind(inference_all, tostore)
    
  }
  
  # Remove row names from MLeff_xgb, MLeff_knn
  rownames(MLeff_xgb) <- NULL
  rownames(MLeff_knn) <- NULL
  rownames(inference_all) <- NULL
  
  return(list(Univar_Cont = univar_cont_all, Univar_Disc = univar_disc_all, Bivar = as.data.frame(bivar_all),
              MLefficacy_XGB = as.data.frame(MLeff_xgb), MLefficacy_KNN = as.data.frame(MLeff_knn),
              Trial_Inference = as.data.frame(inference_all)))
  
}

# R-Vine Copula Sequential
rvine.sim.eval <- function(n, n_obs, real_data, col_order, var_cont, var_disc, random_seed) {
  
  # This function takes as input:
  # n: number of data sets to generate (i.e., number of replications)
  # n_obs: number of observations to generate per data set
  # real_data: the real data set
  # col_order: the order of columns in the real and synthetic data sets
  # var_cont: the names of the continuous columns in the data set
  # var_disc: the names of the discrete columns in the data set
  # random_seed: an integer for the random seed
  
  # First, make sure columns are in the correct order
  data_real <- real_data[, col_order]
  
  # Initialize objects to store all metrics
  
  # Univariate continuous
  univar_cont_all <- matrix(data = NA, nrow = n, ncol = length(var_cont)) %>% as.data.frame()
  colnames(univar_cont_all) <- var_cont
  
  # Univariate discrete
  univar_disc_all <- matrix(data = NA, nrow = n, ncol = length(var_disc)) %>% as.data.frame()
  colnames(univar_disc_all) <- var_disc
  
  # Bivariate (continuous x continuous, discrete x continuous, discrete x discrete)
  bivar_all <- matrix(data = NA, nrow = 0, ncol = 5)
  colnames(bivar_all) <- c("Var1", "Var2", "Corr_Score_Spearman", "Corr_Score_Pearson", "Contin_Score")
  
  # ML efficacy
  MLeff_xgb <- matrix(data = NA, nrow = 0, ncol = 4) 
  colnames(MLeff_xgb) <- c("Data", "Precision", "Recall", "F1")
  
  MLeff_knn <- matrix(data = NA, nrow = 0, ncol = 4)
  colnames(MLeff_knn) <- c("Data", "Precision", "Recall", "F1")
  
  # Trial inference
  inference_all <- matrix(data = NA, nrow = 0, ncol = 3)
  colnames(inference_all) <- c("OR", "Lower CI", "Upper CI")
  
  for(i in 1:n) {
    
    # Generate data set 
    data_rvine <- RVine_data_generation(real_data = data_real,
                                                   random_seed = random_seed + i,
                                                   n_obs = n_obs)
    
    # Make sure columns are in the same order as real data set
    # This is not actually necessary for R, but to make things consistent with python
    data_rvine <- data_rvine[, col_order]
    
    # Compute univariate metrics
    
    # Calculate univariate continuous metrics for current data set
    univar_cont <- ks.stat(Synthetic_Data = data_rvine[, var_cont], 
                           Real_Data = data_real[, var_cont]) %>% t()
    
    # Store in univar_cont_all
    univar_cont_all[i, ] <- univar_cont[2, ]
    
    # Calculate univariate discrete metrics for current data set
    univar_disc <- tvd.stat(Synthetic_Data = data_rvine[, var_disc],
                            Real_Data = data_real[, var_disc]) %>% t()
    
    # Store in univar_disc_all
    univar_disc_all[i, ] <- univar_disc[2, ]
    
    # Calculate all bivariate metrics for current data set
    
    # First, remove id column
    data_rvine_noid <- data_rvine[, !(names(data_rvine) %in% c("pidnum"))]
    data_real_noid <- data_real[, !(names(data_real) %in% c("pidnum"))]
    
    bivar <- bivar.metrics(Synthetic_Data = data_rvine_noid, Real_Data = data_real_noid)
    
    # Store in bivar_all
    bivar_all <- rbind(bivar_all, bivar)
    
    # Calculate ML efficacy metrics
    
    # XGBoost
    
    # Split real data into training set and test set, and separate labels (outcome) from covariates
    # 70:30 split
    train_split <- 0.7
    
    # Set random seed
    set.seed(random_seed + i)
    
    # Sample rows for training set (70%)
    sample_indices <- sample.split(data_real[, 1], SplitRatio = 0.70)
    
    # Make sure data is all type numeric and stored in a matrix
    real_matrix <- apply(data_real, 2, as.numeric) %>% as.matrix()
    
    data_train_real <- real_matrix[sample_indices, ]
    data_test_real <- real_matrix[!sample_indices, ]
    
    # Train the prediction model on training set (real data)
    xgb_mod_real <- xgboost(data = data_train_real[, !(names(data_real) %in% c("cens", "pidnum"))], 
                            label = data_train_real[, (names(data_real) %in% c("cens"))], 
                            max.depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")
    
    # Predict the outcome on the real data test set based on trained model
    y_prob_real <- predict(xgb_mod_real, data_test_real[, !(names(data_real) %in% c("pidnum", "cens"))])
    y_pred_real <- as.numeric(y_prob_real > 0.5)
    
    # Confusion matrix
    cm_real <- confusionMatrix(as.factor(data_test_real[, "cens"]), as.factor(y_pred_real))
    
    # Metrics (Precision, Recall, F1-Score)
    MLmetrics_xgb_real <- cm_real$byClass[c("Precision", "Recall", "F1")]
    
    # Store in MLeff_xgb
    tostore <- c("Real", MLmetrics_xgb_real)
    MLeff_xgb <- rbind(MLeff_xgb, tostore)
    
    # Train another prediction model on synthetic data as the training set
    
    # Make sure data is all type numeric and stored in a matrix
    rvine_matrix <- apply(data_rvine, 2, as.numeric) %>% as.matrix()
    
    data_train_rvine <- rvine_matrix[sample_indices, ]
    data_test_rvine <- rvine_matrix[!sample_indices, ]
    
    # Train the prediction model on training set (synthetic data)
    xgb_mod_rvine <- xgboost(data = data_train_rvine[, !(names(data_rvine) %in% c("cens", "pidnum"))], 
                             label = data_train_rvine[, (names(data_rvine) %in% c("cens"))], 
                             max.depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")
    
    # Predict the outcome on the real data test set based on trained model
    y_prob_rvine <- predict(xgb_mod_rvine, data_test_real[, !(names(data_real) %in% c("pidnum", "cens"))])
    y_pred_rvine <- as.numeric(y_prob_rvine > 0.5)
    
    # Confusion matrix
    cm_rvine <- confusionMatrix(as.factor(data_test_real[, "cens"]), as.factor(y_pred_rvine))
    
    # Metrics (Precision, Recall, F1-Score)
    MLmetrics_xgb_rvine <- cm_rvine$byClass[c("Precision", "Recall", "F1")]
    
    # Store in MLeff_xgb
    tostore <- c("Synthetic", MLmetrics_xgb_rvine)
    MLeff_xgb <- rbind(MLeff_xgb, tostore)
    
    # KNN
    
    set.seed(random_seed + i)
    
    # Need to impute missing values first (for real data)
    preProcValues_real <- caret::preProcess(data_real, method = c("knnImpute"),
                                            k = 5, knnSummary = mean)
    
    knn_imp_real <- predict(preProcValues_real, data_real, na.action = na.pass)
    
    procNames_real <- data.frame(col = names(preProcValues_real$mean), 
                                 mean = preProcValues_real$mean, 
                                 sd = preProcValues_real$std)
    for(i in procNames_real$col){
      knn_imp_real[i] <- knn_imp_real[i]*preProcValues_real$std[i] + preProcValues_real$mean[i] 
    }
    
    # Split real data into training set and test set, and separate labels (outcome) from covariates
    # 70:30 split
    train_split <- 0.7
    sample_indices <- sample.split(knn_imp_real[, 1], SplitRatio = 0.70)
    
    data_train_real <- knn_imp_real[sample_indices, ]
    data_test_real <- knn_imp_real[!sample_indices, ]
    
    # Train the prediction model on training set (real data) and predict outcome 
    # KNN algorithm for prediction of the outcome based on real data
    knn_pred_real <- knn(train = data_train_real[, !(names(data_train_real) %in% "cens")],
                         test = data_test_real[, !(names(data_test_real) %in% "cens")],
                         cl = data_train_real[, names(data_train_real) %in% "cens"], 
                         k = 5)
    
    # Confusion matrix
    cm_knn_real <- confusionMatrix(data_test_real[, "cens"], knn_pred_real)
    
    # Metrics (Precision, Recall, F1-Score)
    MLmetrics_knn_real <- cm_knn_real$byClass[c("Precision", "Recall", "F1")]
    
    # Store in MLeff_knn
    tostore <- c("Real", MLmetrics_knn_real)
    MLeff_knn <- rbind(MLeff_knn, tostore)
    
    # Method: R-vine vbc + exec
    
    # Split real data into training set and test set, and separate labels (outcome) from covariates
    # 70:30 split
    rvine_df <- data_rvine %>% as.data.frame()
    
    data_train_rvine <- rvine_df[sample_indices, ]
    data_test_rvine <- rvine_df[!sample_indices, ]
    
    knn_pred_rvine <- knn(train = data_train_rvine[, !(names(data_train_rvine) %in% "cens")],
                          test = data_test_real[, !(names(data_test_real) %in% "cens")],
                          cl = data_train_rvine[, names(data_train_rvine) %in% "cens"],
                          k = 5)
    
    # Confusion matrix
    cm_knn_rvine <- confusionMatrix(data_test_real[, "cens"], knn_pred_rvine)
    
    # Metrics (Precision, Recall, F1-Score)
    MLmetrics_knn_rvine <- cm_knn_rvine$byClass[c("Precision", "Recall", "F1")]
    
    # Store in MLeff_knn
    tostore <- c("Synthetic", MLmetrics_knn_rvine)
    MLeff_knn <- rbind(MLeff_knn, tostore)
    
    # Trial inference metrics
    
    # Create dichotomous tx variable for simplicity
    data_rvine <- data_rvine %>% 
      mutate(tx_bin = case_when(arms == 0 ~ 0,
                                arms == 1 ~ 1,
                                arms == 2 ~ 1,
                                arms == 3 ~ 1))
    
    # Fit logistic regression model (no confounders)
    mod <- glm(formula = cens ~ as.factor(tx_bin), family = "binomial", data = data_rvine)
    
    # OR, CI for treatment
    OR_est <- exp(mod$coefficients)[2] %>% as.numeric()
    CI_est <- exp(confint(mod))[2,] %>% as.numeric()
    
    # Store in inference_all
    tostore <- c(OR_est, CI_est)
    inference_all <- rbind(inference_all, tostore)
  } 
  
  # Remove row names from MLeff_xgb, MLeff_knn, inference_all
  rownames(MLeff_xgb) <- NULL
  rownames(MLeff_knn) <- NULL
  rownames(inference_all) <- NULL
  
  return(list(Univar_Cont = univar_cont_all, Univar_Disc = univar_disc_all, Bivar = as.data.frame(bivar_all),
              MLefficacy_XGB = as.data.frame(MLeff_xgb), MLefficacy_KNN = as.data.frame(MLeff_knn),
              Trial_Inference = as.data.frame(inference_all)))
  
}


# 4. Run Simulations ----

# ARF Sequential
start_time <- Sys.time()
sim_test_arf <- arf.sim.eval(n = 500, n_obs = 2139, real_data = real,
                             col_order = c("pidnum", "age", "wtkg", "hemo", "homo",
                                           "drugs", "karnof", "oprior", "z30", "preanti",
                                           "race", "gender", "strat", "symptom", "cd40",
                                           "arms", "cd420", "cd496", "cens"),
                             var_cont = c("age", "wtkg", "preanti", "cd40", "cd420",
                                          "cd496"),
                             var_disc = c("hemo", "homo", "drugs", "oprior", "z30", 
                                          "race", "gender", "strat", "symptom", 
                                          "karnof", "arms", "cens"),
                             random_seed = 20240806)
end_time <- Sys.time()
(time_taken <- end_time - start_time)

# R-Vine Copula Sequential
start_time <- Sys.time()
sim_test_rvine <- rvine.sim.eval(n = 500, n_obs = 2139, real_data = real,
                                 col_order = c("pidnum", "age", "wtkg", "hemo", 
                                               "homo", "drugs", "karnof", "oprior", 
                                               "z30", "preanti", "race", "gender", 
                                               "strat", "symptom", "cd40", "arms", 
                                               "cd420", "cd496", "cens"),
                                 var_cont = c("age", "wtkg", "preanti", "cd40", 
                                              "cd420", "cd496"),
                                 var_disc = c("hemo", "homo", "drugs", "oprior", 
                                              "z30", "race", "gender", "strat", 
                                              "symptom", "karnof", "arms", "cens"),
                                 random_seed = 20240806)
end_time <- Sys.time()
(time_taken <- end_time - start_time)
