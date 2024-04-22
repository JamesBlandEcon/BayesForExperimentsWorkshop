library(tidyverse)
library(haven)
library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
rstan_options(threads_per_chain = 7)



ReRun<-FALSE

# Note: The show-up fee for the undergrads was $10 and $40 for the MBA students

D<-"Data/HS2023.dta" |>
  read_dta() |>
  select(id,choice,prize1L:prob4R,endowment,mba) |>
  # add in the endowment
  mutate(
    show_up_fee = ifelse(mba,40,10),
    Left = 1-choice,
    prize1L = prize1L+endowment+show_up_fee,
    prize2L = prize2L+endowment+show_up_fee,
    prize3L = prize3L+endowment+show_up_fee,
    prize4L = prize4L+endowment+show_up_fee,
    prize1R = prize1R+endowment+show_up_fee,
    prize2R = prize2R+endowment+show_up_fee,
    prize3R = prize3R+endowment+show_up_fee,
    prize4R = prize4R+endowment+show_up_fee
  ) |>
  # there were never four possible prizes
  select(-contains("4")) |>
  rowwise() |>
  mutate(
    prizerangeLow = min(c(prize1L,prize2R,prize3L)),
    prizerangeHigh = max(c(prize1L,prize2R,prize3L))
  ) |>
  ungroup() |>
  filter(!is.na(Left))


file<-"Presentation/code_CRRA/CRRAIndividual.csv"

if (!file.exists(file) | ReRun) {

  model<-"Presentation/code_CRRA/CRRAIndividual.stan" |>
    stan_model()
  
  ESTIMATES<-tibble()
  
  for (ii in unique(D$id)) {
    
    print(paste("estimating for id",ii))
    
    d<-D |> filter(id==ii)
    
    dStan<-list(
      n = dim(d)[1],
      Left = d$Left,
      
      prizes = cbind(d$prize1L,d$prize2L,d$prize3L),
      prizeRange=cbind(d$prizerangeLow,d$prizerangeHigh),
      
      probL = cbind(d$prob1L,d$prob2L,d$prob3L),
      probR = cbind(d$prob1R,d$prob2R,d$prob3R),
      
      prior_r = c(0.27,0.36),
      prior_lambda = c(3.45,0.59)
    )
    
    
    Fit <- model |>
      sampling(data=dStan,seed=42,chains=4,cores=4,
               iter=4000,
               control=list(adapt_delta=0.99999))
    
    FitSum<-summary(Fit)$summary |>
      data.frame() |>
      rownames_to_column(var = "parameter") |>
      mutate(id = ii,
             mba = mean(d$mba))
    
    ESTIMATES<-rbind(ESTIMATES,FitSum)
    
    ESTIMATES |>
      write.csv(file)
    
  }
  
  
}

file<-"Presentation/code_CRRA/CRRAHierarchicalEstimates.rds"


if (!file.exists(file) | ReRun) {
  model<-"Presentation/code_CRRA/CRRAHierarchical.stan" |>
    stan_model()
  
  
  # This selects the undergrads in the "house money" treatment. 
  d<-D |> filter(mba==0 & endowment!=80)
  
  dStan<-list(
    n = dim(d)[1],
    
    id = d$id %>% paste("-",.,"-") %>% as.factor() |> as.numeric(),
    nParticipants = d$id %>% paste("-",.,"-") %>% as.factor() |> as.numeric() |> unique() |> length(),
    
    Left = d$Left,
    
    prizes = cbind(d$prize1L,d$prize2L,d$prize3L),
    prizeRange=cbind(d$prizerangeLow,d$prizerangeHigh),
    
    probL = cbind(d$prob1L,d$prob2L,d$prob3L),
    probR = cbind(d$prob1R,d$prob2R,d$prob3R),
    
    prior_r = c(0.27,0.36),
    prior_lambda = c(3.45,0.59),
    
    prior_MU = list(c(0.27,0.18),c(3.45,0.3)),
    prior_TAU = c(0.05,0.05),
    prior_OMEGA = 4,
    
    grainsize = 1
    
  )
  
  Fit<-model |> 
    sampling(
      data=dStan,iter=2000,
      pars = "z", include = FALSE
    )

  Fit |>
    saveRDS(file)
}

Fit<-"Presentation/code_CRRA/CRRAHierarchicalEstimates.rds" |>
  readRDS()
