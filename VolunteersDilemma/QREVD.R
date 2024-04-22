library(kableExtra)
library(tidyverse)
library(mnorm)
library(rstan)
  rstan_options(auto_write = TRUE)
  options(mc.cores = parallel::detectCores())
  rstan_options(threads_per_chain = 1)
  
  
RunML<-TRUE  

if (RunML) {  
  library(bridgesampling)
  
  ML<-list()
  
}
  
D<-read.csv("Data/GHS2017VD.csv") |> 
    # This dataset does not explicitly store the group size
    # I can infer it from the "Other.Id." variable, but
    # this is not consistent with the experiment description
    # in the "Procedures" section of the paper. Specifically,
    # it looked like there were two GroupSize=7 session, when
    # in reality there should have been one with 9, and one
    # with 12. This fixes things
    mutate(GroupSize = str_count(Other.Id., "ID")+1,
           GroupSize = ifelse(SessionID==10,9,GroupSize),
           GroupSize = ifelse(SessionID==11,12,GroupSize)
    ) |>
    rename(Volunteer = Decision..1.vol..) |>
    mutate(ID = paste0(SessionID,"-",ID) |> as.factor() |> as.numeric() ) |>
    select(ID,GroupSize,Volunteer) |>
    # all I need are the counts of actions for each participant:
    group_by(ID,GroupSize) |>
    summarize(Volunteer = sum(Volunteer),count=n())
  
######################################################
# Homogeneous model
######################################################

file<-"Presentation/VolunteersDilemma/Estimates_VDQREhomogeneous.rds"
if (!file.exists(file) | RunML) {
  model<-stan_model("Presentation/VolunteersDilemma/VDQREhomogeneous_CorrectorLogit.stan")
  
  d<-D |>
    group_by(GroupSize) |>
    summarize(Volunteer = sum(Volunteer),
              count = sum(count))
  
  dStan<-list(
    N=dim(d)[1],
    Volunteer = d$Volunteer,
    count = d$count,
    GroupSize = d$GroupSize,
    V = 1.0,
    c = 0.2,
    L = 0.2,
    prior_lambda = c(log(5),1),
    
    tol = 1e-4,
    maxiter = 1000
  )
  
  Fit<-sampling(model,
                data=dStan,
                seed=42,
                chains=4,
                iter=2000
                )
  Fit |> saveRDS(file)
  
  if (RunML) {  
    
    homogeneous<-Fit |> bridge_sampler()
    
  }
}


#############################################
# Warm glow volunteering heterogeneous model
#############################################


file<-"Presentation/VolunteersDilemma/Estimates_VDQREwarmglow.rds"
if (!file.exists(file)  | RunML) {
  model<-stan_model("Presentation/VolunteersDilemma/VDQREwarmglow.stan")
  #stop()
  
  D<-D |>
    mutate(
      GameID = ifelse(
        GroupSize==2,1,ifelse(
          GroupSize==3,2,ifelse(
            GroupSize==6,3,ifelse(
              GroupSize==9,4,5
            )
          )
        )
      )
    )
  
  
  dStan<-list(
    N=dim(D)[1],
    Volunteer = D$Volunteer,
    count = D$count,
    
    nTreatments = 5,
    GroupSize = c(2,3,6,9,12),
    
    GameID = D$GameID,
    
    V = 1.0,
    c = 0.2,
    L = 0.2,
    prior_lambda = c(log(5),1),
    prior_mu = c(0,1),
    prior_tau = 0.05,
    
    maxiter = 1000,
    tol = 1e-4,
    
    nSim = 100,
    # This gets me Halton draws, transformed to be from the
    # standard normal, truncated to lie between -1 and 1
    deltaZ = (pnorm(1)+halton(100)*(1-2*pnorm(1))) |> qnorm() |> as.vector()
  )
  
  Fit<-sampling(model,data=dStan,seed=123,chains=4,iter=2000,
                control=list(adapt_delta = 0.8)
                #,
                #pars = "delta_i",include=FALSE
  )
  Fit |> saveRDS(file)
  
  if (RunML) {  
    
    warmglow<-Fit |> bridge_sampler()
    
  }
}

#############################################
# Duplicate aversion heterogeneous model
#############################################



file<-"Presentation/VolunteersDilemma/Estimates_VDQREduplicateaversion.rds"
if (!file.exists(file)  | RunML) {
  model<-stan_model("Presentation/VolunteersDilemma/VDQREduplicateaversion.stan")
  #stop()
  
  D<-D |>
    mutate(
      GameID = ifelse(
        GroupSize==2,1,ifelse(
          GroupSize==3,2,ifelse(
            GroupSize==6,3,ifelse(
              GroupSize==9,4,5
            )
          )
        )
      )
    )
  
  
  dStan<-list(
    N=dim(D)[1],
    Volunteer = D$Volunteer,
    count = D$count,
    
    nTreatments = 5,
    GroupSize = c(2,3,6,9,12),
    
    GameID = D$GameID,
    
    V = 1.0,
    c = 0.2,
    L = 0.2,
    prior_lambda = c(log(5),1),
    prior_mu = c(0,1),
    prior_tau = 0.05,
    
    maxiter = 1000,
    tol = 1e-4,
    
    nSim = 100,
    # This gets me Halton draws, transformed to be from the
    # standard normal, truncated to lie between -1 and 1
    deltaZ = (pnorm(1)+halton(100)*(1-2*pnorm(1))) |> qnorm() |> as.vector()
  )
  Fit<-sampling(model,data=dStan,seed=42,chains=4,iter=2000,
                control=list(adapt_delta = 0.9,max_treedepth=15)
                #,
                #pars = "gamma_i",include=FALSE
  )
  Fit |> saveRDS(file) 
  
  
  if (RunML) {  
    
    duplicateaversion<-Fit |> bridge_sampler()
    
  }
  
 
}



if (RunML) {  
  
  BF<-list()
  
  BF[["Duplicate aversion vs homogeneous"]]<-bf(duplicateaversion,homogeneous,log=TRUE)
  BF[["Warm glow vs homogeneous"]]<-bf(warmglow,homogeneous,log=TRUE)
  BF[["Warm glow vs duplicate aversion"]]<-bf(warmglow,duplicateaversion)
  
  BF |>
    saveRDS("Presentation/VolunteersDilemma/BayesFactors.rds")
  
  post_prob(homogeneous,duplicateaversion,warmglow)|>
    saveRDS("Presentation/VolunteersDilemma/PostProb.rds")
  
}




