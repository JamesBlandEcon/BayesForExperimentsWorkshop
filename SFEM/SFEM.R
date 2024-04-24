library(tidyverse)
library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

####################################################
# Loading the data from the files available on the 
# journal website
####################################################


DBFData<-data.frame()
for (ff in 1:6) {
  fname<-paste("Data/SFEM/dfformatlab_strg_",ff,"_special.txt",sep="")
  d<-data.frame(read.delim(fname, header = FALSE, sep = "\t", dec = "."))
  colnames(d)<-c("match","round","treatment","coop","id","ocoop_all","strg1","strg2","strg3","strg4","strg5","strg6","session","id2_all")
  DBFData<-rbind(DBFData,d)
}

for (ff in 1:6) {
  str<-paste("DBFData$follow",ff,"<- as.integer(DBFData$coop == DBFData$strg",ff,")",sep="")
  eval(parse(text=str))
}
AggData<-aggregate(DBFData[,c("follow1","follow2","follow3","follow4","follow5","follow6")], by=list(DBFData$id),FUN=sum)
colnames(AggData)[1]<-"id"
AggData$n<-aggregate(DBFData$id,by=list(DBFData$id),FUN=length)$x
AggData$Treatment<-aggregate(DBFData$treatment,by=list(DBFData$id),FUN=mean)$x




priorMix<-c(1,1,1,1,1,1)

# Homogeneous model

file<-"Presentation/SFEM/Estimates_SFEMhomogeneous.csv"

if (!file.exists(file)) {

  Estimates<-tibble()
  
  for (tt in 1:6) {
    print(paste("estimating homogeneous trembles model",tt))
    
    model<-"Presentation/SFEM/SFEMhomogeneous.stan" |>
      stan_model()
    
    Dt<- AggData %>% filter(Treatment==tt)
    
    d = list(
      
      N=dim(Dt)[1],
      S=6,
      y = Dt |> select(follow1:follow6),
      n = Dt |> select(n) |> as.vector() |> unlist(),
       prior_mix = priorMix,
      prior_gamma = c(1,1)
      
      )
    
    # This runs so fast that it's not worth spinning up more than one core
    Fit<-sampling(model,data=d,seed=42,cores=1)
    
    FitSum<-summary(Fit)$summary |>
      data.frame() |>
      rownames_to_column(var = "par") |>
      mutate(treatment=tt)
    
    Estimates<-rbind(
      Estimates,FitSum
    )
  }
  
  Estimates |> write.csv(file)
}

# model with heterogeneous trembles

file<-"Presentation/SFEM/Estimates_SFEMheterogeneous.csv"

if (!file.exists(file)) {
  
  Estimates<-tibble()
  
  for (tt in 1:6) {
    print(paste("estimating heterogeneous trembles model",tt))
    
    model<-"Presentation/SFEM/SFEMheterogeneous.stan" |>
      stan_model()
    
    Dt<- AggData %>% filter(Treatment==tt)
    
    d = list(
      
      N=dim(Dt)[1],
      S=6,
      y = Dt |> select(follow1:follow6),
      n = Dt |> select(n) |> as.vector() |> unlist(),
      prior_mix = priorMix,
      prior_mu = c(0,1),
      prior_tau = 0.05
      
    )
    
    Fit<-sampling(model,data=d,seed=42,
                  iter = 10000,
                  control = list(adapt_delta=0.9999)
                  )
    
    FitSum<-summary(Fit)$summary |>
      data.frame() |>
      rownames_to_column(var = "par") |>
      mutate(treatment=tt)
    
    Estimates<-rbind(
      Estimates,FitSum
    )
  }
  
  Estimates |> write.csv(file)
}


# USING ALL OF THE DATA IN ONE GO

MODELS<-c("SFEMhomogeneous_pooled","SFEMhomogeneous_treatment","SFEMhomogeneous_strategy",
  "SFEMheterogeneous_pooled","SFEMheterogeneous_treatment","SFEMheterogeneous_strategy")
library(bridgesampling)
library(mnorm)

set.seed(42)

bs<-list()

for (mm in MODELS) {
  
  file<-paste0("Presentation/SFEM/Estimates_",mm,".rds")
    
    print(paste("estimating",mm))
    
    model <-paste0("Presentation/SFEM/",mm,".stan") |>
      stan_model()
    
    d = list(
      
      N=dim(AggData)[1],
      S=6,
      y = AggData |> select(follow1:follow6),
      n = AggData |> select(n) |> as.vector() |> unlist(),
      
      nTreatments = 6,
      treatmentID = AggData$Treatment,
      
      prior_mix = priorMix,
      prior_mu = c(0,1),
      prior_tau = 0.05,
      prior_gamma = c(1,1),
      
      nz = 100,
      z = halton(100)[,1] |> qnorm() |> as_vector()
      
    )
    
    if (mm=="SFEMheterogeneous_strategy") {
      
      Fit<- model |> 
        sampling(data=d,seed=42,
                 control = list(adapt_delta = 0.999),
                 iter = 4000
                 )
      
    } else {
      Fit<- model |> 
        sampling(data=d,seed=42)
    }
    
    
    
    bs[[mm]]<-bridge_sampler(Fit)
    
    Fit |>
      saveRDS(file)
    
    
  
}

bs |> saveRDS("Presentation/SFEM/logml.rds")



