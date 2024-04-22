data {
  int<lower=0> n; // number of observations
  
  int Left[n]; // indicates whether (=1) or not (=0) the Left lottery was chosen
  
  matrix[n,3] prizes; // prizes
  matrix[n,2] prizeRange; // minimum and maximum prize
  matrix[n,3] probL; // probabilities for left lottery
  matrix[n,3] probR; // probabilities for right lottery
  
  
  vector[2] prior_r; // prior for r (normal: mean, sd)
  vector[2] prior_lambda; // prior for lambda (log-normal, transformed mean, sd)
  
}

transformed data {
  
}


parameters {
  
  real r; // CRRA parameter in utility function
  real<lower=0> lambda; // logit choice precision
  
}

model {
  
  // priors
  r ~ normal(prior_r[1],prior_r[2]);
  lambda ~ lognormal(prior_lambda[1],prior_lambda[2]);
  
  // utility of each prize
  matrix[n,3] U = pow(prizes,1.0-r)/(1.0-r);
  
  // contextual utility normalization
  vector[n] normalization = pow(prizeRange[,2],1.0-r)/(1.0-r)
                              -pow(prizeRange[,1],1.0-r)/(1.0-r);
  
  // Expected utility of left lottery
  vector[n] UL = (probL.*U)*rep_vector(1.0,3);
  // expected utility of right lottery
  vector[n] UR = (probR.*U)*rep_vector(1.0,3);
  
  // normalized utility difference
  vector[n] DU = (UL-UR)./normalization;
  
  
  // likelihood contribution
  Left ~ bernoulli_logit(lambda * DU);

  
}

generated quantities {
  
  real CE = pow(0.5*pow(10.0,1.0-r)+0.5*pow(50.0,1.0-r),
                  1.0/(1.0-r)
                  );
                  
  real pL = inv_logit(
              lambda*(
                (0.5*pow(10.0,1.0-r)+0.5*pow(50.0,1.0-r)-pow(30.0,1.0-r))/
                (pow(50.0,1.0-r)-pow(10.0,1.0-r))
              )
  );
  
  real DCEprobchoice;
  
  { 
    real UL = 0.5*pow(10.0,1.0-r)/(1.0-r)+0.5*pow(50.0,1.0-r)/(1.0-r);
    real UR = pow(30.0,1.0-r)/(1.0-r);
    real Up = pL*UL+(1.0-pL)*UR;
    
    DCEprobchoice = pow((1.0-r)*fmax(UL,UR),1.0/(1.0-r))
                      - pow((1.0-r)*Up,1.0/(1.0-r))
                      ;
  }
  
}

