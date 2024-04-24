// SFEM with distribution of errors not varying by treatment or strategy
data {
  int<lower=0> N; // number of participants
  int<lower=0> S; // number of strategies
  matrix[N,S] y; // number of times a strategy is followed
  vector[N] n; // number of actions taken
  
  int nTreatments;
  int treatmentID[N];
  
  row_vector<lower=0>[S] prior_mix; // Dirichlet prior
  
  vector<lower=0>[2] prior_gamma;
}

parameters {
  simplex[S] mix[nTreatments]; // mixing probabilities
  
  vector<lower=0.0,upper=0.5>[nTreatments] gamma; // mean tremble probability (transformed)
  
}


model {
  
  
  
  // likelihood contribution
  for (ii in 1:N) {
    
      row_vector[S] like_strg = to_row_vector(y[ii,]).*rep_row_vector(log(1.0-gamma[treatmentID[ii]]),S)
                              +to_row_vector(n[ii]-y[ii,]).*rep_row_vector(log(gamma[treatmentID[ii]]),S)
                              +log(mix[treatmentID[ii]]');
    
    target += log_sum_exp(like_strg);
    
  }
  // priors
  target += dirichlet_lpdf(mix | prior_mix);
  target += beta_lpdf(gamma | prior_gamma[1],prior_gamma[2])-beta_lcdf(0.5|prior_gamma[1],prior_gamma[2]);
  
  
}

