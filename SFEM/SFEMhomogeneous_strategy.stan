// SFEM with distribution of errors varying by strategy
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
  
  row_vector<lower=0,upper=0.5>[S] gamma;
  
}

transformed parameters {
  
}

model {
  
  
  matrix[N,S] like_strg = 
        y.*rep_matrix(log(1.0-gamma),N)
        +(rep_matrix(n,S)-y).*rep_matrix(log(gamma),N)
    ;
  
  for (ii in 1:N) {
    
    target += log_sum_exp(log(mix[treatmentID[ii]]')+like_strg[ii,]);
    
  }
  // priors
  target += dirichlet_lpdf(mix | prior_mix);
  target += beta_lpdf(gamma | prior_gamma[1],prior_gamma[2])-beta_lcdf(0.5|prior_gamma[1],prior_gamma[2]);
  
  
}

