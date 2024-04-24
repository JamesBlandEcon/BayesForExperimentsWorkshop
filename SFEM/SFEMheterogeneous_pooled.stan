// SFEM with distribution of errors not varying by treatment or strategy
data {
  int<lower=0> N; // number of participants
  int<lower=0> S; // number of strategies
  matrix[N,S] y; // number of times a strategy is followed
  vector[N] n; // number of actions taken
  
  int nTreatments;
  int treatmentID[N];
  
  row_vector<lower=0>[S] prior_mix; // Dirichlet prior
  
  vector[2] prior_mu; // mean and sd of normal
  real<lower=0> prior_tau; // half cauchy
}

parameters {
  simplex[S] mix[nTreatments]; // mixing probabilities
  
  real mu; // mean tremble probability (transformed)
  real<lower=0> tau; // sd tremble probability (transformed)
  
  vector[N] z;
  
}

transformed parameters {
  
  // transform z into participant-specific tremble probabilities
  vector[N] gamma = 0.5*Phi_approx(mu+z*tau);
  
}

model {
  
  // hierarchical structure
  target+=std_normal_lpdf(z);
  
  
  
  // likelihood contribution
  for (ii in 1:N) {
    
      row_vector[S] like_strg = to_row_vector(y[ii,]).*rep_row_vector(log(1.0-gamma[ii]),S)
                              +to_row_vector(n[ii]-y[ii,]).*rep_row_vector(log(gamma[ii]),S)
                              +log(mix[treatmentID[ii]]');
    
    target += log_sum_exp(like_strg);
    
  }
  // priors
  target += dirichlet_lpdf(mix | prior_mix);
  target += normal_lpdf(mu | prior_mu[1],prior_mu[2]);
  target += cauchy_lpdf(tau | 0.0,prior_tau);
  
  
}

