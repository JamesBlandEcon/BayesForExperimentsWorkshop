// SFEM with distribution of errors not varying by treatment or strategy
// Truncated beta specification
data {
  int<lower=0> N; // number of participants
  int<lower=0> S; // number of strategies
  matrix[N,S] y; // number of times a strategy is followed
  vector[N] n; // number of actions taken
  
  int nTreatments;
  int treatmentID[N];
  
  row_vector<lower=0>[S] prior_mix; // Dirichlet prior
  
  vector<lower=0>[2] prior_p; // Beta prior
  real<lower=0> prior_kappa; // half Cauchy
}

parameters {
  simplex[S] mix[nTreatments]; // mixing probabilities
  
  
  real<lower=0,upper=1> p;
  real<lower=0> kappa;
  
  
}

transformed parameters {
  
  // transform z into participant-specific tremble probabilities
  real a = p.*kappa;
  real b = (1.0-p).*kappa;
  
}

model {
  

  // likelihood contribution
  
  
  
  
  
  for (ii in 1:N) {
      
      row_vector[S] like_strg = log(mix[treatmentID[ii]]');
      
      
      
      
      
      for (ss in 1:S) {
        like_strg[ss] += log(inc_beta(n[ii]-y[ii,ss]+a,y[ii,ss]+b,0.5))+lbeta(n[ii]-y[ii,ss]+a,y[ii,ss]+b)
              -log(inc_beta(a,b,0.5))-lbeta(a,b);
      }
    
    target += log_sum_exp(like_strg);
    
  }
  
  // priors
  target += dirichlet_lpdf(mix | prior_mix);
  target += beta_lpdf(p | prior_p[1],prior_p[2]);
  target += cauchy_lpdf(kappa | 0.0,prior_kappa);
  
  
  
}

