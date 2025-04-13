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
  
  
  row_vector<lower=0,upper=1>[S] p;
  row_vector<lower=0>[S] kappa;
  
  
}

transformed parameters {
  
  // transform z into participant-specific tremble probabilities
  row_vector[S] a = p.*kappa;
  row_vector[S] b = (1.0-p).*kappa;
  
}

model {
  

  // likelihood contribution
  
  
  
  
  
  for (ii in 1:N) {
      
      row_vector[S] like_strg = log(mix[treatmentID[ii]]');
      
      
      
      
      
      for (ss in 1:S) {
        like_strg[ss] += log(inc_beta(n[ii]-y[ii,ss]+a[ss],y[ii,ss]+b[ss],0.5))+lbeta(n[ii]-y[ii,ss]+a[ss],y[ii,ss]+b[ss])
              -log(inc_beta(a[ss],b[ss],0.5))-lbeta(a[ss],b[ss]);
      }
    
    target += log_sum_exp(like_strg);
    
  }
  
  // priors
  target += dirichlet_lpdf(mix | prior_mix);
  target += beta_lpdf(p | prior_p[1],prior_p[2]);
  target += cauchy_lpdf(kappa | 0.0,prior_kappa);
  
  
  
}

