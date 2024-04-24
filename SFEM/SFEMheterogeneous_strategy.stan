// SFEM with distribution of errors varying by strategy
data {
  int<lower=0> N; // number of participants
  int<lower=0> S; // number of strategies
  matrix[N,S] y; // number of times a strategy is followed
  vector[N] n; // number of actions taken
  
  int nTreatments;
  int treatmentID[N];
  
  row_vector<lower=0>[S] prior_mix; // Dirichlet prior
  
  vector[2] prior_mu; // mean and sd of normal
  real<lower=0> prior_tau; // half cauchy'
  
  int nz;
  vector[nz] z; // standard normals 
  
}

parameters {
  simplex[S] mix[nTreatments]; // mixing probabilities
  
  row_vector[S] mu; // mean tremble probability (transformed)
  row_vector<lower=0>[S] tau; // sd tremble probability (transformed)

  
}

transformed parameters {
  
  // transform z into participant-specific tremble probabilities
  //matrix[N,S] gamma = 0.5*Phi_approx(rep_matrix(mu,N)+z.*rep_matrix(tau,N));
  
}

model {
  
  
  matrix[N,S] like_strg = rep_matrix(0.0,N,S);
  
  for (ss in 1:S) {
    
    matrix[N,nz] GAMMA = rep_matrix(0.5*Phi_approx(mu[ss]+tau[ss]*z'),N);
    
    like_strg[,ss] += log((pow(1.0-GAMMA,rep_matrix(y[,ss],nz)).*pow(GAMMA,rep_matrix(n-y[,ss],nz)))*rep_vector(1.0/(0.0+nz),nz));
    
  }
  
  for (ii in 1:N) {
    
    target += log_sum_exp(log(mix[treatmentID[ii]]')+like_strg[ii,]);
    
  }
  // priors
  target += dirichlet_lpdf(mix | prior_mix);
  target += normal_lpdf(mu | prior_mu[1],prior_mu[2]);
  target += cauchy_lpdf(tau | 0.0,prior_tau);
  
  
}

