data {
  int<lower=0> N; // number of participants
  int<lower=0> S; // number of strategies
  matrix[N,S] y; // number of times a strategy is followed
  vector[N] n; // number of actions taken
  
  row_vector<lower=0>[S] prior_mix; // Dirichlet prior
  
  vector[2] prior_mu; // mean and sd of normal
  real<lower=0> prior_tau; // half cauchy
}

parameters {
  simplex[S] mix; // mixing probabilities
  
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
  z ~ std_normal();
  
  
  
  // likelihood contribution
  matrix[N,S] like_strg = y .* rep_matrix(log(1.0-gamma),S)
                            +(rep_matrix(n,S)-y).*rep_matrix(log(gamma),S)
                            +rep_matrix(log(mix'),N);
  for (ii in 1:N) {
    target += log_sum_exp(like_strg[ii,]);
  }
  // priors
  mix ~ dirichlet(prior_mix);
  
  mu ~ normal(prior_mu[1],prior_mu[2]);
  tau ~ cauchy(0.0,prior_tau);
  
  
}

