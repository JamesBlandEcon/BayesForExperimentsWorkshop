data {
  int<lower=0> N; // number of participants
  int<lower=0> S; // number of strategies
  matrix[N,S] y; // number of times a strategy is followed
  vector[N] n; // number of actions taken
  
  row_vector<lower=0>[S] prior_mix; // Dirichlet prior
  vector<lower=0>[2] prior_gamma; // Beta prior (truncated to (0,0.5))
}

parameters {
  simplex[S] mix; // mixing probabilities
  real<lower=0,upper=0.5> gamma; // tremble probability
}

model {
  
  // likelihood contribution
  matrix[N,S] like_strg = y .* rep_matrix(log(1.0-gamma),N,S)
                            +(rep_matrix(n,S)-y).*rep_matrix(log(gamma),N,S)
                            +rep_matrix(log(mix'),N);
  for (ii in 1:N) {
    target += log_sum_exp(like_strg[ii,]);
  }
  // priors
  mix ~ dirichlet(prior_mix);
  gamma ~ beta(prior_gamma[1],prior_gamma[2]);
  
  
}

