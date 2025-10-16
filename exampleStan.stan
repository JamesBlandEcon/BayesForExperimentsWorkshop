data {
  int<lower=0> N; // number of observations in data
  vector[N] y; // data
  vector[2] prior_mu; // prior for parameter mu (mean and sd)
  real<lower=0> prior_sigma; // prior for sigma (half-cauchy)
}
parameters {
  real mu; // mean of distribution
  real<lower=0> sigma; // standard deviation of distribution
}
model {
  y ~ normal(mu, sigma); // likelihood contribution
  mu ~ normal(prior_mu[1],prior_mu[2]); // prior for mu
  sigma ~ cauchy(0.0,prior_sigma); // prior for sigma
}

