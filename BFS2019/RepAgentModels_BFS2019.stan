
data {
  int<lower=0> n; // number of observations
  vector[n] self_x; // payoff to self with allocation x
  vector[n] other_x; // payoff to other with allocation x
  vector[n] self_y; // payoff to self with allocation y
  vector[n] other_y; // payoff to other with allocation y
  int choice_x[n]; // =1 iff allocation x is chosen
  
  real prior_lambda[2];
  real prior_alpha[2];
  real prior_beta[2];
}

transformed data {
  vector[n] dX; // disadvantageous inequality at allocation x
  vector[n] dY; // disadvantageous inequality at allocation y
  vector[n] aX; // advantageous inequality at allocation X
  vector[n] aY; // advantageous inequality at allocation X
  
  dX = fmax(0,other_x-self_x);
  dY = fmax(0,other_y-self_y);
  aX = fmax(0,self_x-other_x);
  aY = fmax(0,self_y-other_y);
}
parameters {
  real alpha;
  real beta;
  real<lower=0> lambda;
}
model {
  
  vector[n] Ux; // utility of allocation x
  vector[n] Uy; // utility of allocation y
  
  Ux = self_x-alpha*dX-beta*aX;
  Uy = self_y-alpha*dY-beta*aY;
  
  // likelihood
  choice_x ~ bernoulli_logit(lambda*(Ux-Uy));
  
  // priors
  alpha~normal(prior_alpha[1],prior_alpha[2]);
  beta ~normal(prior_beta[1],prior_beta[2]);
  lambda ~ lognormal(prior_lambda[1],prior_lambda[2]);
  
}

