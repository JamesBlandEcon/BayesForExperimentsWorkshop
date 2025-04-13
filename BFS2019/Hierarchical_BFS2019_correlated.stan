
data {
  int<lower=0> n; // number of observations
  vector[n] self_x; // payoff to self with allocation x
  vector[n] other_x; // payoff to other with allocation x
  vector[n] self_y; // payoff to self with allocation y
  vector[n] other_y; // payoff to other with allocation y
  int choice_x[n]; // =1 iff allocation x is chosen
  
  int id[n];
  int nParticipants;
  

  
  vector[2] prior_mu[3];
  real prior_tau[3];
  real prior_LKJ;
  
  // number of simulation steps to approximate demand
  int nsim; 
  // Endowment of tokens for demand
  int W;
  // number of prices to evaluate demand at
  int nrho;
  // prices to evaluate demand at
  vector[nrho] rho; 
  
  
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
  
  // some things used to produce demand 
  vector[W+1] X;
    for (xx in 1:(W+1)) {
      X[xx] = xx-1;
    }
  
  
  
}
parameters {
  
  vector[3] mu; // population mean
  vector<lower=0>[3] tau; // population standard deviation
  cholesky_factor_corr[3] L_Omega; // Cholesky factorization of the correlation matrix
  
  matrix[3,nParticipants] z; // standard normals determinig participants' parameters
  
}
model {
  
  vector[n] Ux; // utility of allocation x
  vector[n] Uy; // utility of allocation y
  vector[n] alpha;
  vector[n] beta;
  vector[n] lambda;
  
  matrix[3,nParticipants] theta;
  
  theta = mu*rep_row_vector(1.0,nParticipants)
                    +diag_pre_multiply(tau,L_Omega)*z;
  
  alpha  = theta[1,id]';
  beta   = theta[2,id]';
  lambda = exp(theta[3,id]');

  Ux = self_x-alpha .* dX-beta .* aX;
  Uy = self_y-alpha .* dY-beta .* aY;
  
  choice_x ~ bernoulli_logit(lambda .* (Ux-Uy));
  
  to_vector(z) ~ std_normal();
  
  for (pp in 1:3) {
    mu[pp] ~ normal(prior_mu[pp][1],prior_mu[pp][2]);
    tau[pp] ~ cauchy(0.0,prior_tau[pp]);
  }
  
   L_Omega ~ lkj_corr_cholesky(prior_LKJ);
}

generated quantities {
  matrix[3,3] Omega;
  matrix[3,nParticipants] theta;
  theta = mu*rep_row_vector(1.0,nParticipants)
                    +diag_pre_multiply(tau,L_Omega)*z;
  Omega = L_Omega*L_Omega';
  
  // store demand in this vector
  vector[nrho] demand;
  
  {
    // simulated standard normals
    matrix[3,nsim] zsim;
    for (rr in 1:3) {
      zsim[rr,] = to_row_vector(normal_rng(rep_vector(0.0,nsim),rep_vector(1.0,nsim)));
    }
    
    // simulate parameters
    matrix[3,nsim] thetasim;
    thetasim = mu*rep_row_vector(1.0,nsim)
                    +diag_pre_multiply(tau,L_Omega)*zsim;
                    
    vector[nsim] alphasim = thetasim[1,]';
    vector[nsim] betasim = thetasim[2,]';
    vector[nsim] lambdasim = exp(thetasim[3,]');
    
    // go through each value of rho 
    for (rr in 1:nrho) {
      
      // dump individual-level expected demand in here
      vector[nsim] Y;
      for (ss in 1:nsim) {
        
        // utility of choosing each allocation
        vector[W+1] U = X
            -alphasim[ss]*fmax(0.0,(W-X*(1.0+rho[rr]))/rho[rr])
            -betasim[ss]*fmax(0.0,(X*(1+rho[rr])-W)/rho[rr]);
        // probability of choosing each allocation
        vector[W+1] pr = softmax(lambdasim[ss]*U);
        // expected tokens kept
        real EX = sum(X.*pr);
        // expected tokens bought for the other participant
        Y[ss] = (W-EX)/rho[rr];
        
      }
      
      demand[rr] = mean(Y);
      
    }
    
    
    
  }
  
  
  
  
  
  
  
  
  
}

