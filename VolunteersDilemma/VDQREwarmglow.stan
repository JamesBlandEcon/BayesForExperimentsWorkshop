
/* Warm glow quantal response equilibrium model for the Volunteer's Dilemma
*/

functions {
  
  
  /* Here I am using "PBR" (i.e. probabilistic best response) to 
  distinguish between \sigma and \bar\sigma_V
  */
  
  vector PBR(
    real lambda, vector delta, real sigmaV,
    data real V, data real c, data real L, data real n
  ) {
    
    return inv_logit(lambda*(
      V-c+delta-V*(1.0-pow(1.0-sigmaV,n-1.0))-L*pow(1.0-sigmaV,n-1.0)
    ));
    
  }


 // derivative wrt sigmaV
 vector PBR_sigmaV(
    real lambda, vector delta, real sigmaV,
    data real V, data real c, data real L, data real n
  ) {
    return -lambda*(V-L)*(n-1.0).*pow(1.0-sigmaV,n-2.0)
    *PBR(lambda,delta,sigmaV,V,c,L,n) .*(1.0-PBR(lambda,delta,sigmaV,V,c,L,n));
  }
  
  
  // Zero condition for equilibrium
  real Hfun(real l,
    real lambda, vector delta,
    data real V, data real c, data real L, data real n
    ) {
      real sigmaV = inv_logit(l);
      real meanPBR = mean(PBR(lambda,delta,sigmaV,V,c,L,n));
      return l - log(meanPBR)+log(1.0-meanPBR);
    }
  
  // derivative of zero condition
  real Hl(real l,
    real lambda, vector delta,
    data real V, data real c, data real L, data real n
    ) {
      real sigmaV = inv_logit(l);
      real meanPBR = mean(PBR(lambda,delta,sigmaV,V,c,L,n));
      return 1.0-sigmaV*(1.0-sigmaV)/(meanPBR*(1.0-meanPBR))*mean(PBR_sigmaV(lambda,delta,sigmaV,V,c,L,n));
    }
  
  // Compute QRE using constant-lambda corrector steps
  real lpQRE(
    real lambda, vector delta,
    data real V, data real c, data real L, data real n,
    data real tol, data int maxiter
  ) {
    
    // start at l=0 (sigmaV=0.5)
    real l = 0.0;
    
    for (it in 1:maxiter) {
      
      real dl = -Hfun(l,lambda,delta,V,c,L,n)/Hl(l,lambda,delta,V,c,L,n);
      
      l = l+dl;
      
      if (abs(dl)<tol) {
        break;
      }
      
      if (it==maxiter) {
        print("Maximum iterations error, lambda = ",lambda, ", dl = ",abs(dl));
      }
      
    }
    
    return inv_logit(l);
    
  }
  
  
  
  
  
  
}

data {
  int<lower=0> N; // Number of participants
  int<lower=0> Volunteer[N];
  int<lower=0> count[N];
  
  int<lower=0> nTreatments;
  vector[nTreatments] GroupSize;
  
  int<lower=0> GameID[N];
  
  
  // Game parameters
  real V; // Benefit of volunteering
  real c; // Cost of volunteering
  real L; // Benefit if nobody volunteers
  
  real prior_lambda[2]; // lognormal prior
  real prior_mu[2];
  real prior_tau;
  
  
  
  int<lower=0> maxiter;
  real<lower=0> tol; // tolerance for the corrector step
  
  int<lower=1> nSim; // simulation size for Monte Carlo integration
  vector<lower=-1.0,upper=1.0>[nSim] deltaZ; // standard normals truncated to be between -1 and 1
  
}

parameters {
  // logit choice precision
  real<lower=0> lambda;
  
  // Distribution of delta
  real mu;
  real<lower=0> tau;
  
  vector<lower = mu-tau,upper=mu+tau>[N] theta_i;
  
  
}

model {
  
  
  
  vector[nSim] thetaSim = mu+tau*deltaZ;
  
  vector[nTreatments] sigmaV;
  
  for (tt  in 1:nTreatments) {
    sigmaV[tt] = lpQRE(
    lambda,thetaSim,
    V,c,L,GroupSize[tt],
    tol,maxiter
  );
  
  }
  
  Volunteer ~ binomial_logit(count,lambda*(
      V-c+theta_i-V*(1.0-pow(1.0-sigmaV[GameID],GroupSize[GameID]-1.0))-L*pow(1.0-sigmaV[GameID],GroupSize[GameID]-1.0)
    ));
  
  theta_i ~ normal(mu,tau);
  
  lambda ~ lognormal(prior_lambda[1],prior_lambda[2]);
  mu ~ normal(prior_mu[1],prior_mu[2]);
  tau ~ cauchy(0,prior_tau);
  
}

generated quantities {
  
  
  vector[nTreatments] sigmaV;
  {
  vector[nSim] thetaSim = mu+tau*deltaZ;
  for (tt  in 1:nTreatments) {
    sigmaV[tt] = lpQRE(
    lambda,thetaSim,
    V,c,L,GroupSize[tt],
    tol,maxiter
  );
  
  }
  }
  
}

