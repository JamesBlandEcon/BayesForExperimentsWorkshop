
/* Homogeneous quantal response equilibrium model for the Volunteer's Dilemma
*/

functions {
  
 
  /* Returns the log-QRE probabilities
  
  */
  vector lpQRE(
    real lambda,
    data int n,
    data real V, data real c, data real L,
    data real tol,
    data int maxiter
  ) {
    
    int it = 0;

    // initial conditions
    real l = 0;
    
    real dist = 1e12;
    
    while (dist>tol) {
      
      real sigmaV = inv_logit(l);
      real sigmaN = inv_logit(-l);

      real H = l-lambda*(
        V-c-V*(1.0-pow(sigmaN,n-1.0))-L*pow(sigmaN,n-1.0)
      );
      
      real Hl = 1.0-lambda*sigmaV*sigmaN*(
        -V*(n-1.0)*pow(sigmaN,n-2.0)
        +L*(n-1.0)*pow(sigmaN,n-2.0)
      );
      
      real dl = -H/Hl;
      
      l = l+dl;
      
      dist = abs(dl);
      
      it +=1;
      
      if (it>=maxiter) {
        print("maxiter reached");
        break;
      }

    }
    
    return [log(inv_logit(l)),log(inv_logit(-l))]';
    
    
      
  }
    
    
    
  
}

data {
  int<lower=0> N; // Number of participants
  int<lower=0> Volunteer[N];
  int<lower=0> count[N];
  int<lower=2> GroupSize[N];
  
  // Game parameters
  real V; // Benefit of volunteering
  real c; // Cost of volunteering
  real L; // Benefit if nobody volunteers
  
  real prior_lambda[2]; // lognormal prior
  
  
  real<lower=0> tol; // tolerance for the corrector step
  int maxiter;
  
}

parameters {
  // logit choice precision
  real<lower=0> lambda;
  
}

model {
  
  
  for (ii  in 1:N) {
    vector[2] lp = lpQRE(
     lambda,
     GroupSize[ii],// Group size
     V,  c,  L,
     tol,maxiter
    );
  
    //target+=Volunteer[ii]*lp[1]+(count[ii]-Volunteer[ii])*lp[2];
  
    Volunteer[ii] ~ binomial_logit(count[ii],lp[1]-lp[2]);
  
  
  
  }
  
  lambda ~ lognormal(prior_lambda[1],prior_lambda[2]);
  
}

generated quantities {
  vector[N] pV;
  
  for (ii in 1:N) {
    vector[2] lp = lpQRE(
     lambda,
    GroupSize[ii],// Group size
    V, c,  L,
    tol,maxiter
  );
    
    pV[ii] = exp(lp[1]);
  }
}

