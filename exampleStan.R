library(tidyverse)
library(rstan)
  # Tell R to use all of my computer's cores
  options(mc.cores = parallel::detectCores())
  # Stan model gets saved when it is compiled
  rstan_options(auto_write = TRUE)

# Simulate some data
set.seed(42)
y <- rnorm(100,mean=1,sd=3)

# compile the Stan program
model<-"Presentation/exampleStan.stan" |>
  stan_model()

# Set up the data to pass to Stan
d<-list(
  N = length(y),
  y = y,
  prior_mu = c(0,10),
  prior_sigma = 10
)

# Simulate the posterior
Fit<-model |>
  sampling(data=d,seed=42)

# Display a summary of the posterior simulation
Fit |> print()
