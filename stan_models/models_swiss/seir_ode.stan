functions {
  vector sir( real t, 
              vector y, 
              vector theta, 
              array[] real x_r, 
              array[] int x_i) {
      
      vector[4] dy_dt;
      
      real N = x_i[1];
      
      real beta = theta[1];
      real gamma = theta[2];
      real alpha = theta[3];
      real i0 = theta[4];
      real e0 = theta[5];
      
      array[4] real init = {N - i0 - e0, e0, i0, 0};
      
      real S = y[1] + init[1];
      real E = y[2] + init[2];
      real I = y[3] + init[3];
      real R = y[4] + init[4];      
      
      
      dy_dt[1] = -beta * I * S / N;
      dy_dt[2] = beta * I * S / N - alpha * E;
      dy_dt[3] = alpha * E - gamma * I;
      dy_dt[4] =  gamma * I;
      
      return dy_dt;
  }
}
data {
  int<lower=1> n_days;
  //vector[3] y0;
  real t0;
  array[n_days] real ts;
  int N;
  array[n_days] int cases;
}
transformed data {
  real x_r[0];
  int x_i[1] = { N };
}
parameters {
  real<lower=0> gamma;
  real<lower=0> beta;
  real<lower=0> alpha; // rate of exposure 
  real<lower=0> phi_inv;
  real<lower=0, upper=1> p_rep; # proportion of infected reported
  real<lower=0> e0; // number of initialll exposed
  real<lower=0> i0; // number of initially infected
}
transformed parameters{
  array[n_days] vector[3] y;
  array[n_days-1] real incidence;
  real phi = 1. / phi_inv;
  
  {
    vector[5] theta;
    theta[1] = beta;
    theta[2] = gamma;
    theta[3] = alpha;
    theta[4] = i0;
    theta[5] = e0;

    y = ode_rk45(sir, rep_vector(0.0, 4), t0, ts, theta, x_r, x_i);
  }
  for (i in 1:n_days-1){
    incidence[i] =  -(y[i+1, 2] - y[i, 2] + y[i+1,1] - y[i,1]) * p_rep; //S(t+1) - S(t) + E(t+1) - S(t)
  }
}
model {
  //priors
  beta ~ normal(2, 1);
  gamma ~ normal(0.4, 0.5);
  phi_inv ~ exponential(5);
  alpha ~ normal(0.4, 0.5);
  p_rep ~ beta(1, 2);
  i0 ~ normal(0, 10);
  e0 ~ normal(0, 10);
  
 //likelihood
  cases[1:(n_days-1)] ~ neg_binomial_2(incidence, phi);
}
generated quantities {
  real R0 = beta / gamma;
  real recovery_time = 1 / gamma;
  array[n_days-1] real pred_cases;
  pred_cases = neg_binomial_2_rng(incidence, phi);
}

