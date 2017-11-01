
# first source file funs-copy.R

# load libraries
library(data.table)
library(ggplot2)
library(rjags)



# load data
trajectory <- fread("metropolis_trajectory.csv")
setnames(trajectory, c("p", "beta", "user_ids"))

covariates <- fread("covariates.csv")


#trajectory and hierarchical data user id union
trajectory[,keep := rbinom(dim(trajectory)[1],1,0.05)]
trajectory[,mean(keep)]

#had to "thin-out" trajectory in order to make run faster and cope with memory requirements
trajectory <- trajectory[keep == 1]

#model
p <- trajectory[,p]
beta <- trajectory[,beta]

p <- log(p/(1-p))
beta <- log(beta)


index <- trajectory[,user_ids]

discounter <- covariates[,discounter]
loyalty <- covariates[,loyalty]
mean_income <- covariates[,mean_income]
mean_transaction <- covariates[,mean_transaction]
preference <- covariates[,preference]
public_transport <- covariates[,public_transport]

N <- dim(trajectory)[1]
P <- trajectory[,uniqueN(user_ids)]



# create jags data list
data.jags <- list(p = p, beta = beta, 
                  N = N, P = P, 
                  index = index,
                  discounter = discounter, loyalty = loyalty, mean_income = mean_income, mean_transaction = mean_transaction, 
                  preference = preference, public_transport = public_transport)

# create inits
inits.jags <- list(".RNG.name" = "base::Wichmann-Hill",
                   ".RNG.seed" = 111)

# jags model string
# currently prior of variances differently implemented than illustrated by Chen & Steckel
modelString <- 
  "model{
#Likelihood
for (i in 1:N){
p[i] ~ dnorm(theta_p[index[i]], tau_p)
beta[i] ~ dnorm(theta_beta[index[i]], tau_beta) 
}
for (j in 1:P){
theta_p[j] <- e_0 + e_1 * loyalty[j] +  e_2 * discounter[j]+ e_3 * preference[j] + error_p[j]
theta_beta[j] <- d_0 + d_1 * mean_transaction[j] + d_2 * mean_income[j] + d_3 * public_transport[j] + d_4 *  loyalty[j] + error_beta[j]
real_p[j] <- exp(theta_p[j])/(1 + exp(theta_p[j]))
real_beta[j] <- exp(theta_beta[j])
error_p[j] ~ dnorm(0,0.1)
error_beta[j] ~ dnorm(0,0.1)
}
#Priors
d_0 ~ dnorm(0,0.01)
d_1 ~ dnorm(0,0.01)
d_2 ~ dnorm(0,0.01)
d_3 ~ dnorm(0,0.01)
d_4 ~ dnorm(0,0.01)
e_0 ~ dnorm(0,0.01)
e_1 ~ dnorm(0,0.01)
e_2 ~ dnorm(0,0.01)
e_3 ~ dnorm(0,0.01)
sigma_beta ~ dunif(0,100)
sigma_p ~ dunif(0,100)
tau_beta <- 1/pow(sigma_beta,2)
tau_p <- 1/pow(sigma_p,2)
}"


m <- jags.model(textConnection(modelString), data = data.jags, inits = inits.jags)

# set jags parameters
t.chains <- 1
t.iter <- 20000
t.burning <- 5000

#update(m, t.burning)
model <- coda.samples(m, c("theta_p", "theta_beta", 
                           "sigma_beta", "sigma_p", 
                           "real_beta", "real_p" ,
                           "e_0", "e_1", "e_2", "e_3",
                           "d_0","d_1", "d_2", "d_3", "d_4"), 
                      n.iter =  t.iter,
                      n.chains = t.chains)

plot(model[,c("d_1", "d_2")])
plot(model[,c("d_3", "d_4")])
plot(model[,c("e_1", "e_2", "e_3")])
plot(model[,c("d_0", "e_0")])
plot(model[,c("theta_p[1]", "theta_beta[1]")])
plot(model[,c("sigma_p", "sigma_beta")])

plot(model[,c("real_p[10]", "real_beta[10]")])
plot(model[,c("theta_p[10]", "theta_beta[10]")])
plot(model[,c("real_p[20]", "real_p[21]", "real_p[22]")])
plot(model[,c("real_beta[20]", "real_beta[21]", "real_beta[22]")])

#compare with covariates and true coefficient from data generation process
covariates

#true parameters from data generation in python
# theta_beta = 1 -0.2*mean_transaction + 0.0005*mean_income + 0.5*public_transport + 0.6*loyalty + np.random.normal(0.01, 0.01, People) 
# theta_p = -0.2 +0.4*loyalty - 0.8*discounter + preference + np.random.normal(0.01, 0.01, People)
