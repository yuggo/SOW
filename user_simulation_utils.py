#    FOR THE WHOLE SCRIPT THE LOGIC APPLIED FOR P0 AND P1 IS:
#    P1 REPRESENTS THE PROBABILITY OF COMING TO OUR STORE AFTER HAVING COME LAST TIME AS WELL
#    P0 REPRESENTS THE PROBABILITY OF COMING TO OUR STORE IF LAST TIME THE CLIENT WENT ELSEWHERE

############ FUNCTIONS DESCRIPTION ##########
#   expit:  takes as input a numpy array NxK, applies a logistic sigmoid elementwise,
#           and gives as output a numpy array NxK.
#
#   logit:  takes as input a numpy array of strictly positive numbers NxK, applies a logit elementwise,
#           and gives as output a numpy array NxK.
#
#   to_int: takes as input a list of numbers and converts them to integer, and gives in output a list of the same
#           dimension where each element is now of the integer type
#
#   generate_people: takes as input a dictionary of parameters (pars), a seed, a standard error for the beta (error_beta)
#           and a standard error for the p (error_p). pars must contain:
#
#               - 'd_0', the intercept of the beta generating function
#               - 'd_1', a coefficient in the beta generating function
#               - 'd_2', a coefficient in the beta generating function
#               - 'd_3', a coefficient in the beta generating function
#               - 'e_0', the intercept of the p generating function
#               - 'e_1', a coefficient in the p generating function
#               - 'e_2', a coefficient in the p generating function
#               - 'e_3', a coefficient in the p generating function
#               - 'People', the number of people you want to generate
#
#           the output is a dataframe with one row for each of the people generated. The columns, which are the
#           variables that characterize the peoplr, are the following:
#
#               -'user', the code of the person
#               -'mean_transaction', the predictor associated to d_1
#               –'mean_income', the predictor associated to d_2
#               –'public_transport', the predictor associated to d_3
#               –'loyalty', the predictor associated to e_1
#               –'discounter', the predictor associated to e_2
#               –'preference', the predictor associated to e_3
#               –'p', the generated sow of the person
#               –'beta', the generated rate of the person
#
#   simulate_data:  takes as input the p parameter (share_observed) and beta parameter (rate_input) as well as the
#           total amount of observations to generate (observations_total) and gives as output a list in which the
#           first elements are the observed ipts and the second element is the time since the last purchase
#
#   repeated_simulation:    takes as input a vector of ps and betas, as well as the total number of observations
#           of ipts to generate for each person (observations_total) and a number of repetitions (rep).
#           The function returns a list of two elements, the first being rep simulations of each of the possible
#           combinations of ps and betas, and the second being the parameters given as input.
#   generate_people: takes as input a dictionary of parameters (pars), a seed, a standard error for the beta (error_beta)
#           and a standard error for the p (error_p). pars must contain:
#
#               - 'd_0', the intercept of the beta generating function
#               - 'd_1', a coefficient in the beta generating function
#               - 'd_2', a coefficient in the beta generating function
#               - 'd_3', a coefficient in the beta generating function
#               - 'e_0', the intercept of the p0 and p1 generating function
#               - 'e_1', a coefficient in the p0 generating function
#               - 'e_2', a coefficient in the p0 generating function
#               - 'e_3', a coefficient in the p1 generating function
#               - 'People', the number of people you want to generate
#
#           the output is a dataframe with one row for each of the people generated. The columns, which are the
#           variables that characterize the peoplr, are the following:
#
#               -'user', the code of the person
#               -'mean_transaction', the predictor associated to d_1
#               –'mean_income', the predictor associated to d_2
#               –'public_transport', the predictor associated to d_3
#               –'loyalty', the predictor associated to e_1
#               –'discounter', the predictor associated to e_2
#               –'preference', the predictor associated to e_3
#               –'p0', the generated probability(X(k) = 1|X(k-1) = 0)
#               –'p1', the generated probability(X(k) = 1|X(k-1) = 1)
#               –'beta', the generated rate of the person
#
#   simulate_data_3par: takes as input the p0 (probability of coming if last time the person went to another store)
#           and p1 (probability of coming if last time came to the store) parameters as well as a beta parameter,
#           it also takes as input Time_span (amount of time passed since the person was forst observed in the store)
#           and gives as output a list in which the first elements are the observed ipts and the second element
#           is the time since the last purchase.
#
#   repeated_simulation_3par:   takes as input a vector of ps and betas, as well as the time span for which the people
#           have been observed (Time_span) and a number of repetitions (rep).
#           The function returns a list of two elements, the first being rep simulations of each of the possible
#           combinations of ps and betas, and the second being the parameters given as input.
#

import math
import numpy as np
from scipy import stats
import pandas as pd
import random as rnd

def expit(x):
    '''takes as input a numpy array NxK, applies a logistic sigmoid elementwise,
     and gives as output a numpy array NxK.'''
    return(np.exp(x)/(1 + np.exp(x)))

def logit(x):
    '''takes as input a numpy array of strictly positive numbers NxK, applies a logit elementwise,
    and gives as output a numpy array NxK.'''
    return(np.log(x/(1-x)))

def to_int(list):
    '''takes as input a list of numbers and converts them to integer, and gives in output a list of the same
    dimension where each element is now of the integer type'''
    new_list = [int(i) for i in list]
    return(new_list)


def generate_people(pars, seed = 123, error_beta = 0.01, error_p = 0.01):
    '''takes as input a dictionary of parameters (pars), a seed, a standard error for the beta (error_beta)
    and a standard error for the p (error_p). pars must contain:

                - 'd_0', the intercept of the beta generating function
                - 'd_1', a coefficient in the beta generating function
                - 'd_2', a coefficient in the beta generating function
                - 'd_3', a coefficient in the beta generating function
                - 'e_0', the intercept of the p generating function
                - 'e_1', a coefficient in the p generating function
                - 'e_2', a coefficient in the p generating function
                - 'e_3', a coefficient in the p generating function
                - 'People', the number of people you want to generate

            the output is a dataframe with one row for each of the people generated. The columns, which are the
            variables that characterize the people, are the following:

                -'user', the index of the person
                -'mean_transaction', the predictor associated to d_1 (hypothetical
                –'mean_income', the predictor associated to d_2
                –'public_transport', the predictor associated to d_3
                –'loyalty', the predictor associated to e_1
                –'discounter', the predictor associated to e_2
                –'preference', the predictor associated to e_3
                –'p', the generated sow of the person
                –'beta', the generated rate of the person'''
    ###Generation of people and their characteristics

    np.random.seed(seed)
    N_people = pars['People']
    # beta predictors
    # assume beta (rate parameter) is determined by avg. transaction size
    # avg income and whether public transport was used
    # create this feature matrix
    # mean paid per transaction
    mean_transaction = np.random.normal(np.log(15), 0.02, N_people)
    mean_transaction = np.exp(mean_transaction)

    mean_income = np.random.normal(1800, 100, N_people)

    public_transport = to_int(np.random.binomial(1, 0.5, N_people))
    # reset seed for easier reproducability
    # p predictors
    # assume p (SOW) is driven by loyalty card membership and whether discounter is present
    # create this feature matrix
    # membership in loyalty
    loyalty = to_int(np.random.binomial(1, 0.6, N_people))
    # median income --> from beta
    # discounter
    discounter = to_int(np.random.binomial(1, 0.5, N_people))

    preference = np.array(stats.uniform.rvs(0, 1, size=N_people, random_state=123))


    theta_p = pars['e_0'] + pars['e_1'] * np.array(loyalty) + pars['e_2'] * np.array(discounter) + pars['e_3'] * preference
    p = expit(theta_p) + np.random.normal(0.01, error_p, N_people)

    theta_beta = pars['d_0'] + pars['d_1'] * mean_transaction + pars['d_2'] * mean_income + pars['d_3'] * np.array(public_transport)
    beta = np.exp(theta_beta) + np.random.normal(0.01, error_beta, N_people)

    labels = ['user', 'mean_transaction', 'mean_income','public_transport',
              'loyalty', 'discounter', 'preference', 'p', 'beta']
    # generate data
    input_parameters = pd.DataFrame(np.transpose([np.arange(1, N_people + 1), mean_transaction, mean_income,
                                                  public_transport, to_int(loyalty), discounter, preference, p, beta]),
                                    columns=labels)
    return(input_parameters)


def simulate_data(share_observed = 11/20, rate_input = 0.4, observations_total = 20):
    '''takes as input the p parameter (share_observed) and beta parameter (rate_input) as well as the
    total amount of observations to generate (observations_total) and gives as output a list in which the
    first elements are the observed ipts and the second element is the time since the last purchase'''
    obs_true = math.ceil(observations_total/share_observed)
    gen = np.random.gamma(2, scale=1/rate_input, size = obs_true)
    obs_seen = observations_total

    idx = list(range(1,obs_true+1))

    id_obs = np.sort(rnd.sample(idx, obs_seen))

    id_obs_0 = list(id_obs)
    id_obs_0.append(0)
    id_obs_0 = np.sort(id_obs_0)

    #Sum up all unobserved and last observed
    x = [sum([gen[j] for j in range(int(id_obs_0[i]),int(id_obs_0[i+1]))]) for i in range(obs_seen)]
    x = np.asarray(x)
    #how many day unobserved until end of period, needed for right censoring
    last = sum([gen[j] for j in range(int(id_obs_0[-1]),obs_true)])
    output = [x, last]

    return(output)

#Those take quite some time
def repeated_simulation(betas, ps, observations_total = 50, rep = 20):
    '''   takes as input a vector of ps and betas, as well as the total number of observations
          of ipts to generate for each person (observations_total) and a number of repetitions (rep).
          The function returns a list of two elements, the first being rep simulations of each of the possible
          combinations of ps and betas, and the second being the parameters given as input.'''
    results = []
    parameters = np.zeros([ps.shape[0]*betas.shape[0]*rep,2])
    for j in range(len(betas)):
        for k in range(len(ps)):
            observed_share_param = ps[k]
            rate_param = betas[j]
            print(rate_param, observed_share_param)
            for i in range(rep):
                x = simulate_data(observed_share_param,rate_param,observations_total)
                results.append(x)
                parameters[j*len(ps)*rep+k*rep+i,0] = observed_share_param
                parameters[j*len(ps)*rep+k*rep+i,1] = rate_param
    return([results, parameters])



#Now all the new functions that needed to be changed for the 3 parameters version
def generate_people_3par(pars, seed = 123, error_beta = 0.01, error_p0 = 0.01, error_p1 = 0.01):
    '''takes as input a dictionary of parameters (pars), a seed, a standard error for the beta (error_beta)
    and a standard error for the p (error_p). pars must contain:

               - 'd_0', the intercept of the beta generating function
               - 'd_1', a coefficient in the beta generating function
               - 'd_2', a coefficient in the beta generating function
               - 'd_3', a coefficient in the beta generating function
               - 'e_0', the intercept of the p0 and p1 generating function
               - 'e_1', a coefficient in the p0 generating function
               - 'e_2', a coefficient in the p0 generating function
               - 'e_3', a coefficient in the p1 generating function
               - 'People', the number of people you want to generate

           the output is a dataframe with one row for each of the people generated. The columns, which are the
           variables that characterize the peoplr, are the following:

               -'user', the code of the person
               -'mean_transaction', the predictor associated to d_1
               –'mean_income', the predictor associated to d_2
               –'public_transport', the predictor associated to d_3
               –'loyalty', the predictor associated to e_1
               –'discounter', the predictor associated to e_2
               –'preference', the predictor associated to e_3
               –'p0', the generated probability(X(k) = 1|X(k-1) = 0)
               –'p1', the generated probability(X(k) = 1|X(k-1) = 1)
               –'beta', the generated rate of the person'''
    ###Generation of people and their characteristics
    np.random.seed(seed)
    N_people = pars['People']
    # beta predictors
    # assume beta (rate parameter) is determined by avg. transaction size
    # avg income and whether public transport was used
    # create this feature matrix
    # mean paid per transaction
    mean_transaction = np.random.normal(np.log(15), 0.02, N_people)
    mean_transaction = np.exp(mean_transaction)

    mean_income = np.random.normal(1800, 100, N_people)

    public_transport = to_int(np.random.binomial(1, 0.5, N_people))
    # reset seed for easier reproducability
    # p predictors
    # assume p (SOW) is driven by loyalty card membership and whether discounter is present
    # create this feature matrix
    # membership in loyalty
    loyalty = to_int(np.random.binomial(1, 0.6, N_people))
    # median income --> from beta
    # discounter
    discounter = to_int(np.random.binomial(1, 0.5, N_people))

    preference = np.array(stats.uniform.rvs(0, 1, size=N_people, random_state=123))


    theta_p0 = pars['e_0'] + pars['e_3'] * preference
    p0 = expit(theta_p0) + np.random.normal(0.01, error_p0, N_people)

    theta_p1 = pars['e_0'] + pars['e_1'] * np.array(loyalty) + pars['e_2'] * np.array(discounter)
    p1 = expit(theta_p1) + np.random.normal(0.01, error_p1, N_people)

    theta_beta = pars['d_0'] + pars['d_1'] * mean_transaction + pars['d_2'] * mean_income + pars['d_3'] * np.array(public_transport)
    beta = np.exp(theta_beta) + np.random.normal(0.01, error_beta, N_people)

    labels = ['user', 'mean_transaction', 'mean_income','public_transport',
              'loyalty', 'discounter', 'preference', 'p0', 'p1', 'beta']
    # generate data
    input_parameters = pd.DataFrame(np.transpose([np.arange(1, N_people + 1), mean_transaction, mean_income,
                                                  public_transport, to_int(loyalty), discounter, preference, p0, p1, beta]),
                                    columns=labels)
    return(input_parameters)

#Parameters = dict(d_0 = +1, d_1 = -0.2, d_2 = +0.0005, d_3 = +0.5,
#                  e_0 = -0.2, e_1 = +0.4, e_2 = -0.8, e_3 = +1,
#                  People = 50, number_simulations = 5000)

#Peeps = generate_people(Parameters, seed = 123, error_beta = 0.01, error_p1 = 0.01, error_p2 = 0.01)
#plt.hist(Peeps['p1'])
#plt.hist(Peeps['p2'])
#plt.hist(Peeps['beta'])



#pi0 is the probability of coming if didn't come last time
#pi1 is the probability of coming if he came last time
#Time_span is in days, default to one year
#If pi0 = pi1 we have the classical generator we always used, with p = pi1 = pi2 = sow
#Else we should have sow = p1/(1 - p0 + p1)
def simulate_data_3par(pi0 = 0.5, pi1 = False, beta = 0.4, Time_span = 365.0):
    '''takes as input the p0 (probability of coming if last time the person went to another store)
    and p1 (probability of coming if last time came to the store) parameters as well as a beta parameter,
    it also takes as input Time_span (amount of time passed since the person was forst observed in the store)
    and gives as output a list in which the first elements are the observed ipts and the second element
    is the time since the last purchase.'''
    if pi1 == False:
        pi1 = pi0
    T = Time_span
    alpha = 2
    Xs = []
    total_trips = 0
    while np.sum(Xs)<=T:
        new_trip_time = np.random.gamma(alpha, scale=1/beta, size = 1)[0]
        Xs.append(new_trip_time)
        total_trips += 1

    total_trips = total_trips - 1
    Xs = list(Xs[:-1])
    Init_last = T - np.sum(Xs)

    U = np.random.uniform(size = total_trips)
    seen = np.zeros(total_trips)
    customer_comes = 1
    seen[0] = customer_comes

    our_obs = []
    p = [pi0, pi1]
    time_pile = 0
    for i in range(total_trips):
        customer_comes = p[int(customer_comes)] > U[i]
        seen[i] = customer_comes
        if customer_comes:
            time = time_pile + Xs[i]
            our_obs.append(time)
            time_pile = 0
        else:
            time_pile += Xs[i]
    our_obs = np.array(our_obs)
    Xs = np.array(Xs)
    Last = Init_last + time_pile
    output = [our_obs, Last]
    return(output)

#Those take quite some time
def repeated_simulation_3par(p0s, p1s, betas, time_span = 365.0, rep = 20):
    '''takes as input a vector of ps and betas, as well as the time span for which the people
    have been observed (Time_span) and a number of repetitions (rep).
    The function returns a list of two elements, the first being rep simulations of each of the possible
    combinations of ps and betas, and the second being the parameters given as input.'''
    results = []
    p0s = np.array(p0s)
    p1s = np.array(p1s)
    betas = np.array(betas)
    parameters = np.zeros([p0s.shape[0] * p1s.shape[0] * betas.shape[0] * rep,3])
    Idx = 0
    for j in range(len(p0s)):
        p0 = p0s[j]
        for k in range(len(p1s)):
            p1 = p1s[k]
            for m in range(len(betas)):
                beta = betas[m]
                for i in range(rep):
                    x = simulate_data_3par(p0, p1, beta, time_span)
                    results.append(x)
                    parameters[Idx, 0] = p0
                    parameters[Idx, 1] = p1
                    parameters[Idx, 2] = beta
                    Idx += 1
    return([results, parameters])
