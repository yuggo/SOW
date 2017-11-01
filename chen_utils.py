#    FOR THE WHOLE SCRIPT THE LOGIC APPLIED FOR P0 AND P1 IS:
#    P1 REPRESENTS THE PROBABILITY OF COMING TO OUR STORE AFTER HAVING COME LAST TIME AS WELL
#    P0 REPRESENTS THE PROBABILITY OF COMING TO OUR STORE IF LAST TIME THE CLIENT WENT ELSEWHERE

############ FUNCTIONS DESCRIPTION ##########
#   _gamma_logcalc: takes as input vectors of ipts (x) and alphas and a scalar beta and outputs a matrix of dimensions
#           length(x) x length(alphas) with the log of the probabilities of each combination of them calculated through
#           a gamma distribution
#
#   _dchen_sum: takes as input vectors of ipts (ipt) and ks (k) as well as scalars beta and p and returns a vector
#           of probabilities o each ipt calculated through the chen probability density function
#
#   _likelihood: takes as input a vector with the parameters (p, beta) and a vector of ipts (x) gives as output the
#           likelihood of each ipt given those parameters based on the chen density with maximum k equal to 100
#
#   _loglikelihood: takes as input a vector with the parameters (p, beta) and a vector of ipts (x) gives as output the
#           loglikelihood of each ipt given those parameters based on the chen density with maximum k equal to 100
#
#   _pchen: takes as input a vector of ks (k) as well as scalars ipt, beta and p and returns the probability
#           waiting more than the observed time since last purchase before next purchase
#
#   _right_censoring:   takes as input a vector of parameters (p, beta) and the time since last purchase (last) and
#           returns the probability of this combination
#
#   _likelihood_helper: support function for the likelihood of internal use
#
#   maximum_likelihood_estimate: takes as input a vector of ipts (x) and returns a list with, as first element the MLE
#           of the parameters (p, beta) based on those and as second the probability associated to them.
#           Uses a brute force method to find it on a grid of points of distance 0.025.
#
#   plot_model: takes as input a vector of ipts (new_x), the maximum_likelihood_estimate output (sol) and the true beta
#           of those data and plots a histogram of the data with three lines of gamma distributions, one based on
#           the true beta, one on the chen estimate through MLE and one through the classical Erlang2 MLE. No output.
#
#   total_pipeline: takes as input an vector of ipts or a list of vectors and
#   solves for our two parameters using a maximum likelihood (over a grid of
#   possible parameters) and then plots the true parameter, the model estimate
#   and a naive estimate for the rate parameters
#
#   _prior: takes as input a vector of parameters (p, beta) and returns their prior probability, which is based on
#           2 independent beta(2, 2) functions
#
#   posterior: takes as input a vector of parameters theta (p, beta) and a vector of ipts (x) as well as a time from
#           the last purchase (in the form of the output of simulate_data) and returns the posterior probability the
#           theta vector, using a combination of _prior, _likelihood and _right_censoring.
#
#   log_posterior_new: takes as input a vector of parameters theta (p, beta) and a vector of ipts (x) as well as a time
#           from the last purchase (in the form of the output of simulate_data) and returns the logarithm of the
#           posterior probability the theta vector, using a combination of _prior, _loglikelihood and _right_censoring.
#
#   log_posterior: takes as input a vector of parameters theta (p, beta) and a vector of ipts (x) as well as a time
#           from the last purchase (in the form of the output of simulate_data) and returns the logarithm of the
#           posterior probability the theta vector, using a combination of _prior, _likelihood and _right_censoring.
#
#   metropolis: takes as input an x of the same format as the output of simulate_data, a starting_point, a chain_length
#           and a burn_in (a rate) and uses a metropolis random walk to simulate the parameters. It returns the trajectory
#           of the parameters cleaned of the initial burn_in percent. (Simulates 2d parameters)
#
#   metropolis_new: takes as input an x of the same format as the output of simulate_data, a starting_point, a chain_length
#           and a burn_in (a rate) and uses a metropolis random walk to simulate the parameters. It returns the trajectory
#           of the parameters cleaned of the initial burn_in percent. The difference with metropolis lies in small
#           changes in computations that improve the accuracy. (Simulates 2d parameters)
#
#   plot_trajectory: takes as input the trajectory resulting from the random walk and plots it in a 2d space.
#
#   plot_region: takes as input the trajectory resulting from the random walk as well as the initial ipts (x) in the
#           format that is returned by simulate_data, and a credibility rate (credmass). It plots the point in a
#           2D scatterplot that shows the credibility interval with colors.
#
#   metropolis_solution: takes as input a list of observations and uses the
#   metropolis algorithm to simulate parameters. Parameter estimates are then
#   obtained as the mean of the trajectory. Standard deviation and a posterior
#   predicitve expeted ipt is also saved
#
#   metropolis_solution_all: same as above, just saves all of the trajectory
#   and not only mean, standard deviation and posterior predictive estimate.
#   This is required later on for the hierarchical modelling.
#
#   _dchen_sum_3par: takes as input vectors of ipts (ipt) and ks (k) as well as scalars beta, p0 and p1 and returns a
#           vector of probabilities o each ipt calculated through the chen probability density function
#
#   _likelihood_3par:   takes as input a vector (theta) of the parameters (p0, p1, beta), a vector of the ipts (x) and
#           a maximum k (max_k = 100) and gives as output the likelihood of this combination
#
#   _loglikelihood_3par:   takes as input a vector (theta) of the parameters (p0, p1, beta), a vector of the ipts (x) and
#           a maximum k (max_k = 100) and gives as output the loglikelihood of this combination
#
#   _pchen: takes as input a vector of ks (k) as well as scalars ipt, beta, p0 and p1 and returns the probability
#           of waiting more than the observed time since last purchase before next purchase
#
#   _right_censoring:   takes as input a vector of parameters (p0, p1, beta) and the time since last purchase (last) and
#           returns the probability of this combination
#
#   _likelihood_helper_3par:    support function that puts together _loglikelihood_3par and _right_censoring
#
#   maximum_likelihood_estimate_3par: takes as input a vector of ipts (x) and returns a list with, as first element the
#           MLE of the parameters (p0, p1, beta) based on those and as second the probability associated to them.
#           Uses a brute force method to find it on a grid of points of distance 0.025.
#
#   plot_model_3par: takes as input a vector of ipts (new_x), the maximum_likelihood_estimate_3par output (sol) and the
#           true beta of those data and plots a histogram of the data with three lines of gamma distributions, one based
#           on the true beta, one on the chen estimate through MLE and one through the classical Erlang2 MLE. No output.
#
#   total_pipeline_3par: takes as input an vector of ipts or a list of vectors and
#   solves for our two parameters using a maximum likelihood (over a grid of
#   possible parameters, here 3 parameters in total) and then plots the true parameter, the model estimate
#   and a naive estimate for the rate parameters
#
#   _prior_3par: takes as input a vector of parameters (p0, p1, beta) and returns their prior probability, which is
#           based on 3 independent beta(2, 2) functions
#
#   posterior_3par: takes as input a vector of parameters theta (p0, p1, beta) and a vector of ipts (x) as well as a
#           time from the last purchase (in the form of the output of simulate_data) and returns the posterior
#           probability of the theta vector, using a combination of _prior_3par, _likelihood_3par and
#           _right_censoring_3par.
#
#   logposterior_3par:  takes as input a vector of parameters theta (p0, p1, beta) and a vector of ipts (x) as well as a
#           time from the last purchase (in the form of the output of simulate_data) and returns the posterior
#           logprobability of the theta vector, using a combination of _prior_3par, _loglikelihood_3par and
#           _right_censoring_3par.
#
#   metropolis_3par:    takes as input an x of the same format as the output of simulate_data, a starting_point, a
#           chain_length and a burn_in (a rate) and uses a metropolis random walk to simulate the parameters. It returns
#           the trajectory of the parameters cleaned of the initial burn_in percent. (Simulates 3 parameters)
#
#   plot_trajectory_3par: takes as input the trajectory resulting from the 3par random walk as well as 3 logical
#           parameters p0, p1 and beta, at least 2 of them must be true, the plot is either a 3d plot of the trajectory
#           with the 3 axis for the 3 parameters if they are all true or a 2d plot of the trajectory over the two
#           parameters of interest.
#
#   plot_region: takes as input the trajectory resulting from the random walk as well as the initial ipts (x) in the
#           format that is returned by simulate_data_3par and a credibility rate (credmass). It also takes as input
#           4 logical parameters p0, p1, beta and beta_vs_sow. It then creates a 3d scatterplot if p0, p1 and beta are all true
#           or in 2d on the selected ones if only 2 of the parameters are true.
#           I can also plot the beta parameter vs an estimation of the sow based on the simulated p0 and p1 if beta_vs_sow is true
#           The scatterplots always use colors in order to show the credibility interval on the parameters.
#


import math
import numpy as np
from scipy import stats, special
from scipy import optimize as opt
import random as rnd
from matplotlib import pyplot as plt
import numpy.random
import pandas as pd
from pandas.tools.plotting import *
import matplotlib.pyplot as plt
import json
#import corner
import os

def _gamma_calc(x, alpha, beta):
    x = np.array(x)
    alpha = np.array(alpha)
    #Beta should be a single value though
    beta = np.array(beta)

    Ki = np.divide(np.power(beta, alpha), special.gamma(alpha))
    Xj = np.exp(-beta * x)
    KXij = np.outer(Xj, Ki)
    #This is the only element that requires an operation in both x and alpha, hence we will turn this power
    #into a vactorized form
    LMij = np.outer(np.log(x), alpha - 1)
    Mij = np.exp(LMij)
    prob_mat = Mij * KXij
    return(prob_mat)



def _gamma_logcalc(x, alpha, beta):
    '''takes as input vectors of ipts (x) and alphas and a scalar beta and outputs a matrix of dimensions
    length(x) x length(alphas) with the log of the probabilities of each combination of them calculated through
    a gamma distribution'''
    x = np.array(x)
    alpha = np.array(alpha)
    # Beta should be a single value though
    beta = np.array(beta)

    Ki = np.log(np.divide(np.power(beta, alpha), special.gamma(alpha)))
    Xj = -beta * x
    KXij = np.add.outer(Xj, Ki)
    #This is the only element that requires an operation in both x and alpha, hence we will turn this power
    #into a vactorized form
    LMij = np.outer(np.log(x), alpha - 1)

    log_prob_mat = LMij + KXij
    return(log_prob_mat)

def _dchen_sum(ipt, k, p, beta):
    '''takes as input vectors of ipts (ipt) and ks (k) as well as scalars beta and p and returns a vector
    of probabilities o each ipt calculated through the chen probability density function'''
    shape = 2 * (k+1)
    p_vec = p * np.power(1-p, k)
    M_gamma = np.exp(_gamma_logcalc(ipt, shape, beta))
    result = M_gamma.dot(p_vec)
    return(result)

def _likelihood(theta,x):
    '''takes as input a vector with the parameters (p, beta) and a vector of ipts (x) gives as output the
    likelihood of each ipt given those parameters based on the chen density with maximum k equal to 100'''
    #get share observed
    p = theta[0]
    #get rate parameter from input
    beta = theta[1]
    #max number of unobserved potential other store visits for iteration
    max_k = 100
    unobserved_id = np.arange(max_k+1)
    ipt_observed = np.array(x)
    #loop through all observations and k and model likelyhood of observing ipt
    likelihood_return = np.prod(_dchen_sum(ipt_observed, unobserved_id, p, beta))
    return(likelihood_return)

def _loglikelihood(theta,x):
    '''takes as input a vector with the parameters (p, beta) and a vector of ipts (x) gives as output the
    loglikelihood of each ipt given those parameters based on the chen density with maximum k equal to 100'''
    #get share observed
    p = theta[0]
    #get rate parameter from input
    beta = theta[1]
    #max number of unobserved potential other store visits for iteration
    max_k = 100
    unobserved_id = np.arange(max_k+1)
    ipt_observed = np.array(x)
    #loop through all observations and k and model likelyhood of observing ipt
    likelihood_return = np.sum(np.log(_dchen_sum(ipt_observed, unobserved_id, p, beta)))
    return(likelihood_return)

# custom cdf for "chen" distribution
def _pchen(ipt, k, p, beta):
    '''takes as input a vector of ks (k) as well as scalars ipt, beta and p and returns the probability
    waiting more than the observed time since last purchase before next purchase'''
    shape = 2 * (k + 1)
    rate = beta
    result = p * stats.gamma.cdf(x=ipt, a=shape, loc=0, scale=1 / rate) * (1 - p) ** k
    return (result)

def _right_censoring(theta, last):
    '''takes as input a vector of parameters (p, beta) and the time since last purchase (last) and
    returns the probability of this combination'''
    p = theta[0]
    beta = theta[1]
    max_k = 100
    # loop through all observations and k and model likelyhood of not observing until end
    unobserved_id = list(range(max_k + 1))
    ipt_observed = last
    # loop through all observations and k and model likelyhood of observing ipt
    prob = np.apply_along_axis(lambda k: _pchen(ipt_observed, k, p, beta), 0, unobserved_id)

    # create results list with table and s integral
    s_integral = 1 - np.sum(prob)
    return (s_integral)

def _likelihood_helper(theta, x):
    '''support function for the likelihood of internal use'''
    return(_likelihood(theta, x[0]) * _right_censoring(theta, x[1]))

#find optimum in a grid, steps of 0.025 from 0.05 to 0.95 2-dimensional.
def maximum_likelihood_estimate(x):
    '''takes as input a vector of ipts (x) and returns a list with, as first element the MLE
    of the parameters (p, beta) based on those and as second the probability associated to them.
    Uses a brute force method to find it on a grid of points of distance 0.025.'''
    V1 = np.arange(38)*0.025 + 0.05
    V2 = np.arange(78)*0.025 + 0.05
    argsList = [[i,j] for i in V1 for j in V2]
    Probs = np.array([_likelihood_helper(args, x) for args in argsList])
    Idx = np.argmax(Probs)
    Val = [argsList[Idx], Probs[Idx]]
    return(Val)


def plot_model(new_x, sol, rate_param):
    '''takes as input a vector of ipts (new_x), the maximum_likelihood_estimate output (sol) and the true beta
    of those data and plots a histogram of the data with three lines of gamma distributions, one based on
    the true beta, one on the chen estimate through MLE and one through the classical Erlang2 MLE. No output.'''
    #x1 is needed to set the extremes of the x axis
    x1 = np.arange(math.ceil(np.max(new_x[0])*10))*0.1

    shape = 2.0
    y1 = stats.gamma.pdf(x = x1, a = shape, loc=0.0, scale=1/rate_param)

    y2 = stats.gamma.pdf(x = x1, a = shape, loc=0.0, scale=1/sol[0][1])

    B_mle = np.mean(new_x[0])/2
    y3 = stats.gamma.pdf(x = x1, a = shape, loc=0.0, scale = B_mle)

    plt.hist(list(map(math.ceil, new_x[0])), bins = list(map(float,range(math.ceil(max(new_x[0]))+1))), normed = 1.0)

    A, = plt.plot(x1, y1, 'k', label = 'True Distribution')
    B, = plt.plot(x1, y2, 'b', label = 'Chen Model Erlang')
    C, = plt.plot(x1, y3, 'r', label = 'True Distribution')

    plt.xlabel('Inter Purchase Time - Number of Days')
    plt.ylabel('Density \n Share of Observations')

    Handles = [A, B, C]
    Dist = ['True Distribution', 'Chen Model Erlang', 'Naive MLE Erlang']
    plt.legend(Handles, Dist)

    plt.show()


#function executing the whole pipeline
#1. generate data and delete observations
#2. MLE of parameter pairs
#3. plot true, naive and chen distribution
def total_pipeline(observation_list, parameter_list, idx = None):
    if(idx is None):
        new_x = observation_list
        rate_param = parameter_list[1]
        sow = parameter_list[0]
    else:
        new_x = observation_list[idx]
        rate_param = parameter_list[idx,1]
        sow = parameter_list[idx,0]
    sol = maximum_likelihood_estimate(new_x)
    plot_model(new_x, sol, rate_param)
    print("True SOW", sow)
    print("Chen SOW", sol[0][0])
    print("True beta", rate_param)
    print("Chen beta", sol[0][1])
    print("Naive beta", 2/np.mean(new_x[0]))

# Define the prior density function.
# The input argument is a vector: theta = c( theta1 , theta2 )
def _prior(theta):
    '''takes as input a vector of parameters (p, beta) and returns their prior probability, which is based on
    2 independent beta(2, 2) functions'''
    # Here's a beta-beta prior:
    a1 = 2;
    b1 = 2;
    a2 = 2;
    b2 = 2
    prior = stats.beta.pdf(x=theta[0], a=a1, b=b1) * stats.beta.pdf(x=theta[1]/2, a=a2, b=b2)
    return (prior)


def posterior(theta, x):
    '''takes as input a vector of parameters theta (p, beta) and a vector of ipts (x) as well as a time from
    the last purchase (in the form of the output of simulate_data) and returns the posterior probability the
    theta vector, using a combination of _prior, _likelihood and _right_censoring.'''
    if (all(item >= 0.0 for item in theta) & all(item <= 1.0 for item in theta[:-1])):

        LL = _likelihood(theta, x[0])[2]

        RC = _right_censoring(theta, x[1])

        PT = _prior(theta)

        posterior_prob = LL * RC * PT

    else:
        # This part is important so that the Metropolis algorithm
        # never accepts a jump to an invalid parameter value.
        posterior_prob = 0.0
    return (posterior_prob)


def log_posterior_new(theta, x):
    '''takes as input a vector of parameters theta (p, beta) and a vector of ipts (x) as well as a time
    from the last purchase (in the form of the output of simulate_data) and returns the logarithm of the
    posterior probability the theta vector, using a combination of _prior, _loglikelihood and _right_censoring.'''
    if (all(item >= 0.0 for item in theta) & all(item <= 1.0 for item in theta[:-1])):

        LL = _loglikelihood(theta, x[0])

        RC = np.log(_right_censoring(theta, x[1]))

        PT = np.log(_prior(theta))

        posterior_log_prob = LL + RC + PT

        InSupport = True

    else:
        # This part is important so that the Metropolis algorithm
        # never accepts a jump to an invalid parameter value.
        posterior_log_prob = False
        InSupport = False

    return ([InSupport, posterior_log_prob])

def log_posterior(theta, x):
    '''takes as input a vector of parameters theta (p, beta) and a vector of ipts (x) as well as a time
    from the last purchase (in the form of the output of simulate_data) and returns the logarithm of the
    posterior probability the theta vector, using a combination of _prior, _likelihood and _right_censoring.'''
    if (all(item >= 0.0 for item in theta) & all(item <= 1.0 for item in theta[:-1])):

        LL = np.log(_likelihood(theta, x[0]))

        RC = np.log(_right_censoring(theta, x[1]))

        PT = np.log(_prior(theta))

        posterior_log_prob = LL + RC + PT

        InSupport = True

    else:
        # This part is important so that the Metropolis algorithm
        # never accepts a jump to an invalid parameter value.
        posterior_log_prob = False
        InSupport = False

    return ([InSupport, posterior_log_prob])

def metropolis(x, starting_point=[0.5, 1.0], chain_length=1000, burn_in=0.1):
    '''takes as input an x of the same format as the output of simulate_data, a starting_point, a chain_length
    and a burn_in (a rate) and uses a metropolis random walk to simulate the parameters. It returns the trajectory
    of the parameters cleaned of the initial burn_in percent. (Simulates 2d parameters)'''
    trajLength = math.ceil(chain_length / .9)  # arbitrary large number
    # Initialize the vector that will store the results.
    trajectory = np.zeros(shape=(trajLength + 1, 2))
    # Specify where to start the trajectory
    trajectory[0,] = starting_point  # arbitrary start values of the two param's

    # Specify the burn-in period.
    burnIn = math.ceil(burn_in * trajLength)  # arbitrary number
    # Initialize accepted, rejected counters, just to monitor performance.
    nAccepted = 0
    nRejected = 0
    # Specify the seed, so the trajectory can be reproduced.

    # Specify the covariance matrix for multivariate normal proposal distribution.
    nDim = 2
    sd1 = 0.02
    sd2 = 0.02

    covarMat = np.zeros(shape=(nDim, nDim))
    covarMat[0, 0] = sd1
    covarMat[1, 1] = sd2

    jumpMatrix = stats.multivariate_normal.rvs(mean=np.zeros(2), cov=covarMat, size=trajLength)
    probAcceptMatrix = stats.uniform.rvs(size=trajLength)

    # Now generate the random walk. stepIdx is the step in the walk.
    for stepIdx in range(trajLength):

        currentPosition = trajectory[stepIdx,]
        # Use the proposal distribution to generate a proposed jump.
        # The shape and variance of the proposal distribution can be changed
        # to whatever you think is appropriate for the target distribution.
        proposedJump = jumpMatrix[stepIdx,]
        # Compute the probability of accepting the proposed jump.
        P0 = log_posterior(currentPosition, x)
        P1 = log_posterior(currentPosition + proposedJump, x)
        if P1[0]:

            probAccept = np.exp(P1[1] - P0[1])
        else:
            probAccept = 0.0
        # Generate a random uniform value from the interval [0,1] to
        # decide whether or not to accept the proposed jump.
        if (probAcceptMatrix[stepIdx] < probAccept):
            # accept the proposed jump
            trajectory[stepIdx + 1,] = currentPosition + proposedJump

        else:
            # reject the proposed jump, stay at current position
            trajectory[stepIdx + 1,] = currentPosition

    acceptedTraj = trajectory[(burnIn + 1):, ]
    return (acceptedTraj)


def metropolis_new(x, starting_point=[0.5, 1.0], chain_length=1000, burn_in=0.1):
    '''takes as input an x of the same format as the output of simulate_data, a starting_point, a chain_length
    and a burn_in (a rate) and uses a metropolis random walk to simulate the parameters. It returns the trajectory
    of the parameters cleaned of the initial burn_in percent. The difference with metropolis lies in small
    changes in computations that improve the accuracy. (Simulates 2d parameters)'''
    trajLength = math.ceil(chain_length / .9)  # arbitrary large number
    # Initialize the vector that will store the results.
    trajectory = np.zeros(shape=(trajLength, 2))
    # Specify where to start the trajectory
    trajectory[0,] = starting_point  # arbitrary start values of the two param's

    # Specify the burn-in period.
    burnIn = math.ceil(burn_in * trajLength)  # arbitrary number
    # Initialize accepted, rejected counters, just to monitor performance.
    nAccepted = 0
    nRejected = 0
    # Specify the seed, so the trajectory can be reproduced.

    # Specify the covariance matrix for multivariate normal proposal distribution.
    nDim = 2
    sd1 = 0.02
    sd2 = 0.02

    covarMat = np.zeros(shape=(nDim, nDim))
    covarMat[0, 0] = sd1
    covarMat[1, 1] = sd2

    jumpMatrix = stats.multivariate_normal.rvs(mean=np.zeros(2), cov=covarMat, size=trajLength - 1)
    probAcceptMatrix = stats.uniform.rvs(size=trajLength - 1)

    # Now generate the random walk. stepIdx is the step in the walk.
    for stepIdx in range(trajLength - 1):

        currentPosition = trajectory[stepIdx,]
        # Use the proposal distribution to generate a proposed jump.
        # The shape and variance of the proposal distribution can be changed
        # to whatever you think is appropriate for the target distribution.
        proposedJump = jumpMatrix[stepIdx,]
        # Compute the probability of accepting the proposed jump.
        P0 = log_posterior_new(currentPosition, x)
        P1 = log_posterior_new(currentPosition + proposedJump, x)
        if P1[0]:

            probAccept = np.exp(P1[1] - P0[1])
        else:
            probAccept = 0.0
        # Generate a random uniform value from the interval [0,1] to
        # decide whether or not to accept the proposed jump.
        if (probAcceptMatrix[stepIdx] < probAccept):
            # accept the proposed jump
            trajectory[stepIdx + 1,] = currentPosition + proposedJump

        else:
            # reject the proposed jump, stay at current position
            trajectory[stepIdx + 1,] = currentPosition

    acceptedTraj = trajectory[(burnIn + 1):, ]
    return (acceptedTraj)

def plot_trajectory(acceptedTraj):
    '''takes as input the trajectory resulting from the random walk and plots it in a 2d space.'''
    # Compute the mean of the accepted points.
    meanTraj = np.mean(acceptedTraj, 0)
    # Compute the standard deviations of the accepted points.
    sdTraj = np.std(acceptedTraj, 0)

    # Display the sampled points
    """Consider square shaped plot"""
    # par( pty="s" ) # makes plots in square axes.
    XY = np.transpose(acceptedTraj)
    plt.plot(XY[0], XY[1], 'b')
    plt.xlim([0.0, 1.0])
    if (np.max(acceptedTraj[:,1])>1):
        y_max = 2
    else:
        y_max = 1
    plt.ylim([0.0, y_max])
    plt.xlabel('P')
    plt.ylabel(r'$\beta$')
    # Display means and rejected/accepted ratio in plot.
    if (meanTraj[0] > .5):
        xpos, xadj = 0.01, 0.01
    else:
        xpos, xadj = 0.95, 0.95
    if (meanTraj[1] > .5):
        ypos, yadj = 0.01, 0.01
    else:
        ypos, yadj = 0.95, 0.95
    # text to be modified
    Vals = (round(meanTraj[0], 4), round(sdTraj[0], 4), round(meanTraj[1], 4), round(sdTraj[1], 4))
    plt.text(xpos, ypos, r'$\mu_{p}=(%s), \sigma_{p}=(%s), \mu_{\beta}=(%s), \sigma_{\beta}=(%s).$' % Vals)
    plt.show()


def plot_region(acceptedTrajTable, x, credmass=95.0):
    '''takes as input the trajectory resulting from the random walk as well as the initial ipts (x) in the
    format that is returned by simulate_data, and a credibility rate (credmass). It plots the point in a
    2D scatterplot that shows the credibility interval with colors.'''
    # Estimate highest density region by evaluating posterior at each point.
    npts = np.shape(acceptedTrajTable)[0];
    postProb = np.zeros(npts)

    postProb = np.array([log_posterior([i[0], i[1]], x)[1] for i in acceptedTrajTable])
    print(postProb)

    # Determine the level at which credmass points are above:
    waterline = np.percentile(postProb, 100.0 - credmass)

    # need to add prob line as x[2]
    # acceptedTraj1 = filter(lambda x: x[2] <= waterline, acceptedTrajTable)
    # acceptedTraj1 = filter(lambda x: x[2] > waterline, acceptedTrajTable)
    # Display highest density region in new graph
    AT1 = np.transpose([acceptedTrajTable[i,] for i in range(npts) if postProb[i] <= waterline])
    AT2 = np.transpose([acceptedTrajTable[i,] for i in range(npts) if postProb[i] > waterline])

    # par( pty="s" ) # makes plots in square axes.     also: elements to add in, type="p" , pch="x" , col="grey"
    # plt.plot(acceptedTraj2[ postProb < waterline,],xlim = c(0,1) , xlab = 'P' ,ylim = c(0,1) , ylab = r'$\beta$' ,main=) )
    # points( acceptedTraj2[ postProb >= waterline , ] ,  pch="o" , col="black" )
    ## Change next line if you want to save the graph.


    OCI, = plt.plot(AT1[0], AT1[1], 'ro', label='Outside CI')
    ICI, = plt.plot(AT2[0], AT2[1], 'bo', label='Inside CI')
    plt.xlim([0.0, 1.0])
    if (np.max(acceptedTrajTable[:,1])>1):
        y_max = 2
    else:
        y_max = 1
    plt.ylim([0.0, y_max])
    plt.xlabel('P')
    plt.ylabel(r'$\beta$')
    plt.title(str(credmass) + "% HD region")
    plt.legend([ICI, OCI], ['Inside CI', 'Outside CI'])
    plt.show()

def metropolis_solution(total_simulation_array):
    #error terms
    sol_metropolis = metropolis(total_simulation_array, [np.random.rand(1)[0],np.random.rand(1)[0]], 5000, 0.1)
    sol_mean = sol_metropolis.mean(axis = 0)
    sol_sd = sol_metropolis.std(axis = 0)
    sol_pred = get_posterior_predictive(sol_metropolis)
    return [sol_mean, sol_sd, sol_pred]

def metropolis_solution_3par(total_simulation_array):
    #error terms
    sol_metropolis = metropolis_3par(total_simulation_array, [np.random.rand(1)[0],np.random.rand(1)[0],np.random.rand(1)[0]], 5000, 0.1)
    sol_mean = sol_metropolis.mean(axis = 0)
    sol_sd = sol_metropolis.std(axis = 0)
    sol_pred = get_posterior_predictive(sol_metropolis)
    return [sol_mean, sol_sd, sol_pred]

def metropolis_solution_trajectory(total_simulation_array):
    #error terms
    sol_metropolis = metropolis(total_simulation_array, [np.random.rand(1)[0],np.random.rand(1)[0]], 5000, 0.1)
    return sol_metropolis

def metropolis_solution_3par_trajectory(total_simulation_array):
    #error terms
    sol_metropolis = metropolis_3par(total_simulation_array, [np.random.rand(1)[0],np.random.rand(1)[0],np.random.rand(1)[0]], 5000, 0.1)
    return sol_metropolis


def metropolis_solution_all(total_simulation_array):
    #error terms
    sol = np.zeros([len(total_simulation_array),2])
    sol_sd = np.zeros([len(total_simulation_array),2])

    for i in range(len(total_simulation_array)):
        sol_metropolis = metropolis(total_simulation_array[i], [np.random.rand(1)[0],np.random.rand(1)[0]], 5000, 0.1)
        sol_mean = sol_metropolis.mean(axis = 0)
        sol_sd[i,:] = sol_metropolis.std(axis = 0)
        sol[i,0] = sol_mean[0]
        sol[i,1] = sol_mean[1]

    return [sol, sol_sd]



#X = simulate_data_3par(pi0 = 0.5, pi1 = 0.8, beta = 0.4, Time_span = 365.0)

#repeated_simulation_3par(np.array([0.5, 0.6]), np.array([0.3, 0.8]), np.array([0.6, 0.4]), time_span = 365.0, rep = 2)

def _dchen_sum_3par(ipt, k, p0, p1, beta):
    '''takes as input vectors of ipts (ipt) and ks (k) as well as scalars beta, p0 and p1 and returns a
    vector of probabilities o each ipt calculated through the chen probability density function'''
    shape = 2 * (k+1)
    p_vec = np.zeros(shape = k.shape)
    p_vec[0] = p_vec[0] + p1
    p_vec[1:p_vec.shape[0]] = p_vec[1:p_vec.shape[0]] + (1-p1) * p0 * np.power(1-p0, k[1:p_vec.shape[0]]-1)
    M_gamma = np.exp(_gamma_logcalc(ipt, shape, beta))
    result = M_gamma.dot(p_vec)
    return(result)



def _likelihood_3par(theta, x, max_k = 100):
    '''takes as input a vector (theta) of the parameters (p0, p1, beta), a vector of the ipts (x) and
    a maximum k (max_k = 100) and gives as output the likelihood of this combination'''
    #get share observed
    p0 = theta[0]
    p1 = theta[1]
    #get rate parameter from input
    beta = theta[2]
    #max number of unobserved potential other store visits for iteration

    unobserved_id = np.arange(max_k+1)
    ipt_observed = np.array(x)

    prob_list10 = np.array(_dchen_sum_3par(ipt_observed, unobserved_id, p0, p1, beta)) * 10

    # As i multiply by 10 in order to have workable numbers the probabilities are not correct anymore!!!

    likelihood_return = np.prod(prob_list10)
    return(likelihood_return)



def _loglikelihood_3par(theta, x, max_k = 100):
    '''takes as input a vector (theta) of the parameters (p0, p1, beta), a vector of the ipts (x) and
    a maximum k (max_k = 100) and gives as output the loglikelihood of this combination'''
    #get share observed
    p0 = theta[0]
    p1 = theta[1]
    #get rate parameter from input
    beta = theta[2]
    #max number of unobserved potential other store visits for iteration

    unobserved_id = np.arange(max_k+1)
    ipt_observed = np.array(x)

    logprob_list = np.log(_dchen_sum_3par(ipt_observed, unobserved_id, p0, p1, beta))
    #loop through all observations and k and model likelyhood of observing ipt
    likelihood_return = np.sum(logprob_list)
    return(likelihood_return)


def _pchen_3par(ipt, k, p0, p1, beta):
    '''takes as input a vector of ks (k) as well as scalars ipt, beta, p0 and p1 and returns the probability
    of waiting more than the observed time since last purchase before next purchase'''
    shape = 2 * (k + 1)

    p_vec = np.zeros(k.shape[0])
    p_vec[0] = p_vec[0] + p1
    p_vec[1:p_vec.shape[0]] = p_vec[1:p_vec.shape[0]] + (1 - p1) * p0 * np.power(1 - p0, k[1:p_vec.shape[0]] - 1)

    cumulative_gamma = stats.gamma.cdf(x=ipt, a=shape, loc=0, scale=1 / beta)

    result = np.multiply(p_vec, cumulative_gamma)
    return (result)



def _right_censoring_3par(theta, last, max_k = 100):
    '''takes as input a vector of parameters (p0, p1, beta) and the time since last purchase (last) and
    returns the probability of observing such or longer wait'''
    p0 = theta[0]
    p1 = theta[1]
    beta = theta[2]
    # loop through all observations and k and model likelyhood of not observing until end
    unobserved_id = np.arange(max_k + 1)

    prob = _pchen_3par(last, unobserved_id, p0, p1, beta)
    # create results list with table and s integral
    s_integral = 1 - np.sum(prob)
    return (s_integral)



def _likelihood_helper_3par(theta, x):
    '''support function that puts together _loglikelihood_3par and _right_censoring'''
    LL = _loglikelihood_3par(theta, x[0])
    RC = np.log(_right_censoring_3par(theta, x[1]))
    return(LL + RC)




def maximum_likelihood_estimate_3par(x):
    '''takes as input a vector of ipts (x) and returns a list with, as first element the
    MLE of the parameters (p0, p1, beta) based on those and as second the probability associated to them.
    Uses a brute force method to find it on a grid of points of distance 0.025.'''
    V1 = np.arange(38) * 0.025 + 0.05
    V2 = np.arange(38) * 0.025 + 0.05
    V3 = np.arange(78) * 0.025 + 0.05
    argsList = [[i, j, l] for i in V1 for j in V2 for l in V3]
    Probs = np.array([_likelihood_helper_3par(args, x) for args in argsList])
    Idx = np.argmax(Probs)
    Val = [argsList[Idx], Probs[Idx]]
    return(Val)
#maximum_likelihood_estimate_3par(X)



def plot_model_3par(new_x, sol, rate_param):
    '''takes as input a vector of ipts (new_x), the maximum_likelihood_estimate_3par output (sol) and the
    true beta of those data and plots a histogram of the data with three lines of gamma distributions, one based
    on the true beta, one on the chen estimate through MLE and one through the classical Erlang2 MLE. No output.'''
    # x1 is needed to set the extremes of the x axis
    x1 = np.arange(math.ceil(np.max(new_x[0]) * 10)) * 0.1

    shape = 2.0
    y1 = stats.gamma.pdf(x=x1, a=shape, loc=0.0, scale=1 / rate_param)

    y2 = stats.gamma.pdf(x=x1, a=shape, loc=0.0, scale=1 / sol[0][2])

    B_mle = np.mean(new_x[0]) / 2
    y3 = stats.gamma.pdf(x=x1, a=shape, loc=0.0, scale=B_mle)

    plt.hist(list(map(math.ceil, new_x[0])), bins=list(map(float, range(math.ceil(max(new_x[0])) + 1))), normed=1.0)

    A, = plt.plot(x1, y1, 'k', label='True Distribution')
    B, = plt.plot(x1, y2, 'b', label='Chen Model Erlang')
    C, = plt.plot(x1, y3, 'r', label='True Distribution')

    plt.xlabel('Inter Purchase Time - Number of Days')
    plt.ylabel('Density \n Share of Observations')

    Handles = [A, B, C]
    Dist = ['True Distribution', 'Chen Model Erlang', 'Naive MLE Erlang']
    plt.legend(Handles, Dist)

    plt.show()




#function executing the whole pipeline
#1. generate data and delete observations
#2. MLE of parameter pairs
#3. plot true, naive and chen distribution
def total_pipeline_3par(observation_list, parameter_list, idx = None):
    if(idx is None):
        new_x = observation_list
        rate_param = parameter_list[2]
        p0 = parameter_list[0]
        p1 = parameter_list[1]
    else:
        new_x = observation_list[idx]
        rate_param = parameter_list[idx,2]
        #TODO
    sol = maximum_likelihood_estimate_3par(new_x)
    plot_model_3par(new_x, sol, rate_param)
    #TODO
    print("True SOW", parameter_list[1]/(1 - parameter_list[0] + parameter_list[1]))
    print("True p0", parameter_list[0])
    print("True p1", parameter_list[1])
    print("Chen SOW", sol[0][1]/(1 - sol[0][0] + sol[0][1]))
    print("Chen p0", sol[0][0])
    print("Chen p1",sol[0][1])
    print("True beta", rate_param)
    print("Chen beta", sol[0][2])
    print("Naive beta", 2/np.mean(new_x[0]))
    print("Chen Likelihood", sol[1])


def _prior_3par(theta):
    '''takes as input a vector of parameters (p0, p1, beta) and returns their prior probability, which is
    based on 3 independent beta(2, 2) functions'''
    # Here's a beta-beta prior:
    a1 = 2
    b1 = 2
    a2 = 2
    b2 = 2
    a3 = 2
    b3 = 2
    # because beta goes reasonably to 2, use beta prior with "normalized" x (divide by 2)
    prior = stats.beta.pdf(x=theta[0], a=a1, b=b1) * stats.beta.pdf(x=theta[1], a=a2, b=b2) * stats.beta.pdf(x=theta[2]/2, a=a3, b=b3)
    return (prior)




def posterior_3par(theta, x):
    '''takes as input a vector of parameters theta (p0, p1, beta) and a vector of ipts (x) as well as a
    time from the last purchase (in the form of the output of simulate_data) and returns the posterior
    probability of the theta vector, using a combination of _prior_3par, _likelihood_3par and
    _right_censoring_3par.'''
    if (all(item >= 0.0 for item in theta) & all(item <= 1.0 for item in theta[:-1])):

        LL = _likelihood_3par(theta, x[0])[2]

        RC = _right_censoring_3par(theta, x[1])

        PT = _prior_3par(theta)

        posterior_prob = LL * RC * PT

    else:
        # This part is important so that the Metropolis algorithm
        # never accepts a jump to an invalid parameter value.
        posterior_prob = 0.0
    return (posterior_prob)


def log_posterior_3par(theta, x):
    '''takes as input a vector of parameters theta (p0, p1, beta) and a vector of ipts (x) as well as a
    time from the last purchase (in the form of the output of simulate_data) and returns the posterior
    logprobability of the theta vector, using a combination of _prior_3par, _loglikelihood_3par and
    _right_censoring_3par.'''
    if (all(item >= 0.0 for item in theta) & all(item <= 1.0 for item in theta[:-1])):

        LL = _loglikelihood_3par(theta, x[0])

        RC = np.log(_right_censoring_3par(theta, x[1]))

        PT = np.log(_prior_3par(theta))

        posterior_log_prob = LL + RC + PT

        InSupport = True

    else:
        # This part is important so that the Metropolis algorithm
        # never accepts a jump to an invalid parameter value.
        posterior_log_prob = False
        InSupport = False

    return ([InSupport, posterior_log_prob])

#Data = repeated_simulation_3par(np.array([0.5, 0.6]), np.array([0.3, 0.8]), np.array([0.6, 0.4]), time_span = 365.0, rep = 2)
#log_posterior_3par([0.5,0.5,0.5], Data[0][1]) #Maybe prior too strong??



def metropolis_3par(x, starting_point=[0.5, 0.5, 1.0], chain_length=1000, burn_in=0.1):
    '''takes as input an x of the same format as the output of simulate_data, a starting_point, a
    chain_length and a burn_in (a rate) and uses a metropolis random walk to simulate the parameters. It returns
    the trajectory of the parameters cleaned of the initial burn_in percent. (Simulates 3d parameters)'''
    trajLength = math.ceil(chain_length / .9)  # arbitrary large number
    # Initialize the vector that will store the results.
    trajectory = np.zeros(shape=(trajLength + 1, 3))
    # Specify where to start the trajectory
    trajectory[0,] = starting_point  # arbitrary start values of the two param's

    # Specify the burn-in period.
    burnIn = math.ceil(burn_in * trajLength)  # arbitrary number
    # Initialize accepted, rejected counters, just to monitor performance.
    nAccepted = 0
    nRejected = 0
    # Specify the seed, so the trajectory can be reproduced.

    # Specify the covariance matrix for multivariate normal proposal distribution.
    nDim = 3
    sd1 = 0.02 #sd of p0 jumps
    sd2 = 0.02 #sd of p1 jumps
    sd3 = 0.02 #sd of beta jumps

    covarMat = np.zeros(shape=(nDim, nDim))
    covarMat[0, 0] = sd1
    covarMat[1, 1] = sd2
    covarMat[2, 2] = sd3

    jumpMatrix = stats.multivariate_normal.rvs(mean=np.zeros(3), cov=covarMat, size=trajLength)
    probAcceptMatrix = stats.uniform.rvs(size=trajLength)

    # Now generate the random walk. stepIdx is the step in the walk.
    for stepIdx in range(trajLength):

        currentPosition = trajectory[stepIdx,]
        # Use the proposal distribution to generate a proposed jump.
        # The shape and variance of the proposal distribution can be changed
        # to whatever you think is appropriate for the target distribution.
        proposedJump = jumpMatrix[stepIdx,]
        # Compute the probability of accepting the proposed jump.
        P0 = log_posterior_3par(currentPosition, x)
        P1 = log_posterior_3par(currentPosition + proposedJump, x)
        if P1[0]:
            probAccept = np.exp(P1[1] - P0[1])
        else:
            probAccept = 0.0
        # Generate a random uniform value from the interval [0,1] to
        # decide whether or not to accept the proposed jump.
        if (probAcceptMatrix[stepIdx] < probAccept):
            # accept the proposed jump
            trajectory[stepIdx + 1,] = currentPosition + proposedJump

        else:
            # reject the proposed jump, stay at current position
            trajectory[stepIdx + 1,] = currentPosition

    acceptedTraj = trajectory[(burnIn + 1):, ]
    return (acceptedTraj)

#Data = repeated_simulation_3par(np.array([0.5, 0.6]), np.array([0.3, 0.8]), np.array([0.6, 0.4]), time_span = 365.0, rep = 2)
#Traj = metropolis_3par(Data[0][1], starting_point=[0.5, 0.5, 0.5], chain_length=1000, burn_in=0.1)

def plot_metropolis_solution(x, n_simulations, true_par):
    traj = metropolis_3par(x, chain_length = n_simulations)
    sol = np.mean(traj, axis = 0)
    plot_model_3par(x, [sol,], true_par[2])
    print("Solutions obtained through Metropolis Hastings sampling")
    print("True SOW", true_par[1]/(1 - true_par[0] + true_par[1]))
    print("True p0", true_par[0])
    print("True p1", true_par[1])
    print("Chen SOW", sol[1]/(1 - sol[0] + sol[1]))
    print("Chen p0", sol[0])
    print("Chen p1",sol[1])
    print("True beta", true_par[2])
    print("Chen beta", sol[2])
    print("Naive beta", 2/np.mean(x[0]))

def plot_trajectory_3par(acceptedTraj, p0 = True, p1 = True, beta = True):
    '''takes as input the trajectory resulting from the 3par random walk as well as 3 logical
    parameters p0, p1 and beta, at least 2 of them must be true, the plot is either a 3d plot of the trajectory
    with the 3 axis for the 3 parameters if they are all true or a 2d plot of the trajectory over the two
    parameters of interest.'''
    # Compute the mean of the accepted points.
    meanTraj = np.mean(acceptedTraj, 0)
    # Compute the standard deviations of the accepted points.
    sdTraj = np.std(acceptedTraj, 0)
    if p0 and p1 and beta:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # Compute the standard deviations of the accepted points.
        t_traj = np.transpose(acceptedTraj)

        Vals = (round(meanTraj[0], 4), round(sdTraj[0], 4), round(meanTraj[1], 4), round(sdTraj[1], 4), round(meanTraj[2], 4),round(sdTraj[2], 4))
        ax.text2D(0.05, 0.95,
                  r'$\mu_{p0}=(%s), \sigma_{p0}=(%s),$' % Vals[0:2] + os.linesep +
                  r'$\mu_{p1}=(%s), \sigma_{p1}=(%s),$' % Vals[2:4] + os.linesep +
                  r'$ \mu_{\beta}=(%s), \sigma_{\beta}=(%s).$' % Vals[4:6],
                  transform=ax.transAxes)

        ax.set_xlabel('p0')
        ax.set_ylabel('p1')
        ax.set_zlabel(r'$\beta$')
        ax.plot(t_traj[0], t_traj[1], t_traj[2])

        plt.show()

    # Display the sampled points
    # par( pty="s" ) # makes plots in square axes.
    elif (p0 and p1) or (p0 and beta) or (p1 and beta):
        XY = np.transpose(acceptedTraj[:,[p0, p1, beta]])
        if p0:
            Label1 = 'p0'
            if p1:
                Label2 = 'p1'
                L2 = 'p1'
            else:
                Label2 = r'$\beta$'
                L2 = '\\beta'
        else:
            Label1 = 'p1'
            Label2 = r'$\beta$'
            L2 = '\\beta'
        plt.plot(XY[0], XY[1], 'b')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel(Label1)
        plt.ylabel(Label2)
        # Display means and rejected/accepted ratio in plot.
        if (meanTraj[0] > .5):
            xpos, xadj = 0.01, 0.01
        else:
            xpos, xadj = 0.95, 0.95
        if (meanTraj[1] > .5):
            ypos, yadj = 0.01, 0.01
        else:
            ypos, yadj = 0.95, 0.95
        # text to be modified
        Vals = (Label1, str(round(meanTraj[0], 4)), Label1, str(round(sdTraj[0], 4)),
                L2, str(round(meanTraj[1], 4)), L2, str(round(sdTraj[1], 4)))
        plt.text(xpos, ypos, r'$\mu_{%s}=(%s), \sigma_{%s}=(%s), \mu_{%s}=(%s), \sigma_{%s}=(%s).$' % Vals)
        plt.show()
    else:
        ValueError('Not enough axes')

#Data = repeated_simulation_3par(np.array([0.5, 0.6]), np.array([0.3, 0.8]), np.array([0.6, 0.4]), time_span = 365.0, rep = 2)
#Traj = metropolis_3par(Data[0][1], starting_point=[0.5, 0.5, 0.5], chain_length=1000, burn_in=0.1)

#plot_trajectory_3par(Traj, p0 = True, p1 = False, beta = True)


def plot_region_3par(acceptedTraj, x, credmass=80.0, p0 = True, p1 = True, beta = True, beta_vs_sow = False):
    '''takes as input the trajectory resulting from the random walk as well as the initial ipts (x) in the
    format that is returned by simulate_data_3par and a credibility rate (credmass). It also takes as input
    4 logical parameters p0, p1, beta and beta_vs_sow. It then creates a 3d scatterplot if p0, p1 and beta are all true
    or in 2d on the selected ones if only 2 of the parameters are true.
    I can also plot the beta parameter vs an estimation of the sow based on the simulated p0 and p1 if beta_vs_sow is true
    The scatterplots always use colors in order to show the credibility interval on the parameters.'''
    # Compute the mean of the accepted points.
    meanTraj = np.mean(acceptedTraj, 0)
    # Compute the standard deviations of the accepted points.
    sdTraj = np.std(acceptedTraj, 0)

    npts = np.shape(acceptedTraj)[0]

    postProb = np.array([log_posterior([i[0], i[1], i[2]], x)[1] for i in acceptedTraj])
    print(postProb)

    # Determine the level at which credmass points are above:
    waterline = np.percentile(postProb, 100.0 - credmass)

    # need to add prob line as x[2]
    # acceptedTraj1 = filter(lambda x: x[2] <= waterline, acceptedTrajTable)
    # acceptedTraj1 = filter(lambda x: x[2] > waterline, acceptedTrajTable)
    # Display highest density region in new graph
    AT1 = np.array([acceptedTraj[i,] for i in range(npts) if postProb[i] <= waterline])
    AT2 = np.array([acceptedTraj[i,] for i in range(npts) if postProb[i] > waterline])

    # par( pty="s" ) # makes plots in square axes.     also: elements to add in, type="p" , pch="x" , col="grey"
    # plt.plot(acceptedTraj2[ postProb < waterline,],xlim = c(0,1) , xlab = 'P' ,ylim = c(0,1) , ylab = r'$\beta$' ,main=) )
    # points( acceptedTraj2[ postProb >= waterline , ] ,  pch="o" , col="black" )
    ## Change next line if you want to save the graph.

    if (p0 and p1 and beta):
        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        # Compute the standard deviations of the accepted points.
        AT1 = np.transpose(AT1)
        AT2 = np.transpose(AT2)

        ax.set_xlabel('p0')
        ax.set_ylabel('p1')
        ax.set_zlabel(r'$\beta$')

        ax.scatter(AT1[0], AT1[1], AT1[2], c = 'r')

        ax.scatter(AT2[0], AT2[1], AT2[2], c = 'b')

        plt.show()

    # Display the sampled points
    # par( pty="s" ) # makes plots in square axes.
    elif (p0 and p1) or (p0 and beta) or (p1 and beta):
        AT1 = np.transpose(AT1[:, [p0, p1, beta]])
        AT2 = np.transpose(AT2[:, [p0, p1, beta]])
        if p0:
            Label1 = 'p0'
            if p1:
                Label2 = 'p1'
                L2 = 'p1'
            else:
                Label2 = r'$\beta$'
                L2 = '\\beta'
        else:
            Label1 = 'p1'
            Label2 = r'$\beta$'
            L2 = '\\beta'

        OCI, = plt.plot(AT1[0,:], AT1[1,:], 'ro', label='Outside CI')
        ICI, = plt.plot(AT2[0,:], AT2[1,:], 'bo', label='Inside CI')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel(Label1)
        plt.ylabel(Label2)
        plt.title(str(credmass) + "% HD region")
        plt.legend([ICI, OCI], ['Inside CI', 'Outside CI'])
        plt.show()

    elif beta_vs_sow == True:
        AT1 = np.transpose(AT1)
        AT2 = np.transpose(AT2)
        D1 = AT1[1] / (1 - AT1[0] + AT1[1])
        D2 = AT2[1] / (1 - AT2[0] + AT2[1])
        Label1 = r'$SOW\  or\  \pi(1)$'
        Label2 = r'$\beta$'

        OCI, = plt.plot(D1, AT1[2, :], 'ro', label='Outside CI')
        ICI, = plt.plot(D2, AT2[2, :], 'bo', label='Inside CI')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel(Label1)
        plt.ylabel(Label2)
        plt.title(str(credmass) + "% HD region")
        plt.legend([ICI, OCI], ['Inside CI', 'Outside CI'])
        plt.show()

    else:
        ValueError('Not enough axes')
    # Estimate highest density region by evaluating posterior at each point.

#Data = repeated_simulation_3par(np.array([0.5, 0.6]), np.array([0.3, 0.8]), np.array([0.6, 0.4]), time_span = 365.0, rep = 2)
#Traj = metropolis_3par(Data[0][1], starting_point=[0.5, 0.5, 0.5], chain_length=5000, burn_in=0.1)
#plot_region_3par(Traj, Data[0][1], credmass=80.0, p0 = False, p1 = False, beta = True, beta_vs_sow = True)


def get_posterior_predictive(trajectory, max_ipt = 100):
    #max number of unobserved potential other store visits for iteration
    max_k = 100
    probs_list = np.zeros([len(trajectory), max_ipt +1])
    if(trajectory.shape[1]==2):
        for i in range(len(trajectory)):
            #get share observed
            p = trajectory[i,0]
            #get rate parameter from input
            beta = trajectory[i,1]
            unobserved_id = np.arange(max_k+1)
            ipt_observed = np.arange(max_ipt+1)
            #loop through all observations and k and model likelyhood of observing ipt
            probs_list[i] = _dchen_sum(ipt_observed, unobserved_id, p, beta)
    elif(trajectory.shape[1]==3):
        for i in range(len(trajectory)):
            #get share observed
            p0 = trajectory[i,0]
            p1 = trajectory[i,1]
            #get rate parameter from input
            beta = trajectory[i,2]
            unobserved_id = np.arange(max_k+1)
            ipt_observed = np.arange(max_ipt+1)
            #loop through all observations and k and model likelyhood of observing ipt
            probs_list[i] = _dchen_sum_3par(ipt_observed, unobserved_id, p0, p1, beta)

    result = np.dot(probs_list, ipt_observed)
    return np.mean(result)
