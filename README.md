# SOW

This is a first implementation of several share-of-wallet models (one three-parameter and one two-parameter model) based on

Yuxin _Chen, Joel H. Steckel (2012) Modeling Credit Card Share of Wallet: Solving the Incomplete Information Problem. Journal of Marketing Research: October 2012, Vol. 49, No. 5, pp. 655-669._

The models allow estimating category specific share-of-wallets and modelling the respective true purchasing behaviour of customers within the category with observations from this respective category/ retailer only.
Further, the models allow for incorporating observed heterogeneity through a hierarchical setup.  

The first step of the model, in which the parameters are modelled with a Metropolis-Hastings simulation, is implemented in Python. 
The hierarchical regression of model parameters based on customer or category covariates is based on JAGS in implemented in R using rjags.

***

`chen_utils.py` includes all necessary functions for modelling the parameters of the initial model (Likelihood, Metropolis algorithm and others)

`user_simulation_utils.py` includes several helper functions for modelling the true and observed purchases of fictive users

`chen_model_intro.ipynb` provides a walkthrough through the model.

`metropolis_simulation.ipynb` simulates several users based on predefined covariates and simulates the parameters of interest using the Metropolis-Hastings algorithm.

This trajectory of simulations is then used in a hierarchical model in `jags_chen_simulation.R` to model the relationship between our covariates of interest and the simulated true parameters during the Metropolis-Hastings modelling. 

***

# Requirements

See `requirements.txt`for Python 3. Further, you will need `R` and `JAGS`.
