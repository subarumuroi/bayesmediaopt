import pandas as pd
import numpy as np
import torch
from pyDOE3 import lhs

from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize, Normalize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf

from gpytorch import kernels, means, likelihoods
from gpytorch.priors import LogNormalPrior
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import Interval
from gpytorch.priors import SmoothedBoxPrior

true_optimum = torch.tensor([7.0, 33.5]) # answers determined by querying objective function
max_samples = 200 #This is initial sample + (batch size x number of iterations)


nu = 2.5
lower_noise_bound = .1
upper_noise_bound = .3
bounds = torch.tensor([
    [6.0, 20.0],  # Lower bounds ph, temp
    [8.0, 40.0]   # Upper bounds ph, temp
], dtype=torch.double)
dim = bounds.shape[1]# Extracts number of dimensions from bounds variable

def euclidean_distance(x1, x2):
    return torch.norm(x1-x2).item()

def objective_function(X, pHopt =7, pHopt2 = 5.5, temp_opt =35, temp_opt2=30,  a = 100, b = 20, c = 1, noise_level = 0, seed = None): # changed noise level to 0
    if seed is not None:
            torch.manual_seed(seed) # set for reproducibility
    pH, temp = X[:, 0], X[:, 1]

    # First peak at (pH=7, temp=35)
    pH_term1 = torch.exp(-0.5 * ((pH - pHopt) / 1.5)**2)  # Gaussian term for pH with width 1.5
    temp_term1 = torch.exp(-0.5 * ((temp - temp_opt) / 5.0)**2)  # Gaussian term for temp with width 5.0

    # Second peak at (pH=5.5, temp=30)
    pH_term2 = torch.exp(-0.5 * ((pH - pHopt2) / 1.5)**2)
    temp_term2 = torch.exp(-0.5 * ((temp - temp_opt2) / 5.0)**2)

    # Stronger Sinusoidal Modulation
    sin_component = torch.sin(2 * pH) * torch.cos(1.5 * temp)  # Higher frequency
    wave_strength = 1.5  # Scale up the wave effect

    # Combine the two peaks with the stronger sinusoidal variation
    y = (pH_term1 * temp_term1 + pH_term2 * temp_term2) * (1 + wave_strength * sin_component) *100

    noise = noise_level*torch.randn_like(y) # stddev = noise_level
    
    return y + noise

# GP Model definition
class GPModel(SingleTaskGP):
    def __init__(self, train_X, train_Y, fixed_noise=False, noise_level=0, #changed this to None, since it's defined in the loop
                 lengthscale_prior=None, outputscale_prior=None,
                 lengthscale_constraint = None, outputscale_constraint=None):

        if fixed_noise:
            print(f"Training with FIXED noise: {noise_level} = std dev.")
            noise_variance = (noise_level * train_Y.mean()).pow(2)
            train_Yvar = torch.full_like(train_Y, noise_variance)
            likelihood = None
            super().__init__(
                train_X, train_Y, train_Yvar=train_Yvar, likelihood=likelihood,
                outcome_transform=Standardize(m=1),
                input_transform=Normalize(d=dim)
            )
        else:
            #print("Training with LEARNABLE noise (Gaussian Likelihood).")
            likelihood = likelihoods.GaussianLikelihood()
            super().__init__(
                train_X, train_Y, likelihood=likelihood,
                outcome_transform=Standardize(m=1),
                input_transform=Normalize(d=dim)
            )
            lower_noise = lower_noise_bound**2  # lower noise bound
            upper_noise = upper_noise_bound**2  # upper noise bound

            # Add a **prior** (softly nudges during training)
            
            self.likelihood.noise_covar.register_prior(
                "noise_prior",
                SmoothedBoxPrior(lower_noise, upper_noise),
                "raw_noise"
            )
            
            # Add a **constraint** (hard bounding box)
            self.likelihood.noise_covar.register_constraint(
                "raw_noise",
                Interval(lower_noise, upper_noise)
            )
            

        self.mean_module = means.ConstantMean()#ZeroMean() # this is default ConstantMean() in GPyTorch. Worth investigating

        matern_kernel = kernels.MaternKernel(
            nu=nu,
            ard_num_dims=dim,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=lengthscale_constraint,
        )

        self.covar_module = kernels.ScaleKernel(
            base_kernel=matern_kernel,
            outputscale_prior=outputscale_prior,
            outputscale_constraint=outputscale_constraint,
        )
        
        
        

        
# Training function
def train_GP_model(train_X, train_Y, fixed_noise=False, noise_level=0,# changed this to None, as it's defined in the loop
                   lengthscale_prior=None, outputscale_prior=None,
                   lengthscale_constraint = None, outputscale_constraint=None): 
    model = GPModel(
        train_X, train_Y,
        fixed_noise=fixed_noise,
        noise_level=noise_level,
        lengthscale_prior=lengthscale_prior,
        outputscale_prior=outputscale_prior,
        lengthscale_constraint = lengthscale_constraint, 
        outputscale_constraint=outputscale_constraint
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    
    #===== Fit the model =====#
    fit_gpytorch_mll(mll) # This is the default (turn off the custom training loop below if using this)

    # to use a custom optimizer and make this model fully customizable you can use the bit below 
    # make sure to turn off the fit_gpytorch_mll above
    '''
    # Custom training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    model.train()
    mll.train()
    training_iter = 50
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_X)
        loss = -mll(output, train_Y.squeeze(-1))
        loss.backward()
        optimizer.step()
        print(f"Iter {i+1}/{training_iter} - Loss: {loss.item():.3f}")
    #===== End of custom training loop =====#
    '''
    return model, mll

'''
batch_sizes = [1, 5, 10, 20]
initial_samples = [4, 8, 24, 48]
noise_levels = [0, 10, 20, 50]
seeds = [0, 1, 2, 3, 4]
'''
batch_sizes = [5, 10]
initial_samples = [4, 24]
noise_levels = [0, 10, 20]
seeds = [0,42]
results = []

for batch_size in batch_sizes:
    for initial_sample in initial_samples:
        for noise_level in noise_levels:
            for seed in seeds:
                # initialize initial sample through LHS 
                lhs_design = torch.tensor(
                        lhs(n = dim, samples = initial_sample, criterion = 'maximin', random_state=seed), 
                    dtype=torch.double
                    )
                # flexibly set the bounds using 'bounds' variable (allowing for dimensional scale-up)
                scaled_lhs_design = bounds[0] + (bounds[1]-bounds[0])*lhs_design
                # initialize training data as dictated by LHS and objective function
                train_X = scaled_lhs_design
                train_Y = objective_function(train_X, noise_level) # i don't use seed here to make sure the noise is different for each sample
                train_Y = train_Y.unsqueeze(-1)
                total_sample = initial_sample
                max_iters = int((max_samples - initial_sample) / batch_size) # number of iterations to run
                converged_95 = None
                converged_99 = None
                n_iters = 0

                for iteration in range(max_iters):
                    model, mll = train_GP_model(train_X, train_Y, noise_level= noise_level, fixed_noise=False)

                    model.eval()
                    model.likelihood.eval()

                    acq_func = qLogNoisyExpectedImprovement(model = model, X_baseline = train_X)

                    candidate, _ = optimize_acqf(
                        acq_function=acq_func,
                        bounds=bounds,
                        q=batch_size,
                        num_restarts=5,
                        raw_samples=32
                    )

                    new_y = objective_function(candidate, noise_level= noise_level)
                    train_X = torch.cat([train_X, candidate], dim=0)
                    train_Y = torch.cat([train_Y, new_y.unsqueeze(-1)], dim=0)
                    total_sample += batch_size
                    n_iters += 1

                    best_idx = train_Y.argmax()
                    best_X = train_X[best_idx]

                    dist_to_optim = euclidean_distance(best_X, true_optimum)
                    if converged_95 is None and dist_to_optim <= 0.05:
                        converged_95 = total_sample
                    if converged_99 is None and dist_to_optim <= 0.01:
                        converged_99 = total_sample

                results.append({
                    'batch_size': batch_size,
                    'initial_sample': initial_sample,
                    'noise_level': noise_level,
                    #'beta': beta,
                    'seed': seed,
                    'converged_95': converged_95,
                    'converged_99': converged_99,
                    'final dist to optimum': dist_to_optim,
                    'final best_x': best_X,
                    'final best_y': train_Y[best_idx],
                    'iterations': n_iters
                })

df = pd.DataFrame(results)
print(df)