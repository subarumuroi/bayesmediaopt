import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class SimpleCoCaBO:
    def __init__(self, continuous_dim, categorical_dim, kernel=None, noise_var=1e-5):
        # Initialize the model with dimensions of continuous and categorical variables
        self.continuous_dim = continuous_dim
        self.categorical_dim = categorical_dim
        
        # Define the kernel for the Gaussian Process (RBF + constant kernel)
        self.kernel = kernel if kernel else C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))
        
        # Initialize the Gaussian Process Regressor
        self.gpr = GaussianProcessRegressor(kernel=self.kernel, alpha=noise_var)

        # Storage for past data
        self.X = []  # Stores continuous + categorical variables
        self.y = []  # Stores corresponding objective function values

    def fit(self, X_cont, X_cat, y):
        """Fit the Gaussian Process model on both continuous and categorical data."""
        # Combine continuous and categorical data
        X_combined = np.hstack((X_cont, X_cat))
        
        # Fit the Gaussian Process Regressor model
        self.gpr.fit(X_combined, y)
        
        # Store the data for future optimization
        self.X.extend(X_combined)
        self.y.extend(y)

    def predict(self, X_cont, X_cat):
        """Predict mean and variance for new points."""
        X_combined = np.hstack((X_cont, X_cat))
        mean, std = self.gpr.predict(X_combined, return_std=True)
        return mean, std

    def ucb(self, X_cont, X_cat, kappa=2.0):
        """Upper Confidence Bound (UCB) acquisition function."""
        mean, std = self.predict(X_cont, X_cat)
        ucb_values = mean + kappa * std
        return ucb_values

    def optimize(self, X_cont, X_cat, kappa=2.0):
        """Optimize the acquisition function (UCB)."""
        ucb_values = self.ucb(X_cont, X_cat, kappa)
        best_idx = np.argmax(ucb_values)  # Select the index with the highest UCB value
        return X_cont[best_idx], X_cat[best_idx]

# Example usage
if __name__ == "__main__":
    # Example continuous and categorical variables
    X_cont = np.array([[0.5], [0.2], [0.7]])  # Example continuous variables
    X_cat = np.array([[0], [1], [0]])  # Example categorical variables (just encoded as 0 or 1)
    y = np.array([0.3, 0.7, 0.5])  # Objective values

    # Instantiate the SimpleCoCaBO object
    optimizer = SimpleCoCaBO(continuous_dim=1, categorical_dim=1)

    # Fit the model to the data
    optimizer.fit(X_cont, X_cat, y)

    # Predict UCB values for new points
    new_cont = np.array([[0.6], [0.3]])  # New continuous points to evaluate
    new_cat = np.array([[1], [0]])  # New categorical points

    best_cont, best_cat = optimizer.optimize(new_cont, new_cat)
    print(f"Best continuous: {best_cont}, Best categorical: {best_cat}")
