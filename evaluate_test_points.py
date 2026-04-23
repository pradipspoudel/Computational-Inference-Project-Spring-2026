import torch
import pandas as pd
import numpy as np
import joblib  # Added to load the scaler
from sklearn.preprocessing import StandardScaler

from normalizingFlow import NormalizingFlow

class DensityEstimator:
    def __init__(self, flow_model, scaler):
        """
        Wraps the trained normalizing flow and the scikit-learn scaler 
        to provide a rigorous log density function for original-scale data.
        """
        self.flow_model = flow_model
        self.scaler = scaler
        
        # Calculate the log determinant of the scaler's Jacobian
        # scaler.scale_ holds the standard deviations (sigma) for each feature
        self.scaler_log_det = np.sum(np.log(self.scaler.scale_))
        
    def log_density(self, x):
        """
        Calculates the exact log density of the inputs.
        
        Args:
            x (np.ndarray): The 4D data points in the original space. Shape (N, 4).
            
        Returns:
            np.ndarray: The log densities for each point. Shape (N,).
        """
        # Ensure model is in eval mode and move tensors to the correct device
        self.flow_model.eval()
        device = next(self.flow_model.parameters()).device
        
        # Scale the input data using the fitted scaler
        x_scaled = self.scaler.transform(x)
        
        # Convert to PyTorch tensor and move to device
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)
        
        # Pass through the flow model to get the log likelihood of the scaled data
        with torch.no_grad():
            _, log_prob_scaled = self.flow_model(x_tensor)
            log_prob_scaled = log_prob_scaled.cpu().numpy()
            
        # Apply the change of variables formula for the scaler
        log_prob_original = log_prob_scaled - self.scaler_log_det
        
        return log_prob_original

def evaluate_test_points(trained_model, fitted_scaler, test_points):
    # Initialize the density function
    density_fn = DensityEstimator(trained_model, fitted_scaler)
    
    print(f"\nEvaluating {len(test_points)} test points from the other team...")
    
    # Calculate log densities!
    log_densities = density_fn.log_density(test_points)
    
    print(f"Calculated log densities for {len(log_densities)} points.")
    print(f"Sample log densities (first 5): {log_densities[:5]}")
    print(f"Average log density of test set: {np.mean(log_densities):.4f}")
    
    return log_densities

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize and load the trained model
    print("Loading trained model 'nf_model.pth'...")

    model_params_df = pd.read_csv("artifacts/model_params.csv")
    dim = model_params_df.loc[0, "dimension"]
    hidden_dim = model_params_df.loc[0, "hidden_dimension"]  
    num_layers = model_params_df.loc[0, "num_layers"]
    
    model = NormalizingFlow(dim, hidden_dim, num_layers).to(device)
    model.load_state_dict(torch.load('artifacts/nf_model.pth', map_location=device))
    model.eval()

    # Load the fitted scaler
    print("Loading scaler 'scaler.pkl'...")
    scaler = joblib.load('artifacts/scaler.pkl')

    # Load the test dataset
    print("Loading test dataset...")
    df_test = pd.read_csv('datasets/test_5000.csv')
    
    # Extract just the raw numerical values
    test_points = df_test.values

    # Evaluate the test points
    log_densities = evaluate_test_points(model, scaler, test_points)
    
    df_test['log_density'] = log_densities
    output_filename = 'results/test_5000_evaluated.csv'
    df_test.to_csv(output_filename, index=False)
    print(f"\nSaved evaluated results to '{output_filename}'")

if __name__ == "__main__":
    main()