import torch
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import random

from normalizingFlow import NormalizingFlow
from utils import set_seed

def main():
    set_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the original training data (Assuming no header)
    print("Loading original training dataset...")
    df_train = pd.read_csv('datasets/train_500.csv') 
    
    # Extract just the first coordinate (X1) for the original data
    x1_original = df_train.values[:, 0]
    
    # Load the fitted scaler
    print("Loading scaler...")
    scaler = joblib.load('artifacts/scaler.pkl')
    
    # Initialize and load the trained model
    print("Loading trained model...")

    model_params_df = pd.read_csv("artifacts/model_params.csv")
    dim = model_params_df.loc[0, "dimension"]
    hidden_dim = model_params_df.loc[0, "hidden_dimension"]  
    num_layers = model_params_df.loc[0, "num_layers"]
    
    model = NormalizingFlow(dim, hidden_dim, num_layers).to(device)
    model.load_state_dict(torch.load('artifacts/nf_model.pth', map_location=device))
    model.eval()

    # Sample exactly count_train_data points from the model
    print(f"Generating {df_train.shape[0]} samples from the Normalizing Flow...")
    num_samples = df_train.shape[0]
    with torch.no_grad():
        generated_samples_scaled = model.sample(num_samples).cpu().numpy()
    
    # Inverse transform to put samples back in the original data space
    generated_samples = scaler.inverse_transform(generated_samples_scaled)
    
    # Extract the first coordinate (X1) for the generated data
    x1_generated = generated_samples[:, 0]

    # Plotting
    print("Plotting distributions...")
    plt.figure(figsize=(7, 6))
    
    # Use hex colors similar to the ones in your image
    color_original = '#6389df' # A nice cornflower blue
    color_generated = '#dfbc75' # A muted golden/sand color
    
    # Plot histograms as densities
    plt.hist(x1_original, bins=40, density=True, alpha=0.7, color=color_original, label='Original Data')
    plt.hist(x1_generated, bins=40, density=True, alpha=0.7, color=color_generated, label='NF Generated')
    
    # Add labels and formatting
    plt.title('Marginal Distribution of X1 (First Coordinate)', fontsize=14)
    plt.legend(loc='upper right')
    
    # Save the plot
    output_filename = 'plots/marginal_distribution_x1.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully saved plot to '{output_filename}'")

if __name__ == "__main__":
    main()