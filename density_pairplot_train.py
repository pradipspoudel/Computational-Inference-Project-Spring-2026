import torch
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import random

from normalizingFlow import NormalizingFlow
from utils import set_seed

def main():
    set_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the original training data
    print("Loading original dataset...")
    df_train = pd.read_csv('datasets/train_500.csv')
    raw_data = df_train.values
    
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

    # Generate new samples from the model
    print("Generating samples from the Normalizing Flow...")
    num_samples = df_train.shape[0] 
    generated_samples_scaled = model.sample(num_samples).cpu().numpy()
    
    # Inverse transform to put samples back in the original data space
    generated_samples = scaler.inverse_transform(generated_samples_scaled)

    # Prepare DataFrames for Seaborn plotting
    print("Preparing plot...")
    # Column names X1, X2, X3, X4
    cols = [f'X{i+1}' for i in range(dim)]
    
    df_orig = pd.DataFrame(raw_data, columns=cols)
    df_orig['Source'] = 'Original Data'
    
    df_gen = pd.DataFrame(generated_samples, columns=cols)
    df_gen['Source'] = 'NF Generated'
    
    # Combine both into a single DataFrame
    df_plot = pd.concat([df_orig, df_gen], ignore_index=True)

    # Create the PairGrid plot
    palette = {"Original Data": "blue", "NF Generated": "orange"}
    
    g = sns.PairGrid(df_plot, hue="Source", palette=palette, corner=False)
    
    # Diagonal plots: 1D Kernel Density Estimates (filled)
    g.map_diag(sns.kdeplot, fill=True, alpha=0.3, linewidth=1.5)
    
    # Off-diagonal plots: 2D Kernel Density Estimates (contours)
    g.map_offdiag(sns.kdeplot, fill=False, alpha=0.6, linewidths=1.2)
    
    # Add a legend
    g.add_legend(title="Source", adjust_subtitles=True)

    # Adjust layout and save
    plt.subplots_adjust(top=0.95)
    g.fig.suptitle('Density Comparison: Original vs Normalizing Flow Generated', fontsize=16)
    
    output_filename = 'plots/density_pairplot_train.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully saved plot to '{output_filename}'")

if __name__ == "__main__":
    main()