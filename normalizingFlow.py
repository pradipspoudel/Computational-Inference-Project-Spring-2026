import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from utils import set_seed

# ==========================================
# Normalizing Flow Components
# ==========================================

class AffineCouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim, mask):
        super().__init__()
        # Register mask as a buffer so it automatically moves to the GPU with the model
        self.register_buffer('mask', mask)
        
        # Neural network to predict scale (s) and translation (t)
        # It only operates on the masked part of the input
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim * 2)
        )
        
    def forward(self, x):
        # x_masked is passed through the network, x_unmasked is kept as is
        x_masked = x * self.mask
        out = self.net(x_masked)
        
        s, t = out.chunk(2, dim=-1)
        # Apply mask to s and t so they only affect the unmasked variables
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)
        
        # Scale s to avoid extreme values and numerical instability
        s = torch.tanh(s) 
        
        z = x * torch.exp(s) + t
        log_det_jacobian = s.sum(dim=-1)
        
        return z, log_det_jacobian

    def inverse(self, z):
        z_masked = z * self.mask
        out = self.net(z_masked)
        
        s, t = out.chunk(2, dim=-1)
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)
        
        s = torch.tanh(s)
        
        x = (z - t) * torch.exp(-s)
        return x

class NormalizingFlow(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        
        # Register prior parameters as buffers so they move to the GPU
        self.register_buffer('prior_loc', torch.zeros(dim))
        self.register_buffer('prior_cov', torch.eye(dim))
        
        # Create alternating masks for the coupling layers
        mask1 = torch.tensor([1.0, 0.0, 1.0, 0.0])
        mask2 = torch.tensor([0.0, 1.0, 0.0, 1.0])
        
        for i in range(num_layers):
            mask = mask1 if i % 2 == 0 else mask2
            self.layers.append(AffineCouplingLayer(dim, hidden_dim, mask))
            
    @property
    def prior(self):
        return torch.distributions.MultivariateNormal(self.prior_loc, self.prior_cov)
            
    def forward(self, x):
        log_det_jacobian_total = torch.zeros(x.shape[0], device=x.device)
        z = x
        for layer in self.layers:
            z, log_det_jacobian = layer(z)
            log_det_jacobian_total += log_det_jacobian
            
        # Log likelihood under the standard normal prior
        prior_log_prob = self.prior.log_prob(z)
        log_likelihood = prior_log_prob + log_det_jacobian_total
        return z, log_likelihood

    def sample(self, num_samples):
        with torch.no_grad():
            z = self.prior.sample((num_samples,))
            x = z
            # Pass backwards through the flow
            for layer in reversed(self.layers):
                x = layer.inverse(x)
            return x

# ==========================================
# Training and Evaluation Functions
# ==========================================

def train_flow(model, dataloader, epochs, lr, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.train()
    epoch_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()

            _, log_likelihood = model(batch)
            loss = -log_likelihood.mean()
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        scheduler.step()
        
        avg_loss = epoch_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        
        if (epoch + 1) % 100 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
            
    return epoch_losses

def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    df = pd.read_csv('datasets/train_500.csv')
    raw_data = df.values
    
    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(raw_data)
    
    # Convert to PyTorch dataset
    tensor_data = torch.tensor(data_scaled, dtype=torch.float32)
    dataloader = torch.utils.data.DataLoader(tensor_data, batch_size=64, shuffle=True)
    
    # Initialize Model
    dim = 4
    hidden_dim = 64  
    num_layers = 10   
    epochs = 1000
    lr = 1e-3
    
    # Move model to device
    model = NormalizingFlow(dim, hidden_dim, num_layers).to(device)
    
    # Train Model
    print("Training Normalizing Flow Model...")
    epoch_losses = train_flow(model, dataloader, epochs=epochs, lr=lr, device=device)
    
    # ---------------------------------------------------------
    # Saving Artifacts
    # ---------------------------------------------------------
    print("\nSaving artifacts...")
    # Save the model
    torch.save(model.state_dict(), 'artifacts/nf_model.pth')
    print(" - Saved model weights to 'nf_model.pth'")
    
    # Save the scaler
    joblib.dump(scaler, 'artifacts/scaler.pkl')
    print(" - Saved scaler to 'scaler.pkl'")
    
    # Save loss per epoch
    loss_df = pd.DataFrame({'Epoch': range(1, epochs + 1), 'Loss': epoch_losses})
    loss_df.to_csv('results/training_loss.csv', index=False)
    print(" - Saved training loss to 'training_loss.csv'")
    
    # Make a plot for the loss as a function of epochs
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), epoch_losses, label='Training Loss', color='blue')
    plt.title('Negative Log Likelihood vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('plots/training_loss_plot.png', dpi=300)
    plt.close()
    print(" - Saved loss plot to 'training_loss_plot.png'")
    # ---------------------------------------------------------
    
    # Estimate the 99% CI for the 1st coordinate
    print("\nSampling from the estimated distribution...")
    model.eval()
    
    num_samples = 100000
    generated_samples_scaled = model.sample(num_samples).cpu().numpy()
    
    # Inverse transform to get back to the original data space
    generated_samples = scaler.inverse_transform(generated_samples_scaled)
    
    first_coord_samples = generated_samples[:, 0]
    
    # Calculate the 99% Confidence (Percentile) Interval
    lower_bound = np.percentile(first_coord_samples, 0.5)
    upper_bound = np.percentile(first_coord_samples, 99.5)
    mean_val = np.mean(first_coord_samples)

    ci_df = pd.DataFrame({
        "coordinate": ["X1"],
        "lower_bound": [lower_bound],
        "upper_bound": [upper_bound]
    })

    ci_df.to_csv("results/nf_99ci.csv", index=False)

    model_param_df = pd.DataFrame({
        "dimension": [dim],
        "hidden_dimension": [hidden_dim],
        "num_layers": [num_layers],
        "learning_rate": [lr]
    })

    model_param_df.to_csv("artifacts/model_params.csv", index=False)

    print("\n=== RESULTS ===")
    print(f"Original Data - Coord 1 Mean: {np.mean(raw_data[:, 0]):.4f}")
    print(f"Modeled Data  - Coord 1 Mean: {mean_val:.4f}")
    print(f"99% Confidence Interval for Coordinate 1: [{lower_bound:.4f}, {upper_bound:.4f}]")

if __name__ == "__main__":
    main()