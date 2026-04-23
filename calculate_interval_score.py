import pandas as pd
import numpy as np

def calculate_s2_score(y, l, u, alpha=0.01):
    N = len(y)
    
    # Base interval width (u - l)
    term1 = u - l
    
    # Penalty for falling below the lower bound: (2/alpha) * (l - y) * 1(y < l)
    # y < l creates a boolean array that acts as the indicator function 1(...)
    term2 = (2 / alpha) * (l - y) * (y < l)
    
    # Penalty for exceeding the upper bound: (2/alpha) * (y - u) * 1(y > u)
    term3 = (2 / alpha) * (y - u) * (y > u)
    
    # Sum across all test points
    total_sum = np.sum(term1 + term2 + term3)
    
    # Final Interval Score S2
    s2 = - (1 / N) * total_sum
    
    return s2

def main():
    # Load the test dataset
    print("Loading test dataset 'test_5000.csv'...")
    df_test = pd.read_csv('datasets/test_5000.csv') 
    
    # Extract the first coordinate (y^1_i) for all test points
    y1_test = df_test.values[:, 0]
    
    # Define the lower (l) and upper (u) bounds
    ci_df = pd.read_csv("results/nf_99ci.csv")
    l = ci_df.loc[0, "lower_bound"]
    u = ci_df.loc[0, "upper_bound"]
    
    print(f"Using bounds: Lower (l) = {l}, Upper (u) = {u}")
    
    # Calculate S2 score
    s2_score = calculate_s2_score(y1_test, l, u, alpha=0.01)
    
    # Print results
    print("\n=== RESULTS ===")
    print(f"Number of test points (N): {len(y1_test)}")
    print(f"Interval Score (S2): {s2_score:.4f}")

if __name__ == "__main__":
    main()