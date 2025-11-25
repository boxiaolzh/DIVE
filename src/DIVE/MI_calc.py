import numpy as np
from sklearn.feature_selection import mutual_info_regression
import torch


def calculate_mutual_information(X, Y, n_neighbors=3, n_repeats=10):
    """
    Calculate the average mutual information between input X and output Y.

    Parameters:
    X -- Input data, 2D array of size (samples, n_inputs)
    Y -- Output data, 2D array of size (samples, n_outputs)
    n_neighbors -- Number of neighbors to use
    n_repeats -- Number of repetitions to obtain average value

    Returns:
    mi_avg -- Array of average mutual information values
    """
    if len(X) == 0 or len(Y) == 0:
        # Note: return 2D array here
        n_inputs = X.shape[1]
        return np.zeros((1, n_inputs))

    if len(X) != len(Y):
        min_len = min(len(X), len(Y))
        X = X[:min_len]
        Y = Y[:min_len]

    n_samples, n_inputs = X.shape
    n_outputs = Y.shape[1] if Y.ndim > 1 else 1
    mi_avg = np.zeros((n_outputs, n_inputs))

    # Ensure Y is a 2D array
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    # Use smaller number of neighbors if sample size is small
    actual_n_neighbors = min(n_neighbors, len(X) - 1)

    try:
        for repeat in range(n_repeats):
            for output_index in range(n_outputs):
                mi = mutual_info_regression(X, Y[:, output_index],
                                            n_neighbors=actual_n_neighbors, random_state=42)
                mi_avg[output_index] += mi

        mi_avg /= n_repeats
    except Exception as e:
        print(f"Error in MI calculation: {e}")
        print(f"X shape: {X.shape}, Y shape: {Y.shape}")
        # Return 2D array with correct shape to avoid subsequent indexing errors
        return np.zeros((n_outputs, n_inputs))

    if n_outputs == 1:
        # Ensure return of 2D array (1, n_inputs)
        return mi_avg.reshape(1, -1)
    return mi_avg

def cal_score(mi_results, index, weights):
    return sum(mi_results[i][index] * weight for i, weight in enumerate(weights))


# Calculate MI scores
def calculate_scores(dbx, dby, gain_num, GBW_num, phase_num, iter, init_num, n_neighbors=3, n_repeats=10,
                     input_dim=12):
    """
    Equation 5: Iobj(xi) = I(xi; y0) - Objective-related MI
    Equation 6: Icon(xi) = Σ ωj · I(xi; yj) - Constraint-related MI
    Equation 7: ωj = e^(-Nj/M) - Constraint weight
    Equation 8: S(xi) = Iobj(xi) + Icon(xi) - Final score
    """
    if isinstance(dbx, torch.Tensor):
        dbx = dbx.numpy()
    if isinstance(dby, torch.Tensor):
        dby = dby.numpy()

    # Ensure dby is a 2D array
    if len(dby.shape) == 1:
        dby = dby.reshape(-1, 1)
    elif len(dby.shape) == 3:
        dby = dby.reshape(len(dby), -1)

    # Call function to calculate mutual information - returns (n_outputs, n_inputs) matrix
    mi_results = calculate_mutual_information(dbx, dby, n_neighbors=n_neighbors, n_repeats=n_repeats)

    # Total iterations M for current iteration
    M = iter + init_num + 1

    # Initialize score array
    scores_list = []

    # Calculate CawMI score for each input dimension
    for i in range(input_dim):
        # Equation 5: Objective-related MI - I_obj(xi) = I(xi; y0)
        # y0 is the current target (index 1)
        I_obj = mi_results[1, i]  # Current is the main optimization objective

        # Equations 6 and 7: Constraint-related MI - I_con(xi) = Σ ωj · I(xi; yj)
        I_con = 0.0

        # Constraint 1: Gain (index 0)
        omega_gain = np.exp(-gain_num / M)  # Equation 7
        I_con += omega_gain * mi_results[0, i]

        # Constraint 2: Phase (index 2)
        omega_phase = np.exp(-phase_num / M)  # Equation 7
        I_con += omega_phase * mi_results[2, i]

        # Constraint 3: GBW (index 3)
        omega_gbw = np.exp(-GBW_num / M)  # Equation 7
        I_con += omega_gbw * mi_results[3, i]

        # Equation 8: Final score S(xi) = I_obj(xi) + I_con(xi)
        S_xi = I_obj + I_con
        scores_list.append(S_xi)

    scores = torch.tensor(scores_list)

    # Print debug information
    print(f"CawMI Analysis Results (Iteration {iter}):")
    print(
        f"  Constraint weights: gain={np.exp(-gain_num / M):.3f}, phase={np.exp(-phase_num / M):.3f}, gbw={np.exp(-GBW_num / M):.3f}")
    for i, mi in enumerate(mi_results):
        print(f"  Mutual information with output {i + 1}: {mi}")
    print(f"  Final CawMI scores: {scores[:5].numpy()}")  # Display first 5 dimensions

    return scores
