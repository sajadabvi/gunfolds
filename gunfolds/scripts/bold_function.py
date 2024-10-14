import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import ode
import os

# Set plotting style
plt.style.use('Solarize_Light2')

# Machine epsilon for float type
eps = np.finfo(float).eps

def bold(t, y, ut):
    """
    Computes the derivatives for the BOLD signal model.

    Parameters:
    - t: Time variable
    - y: State variables [s, f, v, q]
    - ut: Input neuronal activity at time t

    Returns:
    - dy: Derivatives of the state variables
    """
    # Parameters for the BOLD model
    epsilon = 0.6       # Neuronal efficacy
    tau_s = 1.54        # Signal decay
    tau_f = 2.43        # Flow-dependent elimination
    tau_0 = 0.98        # Hemodynamic transit time
    alpha = 0.32        # Grubb's exponent
    E_0 = 0.34          # Resting oxygen extraction fraction

    # Unpack state variables
    s, f, v, q = y
    # Clip the values of f, v, q to prevent them from exceeding reasonable bounds.
    f = np.clip(f, 1e-5, 1e5)
    v = np.clip(v, 1e-5, 1e5)
    q = np.clip(q, 1e-5, 1e5)
    # Compute flow-dependent volume change
    fout = v ** (1 / alpha)

    # Differential equations
    ds = epsilon * ut - s / tau_s - (f - 1) / tau_f
    df = s
    dv = (f - fout) / tau_0
    exponent = 1 / (f + eps)
    if exponent > 700:  # To prevent overflow in np.exp
        exponent = 700
    power_term = np.exp(exponent * np.log(1 - E_0))
    dq = (f * (1 - power_term) / E_0 - fout * q / (v + eps)) / tau_0

    return np.array([ds, df, dv, dq])

def outbold(y):
    """
    Computes the BOLD signal from the state variables.

    Parameters:
    - y: State variables over time

    Returns:
    - BOLD signal
    """
    V_0 = 0.02
    E_0 = 0.34
    k1 = 7 * E_0
    k2 = 2
    k3 = 2 * E_0 - 0.2

    v = y[2, :]
    q = y[3, :]

    return V_0 * (k1 * (1 - q) + k2 * (1 - q / (v + eps)) + k3 * (1 - v))

def save_node_data(nodes):
    """
    Saves node data to CSV files.

    Parameters:
    - nodes: List of node data arrays
    """
    for idx, node in enumerate(nodes, start=1):
        filename = f'node{idx}.csv'
        pd.DataFrame(node).to_csv(filename, index=False)

def save_time_series_data(data, prefix, sampling_rates):
    """
    Saves time series data at different sampling rates.

    Parameters:
    - data: Time series data array (nodes x time points)
    - prefix: Prefix for the filename
    - sampling_rates: List of sampling intervals
    """
    for rate in sampling_rates:
        filename = f'{prefix}_{rate}.csv'
        df = pd.DataFrame({
            f'X{idx + 1}': data[idx, ::rate]
            for idx in range(data.shape[0])
        })
        df.to_csv(filename, index=False)

def compute_bold_signals(ut, end_time=100):
    """
    Computes the BOLD signals for the given neuronal time series.

    Parameters:
    - ut: ndarray of shape (num_nodes, num_timepoints), neuronal time series
    - end_time: float, total time duration

    Returns:
    - nodes_array: ndarray of shape (num_nodes, num_timepoints), BOLD signals
    - t_eval: ndarray of time points
    """
    num_nodes, num_timepoints = ut.shape

    # Define time variables
    timestep = end_time / (num_timepoints - 1)
    t_eval = np.linspace(0, end_time, num_timepoints)

    # Compute the BOLD signals
    nodes = []
    for node_idx in range(num_nodes):
        # Define the BOLD model for the current node
        def bold_model(t, y):
            idx = min(int(round(t / timestep)), num_timepoints - 1)
            ut_node = ut[node_idx, idx]
            dy = bold(t, y, ut_node)
            if not np.all(np.isfinite(dy)):
                print(f"Non-finite derivative at time {t}: dy = {dy}, y = {y}")
            return dy

            # return bold(t, y, ut_node)

        # Set up the ODE solver
        r = ode(bold_model).set_integrator('vode', method='bdf')
        r.set_initial_value(0.95 * np.array([1, 0.1, 1, 1]), 0)

        # Integrate over time
        vals = []
        while r.successful() and r.t < end_time:
            r.integrate(r.t + timestep)
            vals.append(r.y)
        vals = np.array(vals).T  # Transpose for consistency

        # Compute the BOLD signal and store it
        nodes.append(outbold(vals))

    nodes_array = np.array(nodes)

    return nodes_array, t_eval

def plot_bold_signals(nodes_array, t_eval):
    """
    Plots the BOLD signals over time.

    Parameters:
    - nodes_array: ndarray of shape (num_nodes, num_timepoints), BOLD signals
    - t_eval: ndarray of time points
    """
    num_nodes = nodes_array.shape[0]
    plt.figure(figsize=(18, 4))
    plt.plot(t_eval, nodes_array.T, lw=0.5)
    plt.xlabel('Time (seconds)')
    plt.ylabel('BOLD Signal')
    plt.tight_layout()
    plt.legend([f'Node {i+1}' for i in range(num_nodes)], loc='upper right')
    plt.show()

if __name__ == "__main__":
    # Example usage
    # Load neuronal data from a file or any other source
    # Replace the path with your actual data file
    path = '~/DataSets_Feedbacks/8_VAR_simulation/ringmore/u2/txtSTD/data1.txt'
    data = pd.read_csv(path, delimiter='\t')
    ut = data.values.T  # Transpose if necessary to get shape (num_nodes, num_timepoints)

    # Compute BOLD signals
    nodes_array, t_eval = compute_bold_signals(ut, end_time=100)

    # Plot the results
    plot_bold_signals(nodes_array, t_eval)
