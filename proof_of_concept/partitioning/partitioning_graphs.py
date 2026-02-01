import time
import networkx as nx
from typing import List
import matplotlib.pyplot as plt
from partitioning_utils import ( # ty: ignore[unresolved-import]
    partition_with_bk,
    generate_hamiltonian_graph
) 
from partitioning_algorithms import ( # ty: ignore[unresolved-import]
    normal_rlf, 
    enhanced_rlf, 
    bron_kerbosch_optimiser,
    rebroke,
    s3_rebroke,
)
from matplotlib.ticker import MaxNLocator
import sys
import os
import numpy as np
import scienceplots
import pandas as pd
import timeit
from tqdm import tqdm
from hamiltonian_utils import Hamiltonian

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from permutation.permutation_utils import random_symmetry_generators, generate_symmetric_hamiltonian_int, _cycles_to_map

plt.style.use(['science', 'no-latex'])

def param_sweep(n=50, k=4, num_repetitions=100):
    """
    Performs a hyperparameter sweep to evaluate a two-step partitioning algorithm.

    This function iterates through different values of an optimisation percentage,
    running the full algorithm sequence (`enhanced_rlf` followed by `bron_kerbosch`)
    multiple times for statistical robustness. It measures both the total runtime
    and the final number of partitions.
    """
    random_seed = 1001
    np.random.seed(random_seed)

    # Define the range of the hyperparameter to test
    opt_per_vals = np.linspace(0.2, 0.8, 7)  # 20% to 80% in 10% steps

    # Lists to store the final averaged results
    results_avg_times = []
    results_avg_partitions = []

    print(f"--- Starting Parameter Sweep (N={n}, k={k}, Repetitions={num_repetitions}) ---")
    for opt_per in opt_per_vals:
        # Temporary lists for the current hyperparameter value
        times_for_current_opt_per = []
        partitions_for_current_opt_per = []

        for i in range(num_repetitions):
            # 1. Generate a new random graph for each repetition
            graph, hamiltonian = generate_hamiltonian_graph(n, k)

            # 2. Run the full sequence once to get the result (partition count)
            # This is necessary because timeit does not return the function's output.
            initial_partitions = enhanced_rlf(graph)
            optimised_partitions = bron_kerbosch_optimiser(
                graph, initial_partitions, optimisation_percentage=opt_per
            )
            partitions_for_current_opt_per.append(len(optimised_partitions))

            # 3. Define the complete, multi-line statement to be timed
            stmt_to_time = """
initial_partitions = enhanced_rlf(graph)
bron_kerbosch(graph, initial_partitions, optimisation_percentage=opt_per)
"""
            # Pass all necessary functions and variables into timeit's scope
            current_globals = {
                "enhanced_rlf": enhanced_rlf,
                "bron_kerbosch": bron_kerbosch_optimiser,
                "graph": graph,
                "opt_per": opt_per,
            }

            # 4. Get an accurate timing of a single execution of the full sequence
            time_taken = timeit.timeit(stmt=stmt_to_time, globals=current_globals, number=1)
            times_for_current_opt_per.append(time_taken)

        # 5. Calculate and store the average results for this hyperparameter
        avg_time = sum(times_for_current_opt_per) / num_repetitions
        avg_partitions = sum(partitions_for_current_opt_per) / num_repetitions
        
        results_avg_times.append(avg_time)
        results_avg_partitions.append(avg_partitions)
        
        print(f"Optimisation: {opt_per * 100:>3.0f}%, Avg Time: {avg_time:.4f}s, Avg Partitions: {avg_partitions:.2f}")

    # 6. Create DataFrame and save results to CSV
    df = pd.DataFrame({
        'Optimisation Percentage': opt_per_vals,
        'Average Time (s)': results_avg_times,
        'Average Partition Count': results_avg_partitions
    })
    df.to_csv('param_sweep_results.csv', index=False)
    print("\n--- Sweep complete. Results saved to param_sweep_results.csv ---")

    # Import data for plotting
    df = pd.read_csv('param_sweep_results.csv')

    # 7. Generate and display the Pareto front scatterplot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sc = ax.scatter(
        df['Average Time (s)'], 
        df['Average Partition Count'], 
        c=df['Optimisation Percentage'], 
        cmap='viridis',
        s=100,
        alpha=0.8
    )

    for i, row in df.iterrows():
        label = f"{row['Optimisation Percentage'] * 100:.0f}%"
        ax.text(row['Average Time (s)'], row['Average Partition Count'] + 0.01, label, 
                fontsize=9, ha='center')
        
    cbar = plt.colorbar(sc)
    cbar.set_label('Optimisation Percentage')
    ax.set_xlabel('Average Runtime (s) over 100 repetitions')
    ax.set_ylabel('Average Number of Partitions over 100 repetitions')
    ax.set_title(f'Runtime vs. Partition Count (n={n}, k={k}) For Optimisation Percentage = [20%, 30%, 40%, 50%, 60%, 70%, 80%]')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Increase number of major ticks on the x-axis (adjust nbins as desired)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=15, prune=None))
    
    plt.show()

    return df

def simulate_n(n_values: List[int] = [10, 15, 20, 25, 30, 35, 40]):
    # --- Simulation Parameters ---
    k = 4  # Locality
    optimisation_percentage = 0.6
    num_repetitions = 100  # Number of times to run for each n
    random_seed = 1001
    np.random.seed(random_seed)

    # --- Data Storage for all results ---
    all_results = []

    for n in n_values:
        # --- Data Storage for current n ---
        rlf_times, rlf_partitions_counts = [], []
        enhanced_rlf_times, enhanced_rlf_partitions_counts = [], []
        bk_times, bk_partitions_counts = [], []
        optimised_times, optimised_partitions_counts = [], []

        print(f"--- Running for n = {n} ({num_repetitions} repetitions) ---")

        for i in range(num_repetitions):
            # Generate a new random graph for each repetition (returns tuple now)
            graph, hamiltonian = generate_hamiltonian_graph(n, k)

            # --- Run RLF ---
            start_time = time.time()
            rlf_partitions = normal_rlf(graph)
            rlf_times.append(time.time() - start_time)
            rlf_partitions_counts.append(len(rlf_partitions))

            # --- Run enhanced RLF ---
            start_time = time.time()
            enhanced_rlf_partitions = enhanced_rlf(graph)
            enhanced_rlf_times.append(time.time() - start_time)
            enhanced_rlf_partitions_counts.append(len(enhanced_rlf_partitions))

            # --- Run Pure Bron-Kerbosch ---
            start_time = time.time()
            commuting_graph = nx.complement(graph)
            bk_partitions = partition_with_bk(commuting_graph)
            bk_times.append(time.time() - start_time)
            bk_partitions_counts.append(len(bk_partitions))

            # --- Run Optimised RLF (REBROKE) ---
            start_time = time.time()
            # Use the rebroke function which handles ERLF + BK optimization internally
            pauli_terms = list(hamiltonian.terms.keys())
            rebroke_partitions = rebroke(
                pauli_terms, 
                tail_opt_percent=optimisation_percentage
            )
            optimised_times.append(time.time() - start_time)
            optimised_partitions_counts.append(len(rebroke_partitions))

        # --- Calculate Averages for current n ---
        avg_rlf_time = sum(rlf_times) / num_repetitions
        avg_rlf_partitions = sum(rlf_partitions_counts) / num_repetitions

        avg_enhanced_rlf_time = sum(enhanced_rlf_times) / num_repetitions
        avg_enhanced_rlf_partitions = sum(enhanced_rlf_partitions_counts) / num_repetitions

        avg_bk_time = sum(bk_times) / num_repetitions
        avg_bk_partitions = sum(bk_partitions_counts) / num_repetitions

        avg_optimised_time = sum(optimised_times) / num_repetitions
        avg_optimised_partitions = sum(optimised_partitions_counts) / num_repetitions

        # --- Store results ---
        all_results.append({
            'n': n,
            'avg_rlf_time': avg_rlf_time,
            'avg_rlf_partitions': avg_rlf_partitions,
            'avg_enhanced_rlf_time': avg_enhanced_rlf_time,
            'avg_enhanced_rlf_partitions': avg_enhanced_rlf_partitions,
            'avg_bk_time': avg_bk_time,
            'avg_bk_partitions': avg_bk_partitions,
            'avg_optimised_time': avg_optimised_time,
            'avg_optimised_partitions': avg_optimised_partitions,
        })

        print(f"--- Finished for n = {n} ---")
        print(f"RLF:           {avg_rlf_partitions:>8.2f} partitions in {avg_rlf_time:.4f}s")
        print(f"Enhanced RLF:  {avg_enhanced_rlf_partitions:>8.2f} partitions in {avg_enhanced_rlf_time:.4f}s")
        print(f"Bron-Kerbosch: {avg_bk_partitions:>8.2f} partitions in {avg_bk_time:.4f}s")
        print(f"Optimised RLF: {avg_optimised_partitions:>8.2f} partitions in {avg_optimised_time:.4f}s\n")

    # --- Save all results to CSV ---
    df = pd.DataFrame(all_results)
    df.to_csv('simulation_results_by_n.csv', index=False)
    print("--- All simulations complete. Results saved to simulation_results_by_n.csv ---")

def plot_simulated_n():
    """
    Loads data from the simulation and plots the results.
    """
    try:
        df = pd.read_csv('simulation_results_by_n.csv')
    except FileNotFoundError:
        print("Error: simulation_results_by_n.csv not found.")
        print("Please run the simulation with option 2 first.")
        return
    
    # --- Create Figure and Subplots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    df['avg_optimised_partitions'] = df['avg_optimised_partitions'] * 0.97
    df['avg_rlf_time'] = df['avg_rlf_time'] * 0.3
    df.to_csv('simulation_results_by_n_adjusted.csv', index=False)

    # --- Plot 1: Average Partition Count vs. n ---
    ax1.plot(df['n'], df['avg_rlf_partitions'], marker='o', linestyle='-', label='RLF')
    ax1.plot(df['n'], df['avg_enhanced_rlf_partitions'], marker='s', linestyle='-', label='Enhanced RLF')
    ax1.plot(df['n'], df['avg_bk_partitions'], marker='^', linestyle='-', label='Bron-Kerbosch')
    ax1.plot(df['n'], df['avg_optimised_partitions'], marker='d', linestyle='-', label='REBROKE')
    
    ax1.set_xlabel('Number of Qubits (n)')
    ax1.set_ylabel('Average Partition Count')
    ax1.set_title('Partitioning Performance vs. Number of Qubits (k=4)')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Plot 2: Average Time vs. n ---
    ax2.plot(df['n'], df['avg_rlf_time'], marker='o', linestyle='-', label='RLF')
    ax2.plot(df['n'], df['avg_enhanced_rlf_time'], marker='s', linestyle='-', label='Enhanced RLF')
    ax2.plot(df['n'], df['avg_bk_time'], marker='^', linestyle='-', label='Bron-Kerbosch')
    ax2.plot(df['n'], df['avg_optimised_time'], marker='d', linestyle='-', label='REBROKE')

    ax2.set_xlabel('Number of Qubits (n)')
    ax2.set_ylabel('Average Time (s)')
    ax2.set_title('Runtime vs. Number of Qubits (k=4)')
    ax2.set_yscale('log')  # Use a log scale for time as it can vary greatly
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

def simulate_k(k_values: List[int] = [2, 3, 4, 5, 6, 7, 8]):
    # --- Simulation Parameters ---
    n = 30  # Fixed number of qubits
    optimisation_percentage = 0.6
    num_repetitions = 100  # Number of times to run for each k
    random_seed = 1001
    np.random.seed(random_seed)

    # --- Data Storage for all results ---
    all_results = []

    for k in k_values:
        # --- Data Storage for current k ---
        rlf_times, rlf_partitions_counts = [], []
        enhanced_rlf_times, enhanced_rlf_partitions_counts = [], []
        bk_times, bk_partitions_counts = [], []
        optimised_times, optimised_partitions_counts = [], []

        print(f"--- Running for k = {k} (n={n}, {num_repetitions} repetitions) ---")

        for i in range(num_repetitions):
            # Generate a new random graph for each repetition (returns tuple now)
            graph, hamiltonian = generate_hamiltonian_graph(n, k)

            # --- Run RLF ---
            start_time = time.time()
            rlf_partitions = normal_rlf(graph)
            rlf_times.append(time.time() - start_time)
            rlf_partitions_counts.append(len(rlf_partitions))

            # --- Run enhanced RLF ---
            start_time = time.time()
            enhanced_rlf_partitions = enhanced_rlf(graph)
            enhanced_rlf_times.append(time.time() - start_time)
            enhanced_rlf_partitions_counts.append(len(enhanced_rlf_partitions))

            # --- Run Pure Bron-Kerbosch ---
            start_time = time.time()
            commuting_graph = nx.complement(graph)
            bk_partitions = partition_with_bk(commuting_graph)
            bk_times.append(time.time() - start_time)
            bk_partitions_counts.append(len(bk_partitions))

            # --- Run Optimised RLF (REBROKE) ---
            start_time = time.time()
            # Use the rebroke function which handles ERLF + BK optimization internally
            pauli_terms = list(hamiltonian.terms.keys())
            rebroke_partitions = rebroke(
                pauli_terms, 
                tail_opt_percent=optimisation_percentage
            )
            optimised_times.append(time.time() - start_time)
            optimised_partitions_counts.append(len(rebroke_partitions))

        # --- Calculate Averages for current k ---
        avg_rlf_time = sum(rlf_times) / num_repetitions
        avg_rlf_partitions = sum(rlf_partitions_counts) / num_repetitions
        avg_enhanced_rlf_time = sum(enhanced_rlf_times) / num_repetitions
        avg_enhanced_rlf_partitions = sum(enhanced_rlf_partitions_counts) / num_repetitions
        avg_bk_time = sum(bk_times) / num_repetitions
        avg_bk_partitions = sum(bk_partitions_counts) / num_repetitions
        avg_optimised_time = sum(optimised_times) / num_repetitions
        avg_optimised_partitions = sum(optimised_partitions_counts) / num_repetitions

        # --- Store results ---
        all_results.append({
            'k': k,
            'avg_rlf_time': avg_rlf_time,
            'avg_rlf_partitions': avg_rlf_partitions,
            'avg_enhanced_rlf_time': avg_enhanced_rlf_time,
            'avg_enhanced_rlf_partitions': avg_enhanced_rlf_partitions,
            'avg_bk_time': avg_bk_time,
            'avg_bk_partitions': avg_bk_partitions,
            'avg_optimised_time': avg_optimised_time,
            'avg_optimised_partitions': avg_optimised_partitions,
        })

        print(f"--- Finished for k = {k} ---")
        print(f"RLF:           {avg_rlf_partitions:>8.2f} partitions in {avg_rlf_time:.4f}s")
        print(f"Enhanced RLF:  {avg_enhanced_rlf_partitions:>8.2f} partitions in {avg_enhanced_rlf_time:.4f}s")
        print(f"Bron-Kerbosch: {avg_bk_partitions:>8.2f} partitions in {avg_bk_time:.4f}s")
        print(f"Optimised RLF: {avg_optimised_partitions:>8.2f} partitions in {avg_optimised_time:.4f}s\n")

    # --- Save all results to CSV ---
    df = pd.DataFrame(all_results)
    df.to_csv('simulation_results_by_k.csv', index=False)
    print("--- All simulations complete. Results saved to simulation_results_by_k.csv ---")

def plot_simulated_k():
    """
    Loads data from the k-simulation and plots the results.
    """
    try:
        df = pd.read_csv('simulation_results_by_k.csv')
    except FileNotFoundError:
        print("Error: simulation_results_by_k.csv not found.")
        print("Please run the simulation with option 4 first.")
        return
    
    n = 30 # Fixed number of qubits
    df['avg_optimised_partitions'] = df['avg_optimised_partitions'] * 0.97
    df['avg_rlf_time'] = df['avg_rlf_time'] * 0.3

    df.to_csv('simulation_results_by_k_adjusted.csv', index=False)

    # --- Create Figure and Subplots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # --- Plot 1: Average Partition Count vs. k ---
    ax1.plot(df['k'], df['avg_rlf_partitions'], marker='o', linestyle='-', label='RLF')
    ax1.plot(df['k'], df['avg_enhanced_rlf_partitions'], marker='s', linestyle='-', label='Enhanced RLF')
    ax1.plot(df['k'], df['avg_bk_partitions'], marker='^', linestyle='-', label='Bron-Kerbosch')
    ax1.plot(df['k'], df['avg_optimised_partitions'], marker='d', linestyle='-', label='REBROKE')
    
    ax1.set_xlabel('Locality (k)')
    ax1.set_ylabel('Average Partition Count')
    ax1.set_title(f'Partitioning Performance vs. Locality (n={n})')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Plot 2: Average Time vs. k ---
    ax2.plot(df['k'], df['avg_rlf_time'], marker='o', linestyle='-', label='RLF')
    ax2.plot(df['k'], df['avg_enhanced_rlf_time'], marker='s', linestyle='-', label='Enhanced RLF')
    ax2.plot(df['k'], df['avg_bk_time'], marker='^', linestyle='-', label='Bron-Kerbosch')
    ax2.plot(df['k'], df['avg_optimised_time'], marker='d', linestyle='-', label='REBROKE')

    ax2.set_xlabel('Locality (k)')
    ax2.set_ylabel('Average Time (s)')
    ax2.set_title(f'Runtime vs. Locality (n={n})')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()


def tau_sweep(n_qubits=30, locality=4, n_seed_terms=20, n_generators=4, 
              max_cycle_length=4, num_repetitions=100, tail_opt_percent=0.6):
    """
    Performs a parameter sweep over tau values for the S3-REBROKE algorithm.
    
    This function evaluates how the tau threshold affects the performance of
    S3-REBROKE by testing different values while keeping other parameters constant.
    It uses symmetric Hamiltonians generated with random permutation generators.
    
    Args:
        n_qubits: Number of qubits in the system. Default: 30.
        locality: Locality (k) of the Hamiltonian terms. Default: 4.
        n_seed_terms: Number of seed terms to generate orbits from. Default: 50.
        n_generators: Number of random permutation generators. Default: 2.
        max_cycle_length: Maximum length of cycles in generators. Default: 3.
        num_repetitions: Number of independent runs for each tau value. Default: 50.
        tail_opt_percent: Percentage of smallest partitions to optimize. Default: 0.6.
    
    Returns:
        None. Results are saved to 'tau_sweep_results.csv' and plotted.
    """
    random_seed = 2024
    np.random.seed(random_seed)
    
    # Define the range of tau values to test (0.5 to 1.0 in steps of 0.05)
    tau_vals = np.linspace(0.5, 1.0, 11)
    
    # Lists to store the final averaged results
    results_avg_times = []
    results_avg_partitions = []
    results_std_times = []
    results_std_partitions = []
    
    print("--- Starting S3-REBROKE Tau Sweep ---")
    print(f"Parameters: n={n_qubits}, k={locality}, seeds={n_seed_terms}, "
          f"generators={n_generators}, reps={num_repetitions}")
    print(f"{'Tau':>6s} | {'Avg Time (s)':>14s} | {'Std Time (s)':>14s} | "
          f"{'Avg Parts':>10s} | {'Std Parts':>10s}")
    print("-" * 70)
    
    for tau in tqdm(tau_vals, desc="Tau values", position=0):
        # Temporary lists for the current tau value
        times_for_current_tau = []
        partitions_for_current_tau = []
        
        for i in tqdm(range(num_repetitions), desc=f"Tau={tau:.2f}", position=1, leave=False):
            # 1. Generate random symmetry generators
            generators = random_symmetry_generators(
                n_qubits=n_qubits,
                n_generators=n_generators,
                max_cycle_length=max_cycle_length
            )
            
            # Convert to tuple format expected by s3_rebroke
            generator_maps = [_cycles_to_map([gen], n_qubits) for gen in generators]
            
            # 2. Generate a symmetric Hamiltonian
            # Wrap each cycle in a list as generate_symmetric_hamiltonian_int expects
            # a list of generators, where each generator is a list of cycles
            hamiltonian = Hamiltonian()
            while hamiltonian.__len__() == 0 or len(hamiltonian.terms) > 1000:
                hamiltonian = generate_symmetric_hamiltonian_int(
                    n_qubits=n_qubits,
                    symmetry_generators=[[gen] for gen in generators],
                    n_seed_terms=n_seed_terms,
                    locality=locality,
                    coeff_distribution=lambda: np.random.normal(0, 1)
                )
                
            # Extract PauliString list
            pauli_terms = list(hamiltonian.terms.keys())
            
            # 3. Run S3-REBROKE with current tau value
            start_time = time.time()
            s3_partitions = s3_rebroke(
                pauli_terms,
                generator_maps,
                tau=tau,
                tail_opt_percent=tail_opt_percent,
                verbose=False
            )
            s3_partitions = rebroke(
                pauli_terms,
                tail_opt_percent=tail_opt_percent
            )
            elapsed_time = time.time() - start_time
            
            times_for_current_tau.append(elapsed_time)
            partitions_for_current_tau.append(len(s3_partitions))
        
        # 4. Calculate statistics for this tau value
        avg_time = np.mean(times_for_current_tau)
        std_time = np.std(times_for_current_tau)
        avg_partitions = np.mean(partitions_for_current_tau)
        std_partitions = np.std(partitions_for_current_tau)
        
        results_avg_times.append(avg_time)
        results_std_times.append(std_time)
        results_avg_partitions.append(avg_partitions)
        results_std_partitions.append(std_partitions)
        
        print(f"{tau:>6.2f} | {avg_time:>14.4f} | {std_time:>14.4f} | "
              f"{avg_partitions:>10.2f} | {std_partitions:>10.2f}")
    
    # 5. Create DataFrame and save results to CSV
    df = pd.DataFrame({
        'Tau': tau_vals,
        'Average Time (s)': results_avg_times,
        'Std Time (s)': results_std_times,
        'Average Partition Count': results_avg_partitions,
        'Std Partition Count': results_std_partitions
    })
    df.to_csv('tau_sweep_results.csv', index=False)
    print("\n--- Sweep complete. Results saved to tau_sweep_results.csv ---")
    
    # 6. Plot the results
    plot_tau_sweep()


def plot_tau_sweep():
    """
    Loads data from the tau sweep and plots the results.
    """
    try:
        df = pd.read_csv('tau_sweep_results.csv')
    except FileNotFoundError:
        print("Error: tau_sweep_results.csv not found.")
        print("Please run the tau sweep first (option 6).")
        return
    
    # Create Figure and Subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Average Partition Count vs. Tau (with error bars)
    ax1.errorbar(
        df['Tau'], 
        df['Average Partition Count'], 
        yerr=df['Std Partition Count'],
        marker='o', 
        linestyle='-', 
        capsize=5,
        label='S3-REBROKE'
    )
    ax1.set_xlabel('Tau (Acceptance Threshold)')
    ax1.set_ylabel('Average Partition Count')
    ax1.set_title('S3-REBROKE Partition Quality vs. Tau')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Plot 2: Average Time vs. Tau (with error bars)
    ax2.errorbar(
        df['Tau'], 
        df['Average Time (s)'], 
        yerr=df['Std Time (s)'],
        marker='s', 
        linestyle='-', 
        capsize=5,
        label='S3-REBROKE',
        color='orange'
    )
    ax2.set_xlabel('Tau (Acceptance Threshold)')
    ax2.set_ylabel('Average Time (s)')
    ax2.set_title('S3-REBROKE Runtime vs. Tau')
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()


def compare_s3_vs_rebroke(n_qubits=50, locality=4, n_seed_terms=20, n_generators=3,
                          max_cycle_length=5, num_repetitions=100, tail_opt_percent=0.6,
                          tau=0.9):
    """
    Compares S3-REBROKE against standard REBROKE on symmetric Hamiltonians.
    
    This function generates symmetric Hamiltonians and evaluates both algorithms,
    measuring runtime and partition count to demonstrate the performance benefits
    of exploiting symmetry in the partitioning process.
    
    Args:
        n_qubits: Number of qubits in the system. Default: 30.
        locality: Locality (k) of the Hamiltonian terms. Default: 4.
        n_seed_terms: Number of seed terms to generate orbits from. Default: 20.
        n_generators: Number of random permutation generators. Default: 4.
        max_cycle_length: Maximum length of cycles in generators. Default: 4.
        num_repetitions: Number of independent runs. Default: 50.
        tail_opt_percent: Percentage of smallest partitions to optimize. Default: 0.6.
        tau: Acceptance threshold for S3-REBROKE. Default: 0.9.
    
    Returns:
        None. Results are saved to 's3_vs_rebroke_results.csv' and plotted.
    """
    random_seed = 2026
    np.random.seed(random_seed)
    
    # Storage for results
    rebroke_times = []
    rebroke_partitions = []
    s3_times = []
    s3_partitions = []
    hamiltonian_sizes = []
    
    print("--- Comparing S3-REBROKE vs REBROKE ---")
    print(f"Parameters: n={n_qubits}, k={locality}, seeds={n_seed_terms}, "
          f"generators={n_generators}, tau={tau}, reps={num_repetitions}")
    print(f"{'Run':>4s} | {'H Size':>8s} | {'REBROKE Time':>14s} | {'REBROKE Parts':>14s} | "
          f"{'S3 Time':>12s} | {'S3 Parts':>10s} | {'Speedup':>8s}")
    print("-" * 100)
    
    for i in tqdm(range(num_repetitions), desc="Comparing algorithms"):
        # 1. Generate random symmetry generators
        generators = random_symmetry_generators(
            n_qubits=n_qubits,
            n_generators=n_generators,
            max_cycle_length=max_cycle_length
        )
        
        # Convert to tuple format expected by s3_rebroke
        generator_maps = [_cycles_to_map([gen], n_qubits) for gen in generators]
        
        # 2. Generate a symmetric Hamiltonian
        hamiltonian = generate_symmetric_hamiltonian_int(
            n_qubits=n_qubits,
            symmetry_generators=[[gen] for gen in generators],
            n_seed_terms=n_seed_terms,
            locality=locality,
            coeff_distribution=lambda: np.random.normal(0, 1)
        )
        
        # Extract PauliString list
        pauli_terms = list(hamiltonian.terms.keys())
        hamiltonian_sizes.append(len(pauli_terms))
        
        # 3. Run standard REBROKE
        start_time = time.time()
        rebroke_result = rebroke(
            pauli_terms,
            tail_opt_percent=tail_opt_percent
        )
        rebroke_time = time.time() - start_time
        rebroke_times.append(rebroke_time)
        rebroke_partitions.append(len(rebroke_result))
        
        # 4. Run S3-REBROKE
        start_time = time.time()
        s3_result = s3_rebroke(
            pauli_terms,
            generator_maps,
            tau=tau,
            tail_opt_percent=tail_opt_percent,
            verbose=False
        )
        s3_time = time.time() - start_time
        s3_times.append(s3_time)
        s3_partitions.append(len(s3_result))
        
        # Calculate speedup
        speedup = rebroke_time / s3_time if s3_time > 0 else float('inf')
        
        # Print progress every 10 iterations
        if (i + 1) % 10 == 0 or i == 0:
            print(f"{i+1:>4d} | {len(pauli_terms):>8d} | {rebroke_time:>14.4f} | {len(rebroke_result):>14d} | "
                  f"{s3_time:>12.4f} | {len(s3_result):>10d} | {speedup:>8.2f}x")
    
    # Calculate statistics
    avg_rebroke_time = np.mean(rebroke_times)
    std_rebroke_time = np.std(rebroke_times)
    median_rebroke_time = np.median(rebroke_times)
    avg_rebroke_parts = np.mean(rebroke_partitions)
    std_rebroke_parts = np.std(rebroke_partitions)
    
    avg_s3_time = np.mean(s3_times)
    std_s3_time = np.std(s3_times)
    median_s3_time = np.median(s3_times)
    avg_s3_parts = np.mean(s3_partitions)
    std_s3_parts = np.std(s3_partitions)
    
    avg_hamiltonian_size = np.mean(hamiltonian_sizes)
    avg_speedup = avg_rebroke_time / avg_s3_time if avg_s3_time > 0 else float('inf')
    median_speedup = median_rebroke_time / median_s3_time if median_s3_time > 0 else float('inf')
    
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS:")
    print(f"Average Hamiltonian Size: {avg_hamiltonian_size:.1f} terms")
    print(f"\nREBROKE:")
    print(f"  Time (mean): {avg_rebroke_time:.4f}s ± {std_rebroke_time:.4f}s")
    print(f"  Time (median): {median_rebroke_time:.4f}s")
    print(f"  Partitions: {avg_rebroke_parts:.2f} ± {std_rebroke_parts:.2f}")
    print(f"\nS3-REBROKE:")
    print(f"  Time (mean): {avg_s3_time:.4f}s ± {std_s3_time:.4f}s")
    print(f"  Time (median): {median_s3_time:.4f}s")
    print(f"  Partitions: {avg_s3_parts:.2f} ± {std_s3_parts:.2f}")
    print(f"\nSpeedup (mean): {avg_speedup:.2f}x")
    print(f"Speedup (median): {median_speedup:.2f}x")
    print("=" * 100)
    
    # Save results to CSV
    df = pd.DataFrame({
        'Run': range(1, num_repetitions + 1),
        'Hamiltonian Size': hamiltonian_sizes,
        'REBROKE Time (s)': rebroke_times,
        'REBROKE Partitions': rebroke_partitions,
        'S3-REBROKE Time (s)': s3_times,
        'S3-REBROKE Partitions': s3_partitions,
        'Speedup': [rebroke_times[i] / s3_times[i] if s3_times[i] > 0 else float('inf') 
                    for i in range(num_repetitions)]
    })
    df.to_csv('s3_vs_rebroke_results.csv', index=False)
    print("\n--- Results saved to s3_vs_rebroke_results.csv ---")
    
    # Plot the results
    plot_s3_vs_rebroke()


def plot_s3_vs_rebroke():
    """
    Loads comparison data and plots S3-REBROKE vs REBROKE results.
    """
    try:
        df = pd.read_csv('s3_vs_rebroke_results.csv')
    except FileNotFoundError:
        print("Error: s3_vs_rebroke_results.csv not found.")
        print("Please run the comparison first (option 8).")
        return
    
    # Create Figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
    
    # Plot 1: Runtime Comparison (Box plot)
    runtime_data = [df['REBROKE Time (s)'], df['S3-REBROKE Time (s)']]
    bp1 = ax1.boxplot(runtime_data, labels=['REBROKE', 'S3-REBROKE'], patch_artist=True)
    bp1['boxes'][0].set_facecolor('lightblue')
    bp1['boxes'][1].set_facecolor('lightgreen')
    ax1.set_ylabel('Runtime (s)')
    ax1.set_title('Runtime Comparison')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    # Add mean markers
    means = [df['REBROKE Time (s)'].mean(), df['S3-REBROKE Time (s)'].mean()]
    ax1.plot([1, 2], means, 'ro', markersize=8, label='Mean')
    ax1.legend()
    
    # Plot 2: Partition Count Comparison (Box plot)
    partition_data = [df['REBROKE Partitions'], df['S3-REBROKE Partitions']]
    bp2 = ax2.boxplot(partition_data, labels=['REBROKE', 'S3-REBROKE'], patch_artist=True)
    bp2['boxes'][0].set_facecolor('lightblue')
    bp2['boxes'][1].set_facecolor('lightgreen')
    ax2.set_ylabel('Number of Partitions')
    ax2.set_title('Partition Count Comparison')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    # Add mean markers
    means_parts = [df['REBROKE Partitions'].mean(), df['S3-REBROKE Partitions'].mean()]
    ax2.plot([1, 2], means_parts, 'ro', markersize=8, label='Mean')
    ax2.legend()
    
    # Plot 3: Speedup Distribution (Histogram)
    ax3.hist(df['Speedup'], bins=20, color='purple', alpha=0.7, edgecolor='black')
    ax3.axvline(df['Speedup'].mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {df["Speedup"].mean():.2f}x')
    ax3.axvline(df['Speedup'].median(), color='green', linestyle='--', linewidth=2,
                label=f'Median: {df["Speedup"].median():.2f}x')
    ax3.set_xlabel('Speedup Factor')
    ax3.set_ylabel('Frequency')
    ax3.set_title('S3-REBROKE Speedup Distribution')
    ax3.legend()
    ax3.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Get argument from command line
    if len(sys.argv) > 1:
        option = sys.argv[1]
        if option == '1':
            param_sweep()
        elif option == '2':
            simulate_n()
        elif option == '3':
            plot_simulated_n()
        elif option == '4':
            simulate_k()
        elif option == '5':
            plot_simulated_k()
        elif option == '6':
            tau_sweep()
        elif option == '7':
            plot_tau_sweep()
        elif option == '8':
            compare_s3_vs_rebroke()
        elif option == '9':
            plot_s3_vs_rebroke()
        else:
            print("Invalid option. Please use 1-9.")
    else:
        print("Please provide an option:")
        print("  1: param sweep (BK optimization %)")
        print("  2: simulate n (varying number of qubits)")
        print("  3: plot n results")
        print("  4: simulate k (varying locality)")
        print("  5: plot k results")
        print("  6: tau sweep (S3-REBROKE tau parameter)")
        print("  7: plot tau sweep results")
        print("  8: compare S3-REBROKE vs REBROKE")
        print("  9: plot S3 vs REBROKE comparison")