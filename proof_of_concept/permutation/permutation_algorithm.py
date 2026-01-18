import networkx as nx
import sys
import os
import math

# Add current directory to path for imports when called from other modules
sys.path.insert(0, os.path.dirname(__file__))

from permutation_utils import (  # noqa: E402
    hamiltonian_to_graph,
    check_permutation_symmetry,
    find_minimal_generators,
    visualise_hamiltonian_graph
)
import itertools
from typing import Dict, Tuple, List, Set
from multiprocessing import Pool, cpu_count
from functools import partial

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hamiltonian_utils import PauliString, Hamiltonian

def _map_tuple_to_cycles(perm_tuple: Tuple[int, ...]) -> List[List[int]]:
    """Converts a permutation map tuple, e.g. (1, 2, 0), to cycles."""
    n = len(perm_tuple)
    cycles = []
    visited = [False] * n
    for i in range(n):
        if visited[i]:
            continue
        cycle = []
        current = i
        while not visited[current]:
            visited[current] = True
            cycle.append(current)
            current = perm_tuple[current] # Follow the map
            if current == i:
                break
        if len(cycle) > 1:
            cycles.append(cycle)
    return cycles

def _check_permutation_batch(
    perm_batch: List[Tuple[int, ...]],
    signature_lookup: Set[Tuple[float, frozenset[Tuple[int, str]]]],
    term_signatures_list: List[Tuple[float, Set[Tuple[int, str]]]],
    identity_tuple: Tuple[int, ...]
) -> Tuple[List[List[List[int]]], int]:
    """Helper function to check a batch of permutations in parallel.
    
    Args:
        perm_batch: List of permutation tuples to check
        signature_lookup: Frozen set of valid term signatures
        term_signatures_list: List of (coefficient, signature_set) tuples
        identity_tuple: The identity permutation for comparison
        
    Returns:
        Tuple of (found_symmetries_cycles, valid_count)
    """
    found_symmetries = []
    valid_count = 0
    
    for perm_map in perm_batch:
        is_symmetry = True
        
        for coeff, sig_set in term_signatures_list:
            permuted_sig_set = set()
            for q_index, edge_color in sig_set:
                permuted_q_index = perm_map[q_index]
                permuted_sig_set.add((permuted_q_index, edge_color))
            
            target_sig = (coeff, frozenset(permuted_sig_set))
            
            if target_sig not in signature_lookup:
                is_symmetry = False
                break
        
        if is_symmetry:
            valid_count += 1
            if perm_map != identity_tuple:
                cycles = _map_tuple_to_cycles(perm_map)
                if cycles:
                    found_symmetries.append(cycles)
    
    return found_symmetries, valid_count


def find_automorphism_group_bruteforce_graph(
    graph: nx.Graph,
    parallel: bool = True,
    n_processes: int | None = None
) -> tuple[List[List[List[int]]], int]:
    """Finds the permutation symmetries of a Hamiltonian graph via brute force.

    WARNING: This is O(n! * m * k) and is not scalable beyond n ~ 10.
    
    This function implements the brute-force check by:
    1. Generating all n! permutations of the qubit nodes.
    2. For each permutation, checking if it preserves the Hamiltonian's
       structure (term coefficients and edge colors), as required
       by the paper's proof .

    Args:
        graph (nx.Graph): The Hamiltonian represented as a graph.
        parallel (bool): Whether to use parallel processing. Default True.
        n_processes (int): Number of processes to use. Default uses all CPUs.

    Returns:
        Tuple[List[List[List[int]]], int]: A tuple containing a list of
        all non-identity symmetry ELEMENTS and the total size of the group.
    """
    
    # --- 1. Pre-computation: Build Term Signatures ---
    
    # A "signature" for a term node is its coefficient (vertex color)
    # and the set of its (qubit_index, edge_color) connections.
    term_signatures: Dict[str, Tuple[float, Set[Tuple[int, str]]]] = {}
    
    # A lookup set for fast O(1) checking of a signature's existence.
    signature_lookup: Set[Tuple[float, frozenset[Tuple[int, str]]]] = set()
    
    qubit_nodes = []
    
    for node_name, data in graph.nodes(data=True):
        if data.get('bipartite') == 1: # This is a term node
            # Get vertex color C_V(t_r) = ("term", c_r)
            coeff = data['color'][1]
            sig_set = set()
            
            # Find all neighbors (which must be qubits)
            for neighbor in graph.neighbors(node_name):
                # Get qubit index (e.g., "q0" -> 0)
                q_index = int(neighbor[1:])
                # Get edge color C_E(e) [cite: 345]
                edge_color = graph.edges[node_name, neighbor]['color']
                sig_set.add((q_index, edge_color))
            
            term_signatures[node_name] = (coeff, sig_set)
            signature_lookup.add((coeff, frozenset(sig_set)))
            
        elif data.get('bipartite') == 0: # This is a qubit node
            qubit_nodes.append(node_name)
    
    n_qubits = len(qubit_nodes)
    if n_qubits == 0:
        return [], 1
    
    identity_tuple = tuple(range(n_qubits))
    
    if not parallel or n_qubits <= 6:  # Use serial for small problems
        found_symmetries_cycles = []
        group_size = 0

        # Iterate over all n! permutations of qubit indices
        for perm_map in itertools.permutations(range(n_qubits)):
            is_symmetry = True
            
            # Check if this permutation is a valid automorphism
            for coeff, sig_set in term_signatures.values():
                
                # Apply the permutation to the signature
                permuted_sig_set = set()
                for q_index, edge_color in sig_set:
                    permuted_q_index = perm_map[q_index] # Map index i to perm_map[i]
                    permuted_sig_set.add((permuted_q_index, edge_color))
                
                # Create the target signature we must find in the graph
                target_sig = (coeff, frozenset(permuted_sig_set))
                
                # Check if this permuted signature exists
                if target_sig not in signature_lookup:
                    is_symmetry = False
                    break # This permutation is invalid, stop checking its terms            
            if is_symmetry:
                group_size += 1
                if perm_map != identity_tuple:
                    cycles = _map_tuple_to_cycles(perm_map)
                    if cycles:
                        found_symmetries_cycles.append(cycles)
                    
        return find_minimal_generators(found_symmetries_cycles, n_qubits), group_size
    
    # --- Parallel version ---
    if n_processes is None:
        n_processes = max(1, cpu_count() - 2)
    
    # Convert term_signatures dict to list for pickling
    term_signatures_list = list(term_signatures.values())
    
    # Split permutations into batches
    # Calculate total permutations to determine batch size
    total_perms = math.factorial(n_qubits)
    batch_size = max(1, total_perms // (n_processes * 4))  # 4 batches per process
    
    def perm_batch_generator():
        """Yields batches of permutations without creating the full list."""
        it = itertools.permutations(range(n_qubits))
        while True:
            batch = list(itertools.islice(it, batch_size))
            if not batch:
                break
            yield batch
    
    # Create worker function with fixed arguments
    worker_fn = partial(
        _check_permutation_batch,
        signature_lookup=signature_lookup,
        term_signatures_list=term_signatures_list,
        identity_tuple=identity_tuple
    )
    
    # Process batches in parallel
    found_symmetries_cycles = []
    group_size = 0
    
    with Pool(processes=n_processes) as pool:
        # Use imap to lazily generate and process batches
        # imap_unordered is slightly faster as we don't care about order
        for batch_symmetries, batch_count in pool.imap_unordered(worker_fn, perm_batch_generator()):
            found_symmetries_cycles.extend(batch_symmetries)
            group_size += batch_count
    
    return find_minimal_generators(found_symmetries_cycles, n_qubits), group_size

def print_results(hamiltonian_name: str, generators: List[List[List[int]]], group_size: int):
    print(f"\nHamiltonian: {hamiltonian_name}")
    print(f"Found {group_size} symmetries, with generators:")
    for gen in generators:
        print(f"  {gen}")


if __name__ == "__main__":
    hamiltonian = Hamiltonian.heisenberg_1d(n_qubits=5, Jx=1.0, Jy=1.0, Jz=1.0, periodic=True)

    # Find symmetries via graph brute-force
    graph = hamiltonian_to_graph(hamiltonian)
    generators, group_size = find_automorphism_group_bruteforce_graph(graph, parallel=True)
    print_results("Custom Hamiltonian", generators, group_size)
    