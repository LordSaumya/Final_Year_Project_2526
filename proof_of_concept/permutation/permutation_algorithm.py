import networkx as nx
from permutation_utils import (
    hamiltonian_to_graph,
    _map_to_cycles,
    visualise_hamiltonian_graph,
    random_symmetry_generators,
    generate_symmetric_hamiltonian_int,
    are_generators_equivalent,
    check_permutation_symmetry,
    _hamiltonian_to_string_int
)
import itertools
from collections import defaultdict
from typing import Dict, Tuple, List, Set
from multiprocessing import Pool, cpu_count
from functools import partial


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
    n_processes: int = None
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
                    
        return found_symmetries_cycles, group_size
    
    # --- Parallel version ---
    if n_processes is None:
        n_processes = max(1, cpu_count() - 2)
    
    # Convert term_signatures dict to list for pickling
    term_signatures_list = list(term_signatures.values())
    
    # Split permutations into batches
    all_perms = list(itertools.permutations(range(n_qubits)))
    batch_size = max(1, len(all_perms) // (n_processes * 4))  # 4 batches per process
    perm_batches = [all_perms[i:i + batch_size] for i in range(0, len(all_perms), batch_size)]
    
    # Create worker function with fixed arguments
    worker_fn = partial(
        _check_permutation_batch,
        signature_lookup=signature_lookup,
        term_signatures_list=term_signatures_list,
        identity_tuple=identity_tuple
    )
    
    # Process batches in parallel
    with Pool(processes=n_processes) as pool:
        results = pool.map(worker_fn, perm_batches)
    
    # Combine results
    found_symmetries_cycles = []
    group_size = 0
    for batch_symmetries, batch_count in results:
        found_symmetries_cycles.extend(batch_symmetries)
        group_size += batch_count
    
    return found_symmetries_cycles, group_size


if __name__ == "__main__":
    n_qubits = 12
    n_seed_terms = 4
    locality = 4
    symmetries = random_symmetry_generators(n_qubits, 5, locality)
    print("Symmetries used to generate Hamiltonian:", symmetries)

    hamiltonian = generate_symmetric_hamiltonian_int(
        n_qubits, symmetries, n_seed_terms, locality
    )

    generators, group_size = find_automorphism_group_bruteforce_graph(hamiltonian_to_graph(hamiltonian))
    print("Generators (in disjoint cycle format):", generators)
    print("Group size:", group_size)

    symmetries_present = True
    for gen in generators:
        if not check_permutation_symmetry(hamiltonian, [gen], n_qubits):
            symmetries_present = False
            print("Hamiltonian:", _hamiltonian_to_string_int(hamiltonian))
            print("Symmetry violation found for generator:", gen)
            print("The found generators do NOT match the original symmetries.")
            break
    if symmetries_present:
        print("The found generators match the original symmetries.")