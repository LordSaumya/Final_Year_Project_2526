import networkx as nx
from typing import Set, List, Optional, Tuple
from partitioning_utils import ( # ty: ignore[unresolved-import]
    calculate_b_metric,
    partition_with_bk,
    generate_hamiltonian_graph,
)
from symmetry_scavenger import SymmetryScavenger  # ty: ignore[unresolved-import]
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hamiltonian_utils import PauliString, Hamiltonian


def normal_rlf(graph: nx.Graph) -> List[Set[int]]:
    """
    Partitions the graph using the standard Recursive Largest First (RLF) algorithm.

    This algorithm groups nodes into sets where all nodes in a set are
    mutually non-adjacent (i.e., they form an independent set, representing
    commuting Pauli strings).

    Args:
        graph: The non-commuting graph where an edge indicates non-commutation.
    Returns:
        A list of sets, where each set is a group of commuting nodes.
    """

    unassigned_nodes = set(graph.nodes())
    groups = []

    while unassigned_nodes:
        # Start a new group
        new_group = set()

        # Select the unassigned node with the largest degree in the subgraph of unassigned nodes
        start_node = max(unassigned_nodes, key=lambda node: graph.degree[node])
        
        new_group.add(start_node)
        
        # This is the set of nodes that can be added to the new_group
        neighbors_of_S = set(graph.neighbors(start_node))
        potential_candidates = unassigned_nodes.difference(neighbors_of_S.union({start_node}))

        while potential_candidates:
            best_candidate = None
            max_b_value = -1
            min_c_value = float('inf')

            # Find best candidate
            for candidate in potential_candidates:
                neighbours = set(graph.neighbors(candidate))
                b_value = len(neighbours.intersection(neighbors_of_S))
                c_value = len(neighbours.intersection(potential_candidates))

                # Select candidate with maximum b_value (most connections to excluded)
                if b_value > max_b_value:
                    max_b_value = b_value
                    min_c_value = c_value
                    best_candidate = candidate
                # If tied on b_value, prefer minimum c_value (fewest connections to candidates)
                elif b_value == max_b_value:
                    if c_value < min_c_value:
                        min_c_value = c_value
                        best_candidate = candidate
            
            # If no candidate is found (e.g., potential_candidates is empty), break.
            if best_candidate is None:
                if potential_candidates:
                    best_candidate = potential_candidates.copy().pop()
                else:
                    break

            new_group.add(best_candidate)
            
            # Update potential_candidates and neighbors_of_S efficiently
            potential_candidates.remove(best_candidate)
            best_candidate_neighbors = set(graph.neighbors(best_candidate))
            potential_candidates.difference_update(best_candidate_neighbors)
            neighbors_of_S.update(best_candidate_neighbors)

        groups.append(new_group)
        unassigned_nodes.difference_update(new_group)

    return groups


def enhanced_rlf(graph: nx.Graph) -> List[Set[int]]:
    """
    Partitions the graph using an enhanced Recursive Largest First (RLF) algorithm.

    This algorithm groups nodes into sets where all nodes in a set are
    mutually non-adjacent (i.e., they form an independent set, representing
    commuting Pauli strings).

    Args:
        graph: The non-commuting graph where an edge indicates non-commutation.

    Returns:
        A list of sets, where each set is a group of commuting nodes.
    """
    unassigned_nodes = set(graph.nodes())
    groups = []

    while unassigned_nodes:
        # 2.a: Start a new group
        new_group = set()

        # 2.b: Select the unassigned node with the largest degree
        start_node = max(unassigned_nodes, key=lambda node: graph.degree[node])
        
        new_group.add(start_node)
        unassigned_nodes.remove(start_node)

        # 2.c & 2.d: Iteratively add the best commuting node
        # Pre-calculate the neighbors of the current group S (new_group)
        neighbors_of_S = set(graph.neighbors(start_node))
        potential_candidates = unassigned_nodes.difference(neighbors_of_S)

        # 2.c & 2.d: Iteratively add the best commuting node
        while potential_candidates:
            # 2.c.i: Compute B(u, G) for all candidates
            candidates_with_b_metric = []
            for u in potential_candidates:
                # Pass the pre-calculated neighbors_of_S to the metric function
                b_metric = calculate_b_metric(graph, u, unassigned_nodes, neighbors_of_S)
                candidates_with_b_metric.append((b_metric, u))
            
            # 2.c.ii: Select the term with the highest B(u, G)
            if not candidates_with_b_metric:
                break # No more candidates to add
            
            best_candidate = max(candidates_with_b_metric, key=lambda item: item[0])[1]

            new_group.add(best_candidate)
            unassigned_nodes.remove(best_candidate)
            
            # Update potential_candidates and neighbors_of_S efficiently
            potential_candidates.remove(best_candidate)
            
            # Get neighbors of the new node and update both sets
            best_candidate_neighbors = set(graph.neighbors(best_candidate))
            potential_candidates.difference_update(best_candidate_neighbors)
            neighbors_of_S.update(best_candidate_neighbors)
        
        # 2.e: Mark all items in G as assigned (done by removal) and store the group
        groups.append(new_group)

    return groups

def bron_kerbosch_optimiser(
    graph: nx.Graph,
    partitions: List[Set[int]],
    optimisation_percentage: float = 0.6
) -> List[Set[int]]:
    """
    Optimises a percentage of the smallest partitions using maximal clique search.

    This function sorts partitions by size, identifies the bottom percentage
    to optimise, pools their nodes, and re-partitions them by finding maximal
    cliques in the corresponding commuting graph.

    Args:
        graph: The original non-commuting graph.
        partitions: A list of partitions (independent sets).
        optimisation_percentage: The percentage of smallest partitions to optimise.

    Returns:
        A new list of partitions, with small groups replaced by maximal cliques.
    """
    if not partitions:
        return []

    # Sort partitions by size to identify the smallest ones
    sorted_partitions = sorted(partitions, key=len)
    
    # Determine the number of partitions to optimise based on the percentage
    num_to_optimise = int(len(sorted_partitions) * optimisation_percentage)

    # If there are no partitions to optimise, return the original list
    if num_to_optimise == 0:
        return partitions

    # Split into partitions to optimise and those to keep
    small_partitions = sorted_partitions[:num_to_optimise]
    large_partitions = sorted_partitions[num_to_optimise:]

    if not small_partitions:
        return partitions

    # Pool all nodes from the small partitions
    pool_nodes = set().union(*small_partitions)

    if not pool_nodes:
        return partitions

    # The commuting graph is the complement of the non-commuting graph.
    commuting_subgraph = nx.complement(graph.subgraph(pool_nodes))

    # Partition the commuting subgraph into disjoint maximal cliques using Bron-Kerbosch.
    clique_partitions = partition_with_bk(commuting_subgraph)

    # The final set of partitions includes the original large ones and the new cliques.
    new_partitions = large_partitions + clique_partitions

    return new_partitions


def build_commutation_graph(hamiltonian: List[PauliString]) -> nx.Graph:
    """
    Build a commutation graph from a list of Pauli strings.
    
    Nodes are PauliStrings, and an edge exists between two nodes if 
    the corresponding Pauli strings COMMUTE (i.e., this is the commuting graph).
    
    Args:
        hamiltonian: List of PauliString objects.
        
    Returns:
        A NetworkX graph where edges indicate commutation.
    """
    G = nx.Graph()
    G.add_nodes_from(hamiltonian)
    
    # Add edges between commuting pairs
    for i in range(len(hamiltonian)):
        for j in range(i + 1, len(hamiltonian)):
            if hamiltonian[i].commutes_with(hamiltonian[j]):
                G.add_edge(hamiltonian[i], hamiltonian[j])
    
    return G


def build_non_commutation_graph(hamiltonian: List[PauliString]) -> nx.Graph:
    """
    Build a non-commutation graph from a list of Pauli strings.
    
    Nodes are PauliStrings, and an edge exists between two nodes if 
    the corresponding Pauli strings do NOT commute.
    
    Args:
        hamiltonian: List of PauliString objects.
        
    Returns:
        A NetworkX graph where edges indicate non-commutation.
    """
    G = nx.Graph()
    G.add_nodes_from(hamiltonian)
    
    # Add edges between non-commuting pairs
    for i in range(len(hamiltonian)):
        for j in range(i + 1, len(hamiltonian)):
            if not hamiltonian[i].commutes_with(hamiltonian[j]):
                G.add_edge(hamiltonian[i], hamiltonian[j])
    
    return G


def greedy_grow(
    seed_clique: Set[PauliString], 
    unassigned: Set[PauliString], 
    comm_graph: nx.Graph
) -> Set[PauliString]:
    """
    Greedily grow a clique by adding commuting terms from unassigned set.
    
    This is a fast helper to maximize a scavenged clique without using
    the expensive B(u,S) metric. It simply checks commutation using
    the pre-built graph for O(1) lookup per term.
    
    Args:
        seed_clique: The initial set of commuting PauliStrings.
        unassigned: The set of available PauliStrings to potentially add.
        comm_graph: The commutation graph (edges = terms commute).
        
    Returns:
        The maximized clique containing seed_clique plus additional terms.
    """
    result = seed_clique.copy()
    candidates = unassigned - result
    
    for term in candidates:
        # Check if term commutes with ALL members of result
        commutes_with_all = True
        for existing in result:
            if not comm_graph.has_edge(term, existing):
                commutes_with_all = False
                break
        
        if commutes_with_all:
            result.add(term)
    
    return result


def standard_erlf_discovery(
    unassigned: Set[PauliString],
    non_comm_graph: nx.Graph
) -> Set[PauliString]:
    """
    Use enhanced RLF logic to discover a new clique (independent set in non-comm graph).
    
    This implements the expensive heuristic that maximizes using the B(u,S) metric.
    
    Args:
        unassigned: Set of unassigned PauliStrings.
        non_comm_graph: The non-commutation graph (edges = terms don't commute).
        
    Returns:
        A new group (independent set) of commuting PauliStrings.
    """
    if not unassigned:
        return set()
    
    # Work with the subgraph of unassigned nodes
    subgraph = non_comm_graph.subgraph(unassigned)
    
    new_group: Set[PauliString] = set()
    available = set(unassigned)
    
    # Select the unassigned node with the largest degree in subgraph
    start_node = max(available, key=lambda node: subgraph.degree[node])
    
    new_group.add(start_node)
    available.remove(start_node)
    
    # Track neighbors of the current group S
    neighbors_of_S = set(subgraph.neighbors(start_node))
    
    # Potential candidates are those that commute with start_node
    # (i.e., NOT neighbors in non-commutation graph)
    potential_candidates = available - neighbors_of_S
    
    while potential_candidates:
        # Compute B(u, S) for all candidates
        candidates_with_b_metric = []
        for u in potential_candidates:
            b_metric = calculate_b_metric(subgraph, u, available, neighbors_of_S)
            candidates_with_b_metric.append((b_metric, u))
        
        if not candidates_with_b_metric:
            break
        
        # Select the term with the highest B(u, S)
        best_candidate = max(candidates_with_b_metric, key=lambda item: item[0])[1]
        
        new_group.add(best_candidate)
        available.remove(best_candidate)
        
        # Update potential_candidates and neighbors_of_S
        potential_candidates.remove(best_candidate)
        best_candidate_neighbors = set(subgraph.neighbors(best_candidate))
        potential_candidates -= best_candidate_neighbors
        neighbors_of_S.update(best_candidate_neighbors)
    
    return new_group


def run_tail_optimization(
    partitions: List[Set[PauliString]],
    non_comm_graph: nx.Graph,
    percent: float = 0.6
) -> List[Set[PauliString]]:
    """
    Optimize the smallest partitions using Bron-Kerbosch maximal clique search.
    
    Collects the bottom `percent` of partitions by size, pools their nodes,
    and re-partitions using maximal clique search on the commuting graph.
    
    Args:
        partitions: List of partition sets.
        non_comm_graph: The non-commutation graph.
        percent: Fraction of smallest partitions to optimize (default 0.6).
        
    Returns:
        Optimized list of partitions.
    """
    if not partitions:
        return []
    
    # Convert PauliString sets to node indices for compatibility with existing function
    return bron_kerbosch_optimiser(non_comm_graph, partitions, percent)


def s3_rebroke(
    hamiltonian: List[PauliString],
    generators: List[Tuple[int, ...]],
    tau: float = 0.9,
    tail_opt_percent: float = 0.6,
    verbose: bool = False
) -> List[Set[PauliString]]:
    """
    S3-REBROKE: Symmetry-Scavenging REBROKE algorithm.
    
    An enhanced partitioning algorithm that exploits symmetry to reduce
    computational cost. It uses expensive heuristics only for discovering
    unique clique seeds, then uses symmetry generators to find related
    cliques cheaply via group theory.
    
    Algorithm:
    1. Setup: Build commutation graph and initialize scavenger.
    2. Main Loop:
       - PATH A (Fast): Try to get a candidate from scavenger queue.
         If valid, grow it greedily without expensive metrics.
       - PATH B (Slow): Use standard ERLF to discover a new clique.
         Push the discovery to scavenger to find its orbit.
    3. Tail Optimization: Re-partition smallest groups using Bron-Kerbosch.
    
    Args:
        hamiltonian: List of PauliStrings representing the Hamiltonian terms.
        generators: List of permutation maps representing symmetry generators.
                    Each map is a tuple where perm_map[i] = new position of qubit i.
        tau: Acceptance threshold (0.0 - 1.0) for the scavenger.
             Higher values enforce stricter partition quality. Default: 0.9.
        tail_opt_percent: Fraction of smallest partitions to optimize with
                          Bron-Kerbosch. Default: 0.6.
        verbose: If True, print progress information. Default: False.
        
    Returns:
        List of sets, where each set is a group of commuting PauliStrings.
        
    Example:
        >>> from hamiltonian_utils import PauliString
        >>> terms = [PauliString.from_string(s) for s in ["ZZII", "IZZI", "IIZZ", "ZIIZ"]]
        >>> generators = [(1, 2, 3, 0)]  # Cyclic translation
        >>> partitions = s3_rebroke(terms, generators, tau=0.8)
    """
    # 1. Setup
    unassigned = set(hamiltonian)
    partitions: List[Set[PauliString]] = []
    
    # Build both graph types
    comm_graph = build_commutation_graph(hamiltonian)
    non_comm_graph = build_non_commutation_graph(hamiltonian)
    
    scavenger = SymmetryScavenger(generators, tau)
    
    if verbose:
        print(f"S3-REBROKE: {len(hamiltonian)} terms, {len(generators)} generators, tau={tau}")
    
    iteration = 0
    scavenged_count = 0
    discovered_count = 0
    
    # 2. Main Loop
    while unassigned:
        iteration += 1
        new_group: Optional[Set[PauliString]] = None
        
        # --- PATH A: SCAVENGING (Fast) ---
        seed_remnant = scavenger.pop_candidate(unassigned)
        
        if seed_remnant:
            # Fast grow using simple commutation checks (no expensive B metric)
            new_group = greedy_grow(seed_remnant, unassigned, comm_graph)
            scavenged_count += 1
            
            if verbose:
                print(f"  Iter {iteration}: SCAVENGED group of size {len(new_group)}")
        
        # --- PATH B: DISCOVERY (Slow) ---
        else:
            # Use existing REBROKE ERLF logic
            new_group = standard_erlf_discovery(unassigned, non_comm_graph)
            discovered_count += 1
            
            if verbose:
                print(f"  Iter {iteration}: DISCOVERED group of size {len(new_group)}")
            
            # Feed the expensive find into the scavenger to find its orbit
            if new_group:
                scavenger.push_seed(new_group)
        
        # --- UPDATE ---
        if new_group:
            partitions.append(new_group)
            unassigned -= new_group
    
    if verbose:
        stats = scavenger.get_stats()
        print(f"\nS3-REBROKE complete:")
        print(f"  Total partitions: {len(partitions)}")
        print(f"  Discovered (slow): {discovered_count}")
        print(f"  Scavenged (fast): {scavenged_count}")
        print(f"  Scavenger stats: {stats}")
    
    # 3. Tail Optimization
    if tail_opt_percent > 0 and len(partitions) > 1:
        final_partitions = run_tail_optimization(
            partitions, 
            non_comm_graph, 
            percent=tail_opt_percent
        )
    else:
        final_partitions = partitions
    
    return final_partitions


def rebroke(
    hamiltonian: List[PauliString],
    tail_opt_percent: float = 0.6,
    verbose: bool = False
) -> List[Set[PauliString]]:
    """
    Standard REBROKE algorithm (without symmetry scavenging).
    
    Uses enhanced RLF for initial partitioning, followed by Bron-Kerbosch
    tail optimization on the smallest partitions.
    
    Args:
        hamiltonian: List of PauliStrings representing the Hamiltonian terms.
        tail_opt_percent: Fraction of smallest partitions to optimize. Default: 0.6.
        verbose: If True, print progress information. Default: False.
        
    Returns:
        List of sets, where each set is a group of commuting PauliStrings.
    """
    # Build non-commutation graph
    non_comm_graph = build_non_commutation_graph(hamiltonian)
    
    # Use enhanced RLF for initial partitioning
    unassigned = set(hamiltonian)
    partitions: List[Set[PauliString]] = []
    
    while unassigned:
        new_group = standard_erlf_discovery(unassigned, non_comm_graph)
        if new_group:
            partitions.append(new_group)
            unassigned -= new_group
    
    if verbose:
        print(f"REBROKE: {len(partitions)} initial partitions")
    
    # Tail optimization
    if tail_opt_percent > 0 and len(partitions) > 1:
        final_partitions = run_tail_optimization(
            partitions,
            non_comm_graph,
            percent=tail_opt_percent
        )
    else:
        final_partitions = partitions
    
    if verbose:
        print(f"REBROKE: {len(final_partitions)} final partitions after optimization")
    
    return final_partitions

if __name__ == "__main__":
    print("This module provides partitioning algorithms and is not intended to be run directly.")
    sys.exit(0)