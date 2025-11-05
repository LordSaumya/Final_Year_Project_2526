import networkx as nx
from typing import Set, List
from partitioning_utils import ( # ty: ignore[unresolved-import]
    calculate_b_metric,
    partition_with_bk,
)
import sys


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

if __name__ == "__main__":
    print("This module provides partitioning algorithms and is not intended to be run directly.")
    sys.exit(0)