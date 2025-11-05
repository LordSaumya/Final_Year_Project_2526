import networkx as nx
import numpy as np
from typing import Set, List, Tuple
import random
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hamiltonian_utils import PauliString, Hamiltonian


def generate_random_graph(n: int, p: float) -> nx.Graph:
    """
    Generates a random erdos-renyi graph with n nodes and edge probability p.

    This represents the non-commuting graph where an edge exists if two
    Pauli strings do not commute.

    Args:
        n: The number of nodes (Pauli strings).
        p: The probability of an edge existing between any two nodes.

    Returns:
        A networkx graph.
    """
    return nx.erdos_renyi_graph(n, p)


def calculate_b_metric(
    graph: nx.Graph, u: int, unassigned: Set[int], neighbors_of_S: Set[int]
) -> int:
    """
    Calculates the B(u, S) metric for a given vertex u and set S.

    B(u, S) = |{v in N(u) intersect unassigned : v not in N(S)}|

    Args:
        graph: The non-commuting graph.
        u: The candidate vertex to be added.
        unassigned: The set of vertices not yet assigned to any group.
        neighbors_of_S: The pre-calculated set of neighbors for the current group S.

    Returns:
        The calculated B(u, S) value.
    """
    neighbors_of_u = set(graph.neighbors(u))
    potential_v = neighbors_of_u.intersection(unassigned)
    
    # Efficiently calculate the set difference
    b_set = potential_v - neighbors_of_S
    
    return len(b_set)


def create_commutation_table(graph: nx.Graph) -> np.ndarray:
    """
    Creates a commutation table from a non-commuting graph.

    The table will have entry (i, j) = 1 if nodes i and j commute (no edge),
    and 0 otherwise (edge exists).

    Args:
        graph: The non-commuting graph.

    Returns:
        A numpy array representing the commutation table.
    """
    adjacency_matrix = nx.to_numpy_array(graph, dtype=np.int8)
    commutation_table = 1 - adjacency_matrix
    return commutation_table


def create_colour_map(graph: nx.Graph, partitions: List[Set[int]]) -> List[str]:
    """
    Creates a colour map for drawing the graph with networkx.

    Args:
        graph: The graph to be coloured.
        partitions: A list of sets, where each set is a group of nodes.

    Returns:
        A list of colour strings, ordered according to graph.nodes(),
        suitable for use with nx.draw's node_color argument.
    """
    colours = [
        "red", "blue", "green", "yellow", "purple", "orange", "pink", 
        "brown", "cyan", "magenta", "lime", "teal", "indigo", "violet",
        "gold", "coral", "turquoise", "plum", "khaki", "salmon"
    ]

    node_to_colour = {}
    for i, partition in enumerate(partitions):
        colour = colours[i % len(colours)]
        for node in partition:
            node_to_colour[node] = colour

    colour_map_list = [node_to_colour.get(node, 'gray') for node in graph.nodes()]

    return colour_map_list

def _bron_kerbosch_pivot(graph, R, P, X, cliques):
    """
    Recursive helper for the Bron-Kerbosch algorithm with pivoting.
    
    Args:
        graph: The graph to find cliques in.
        R: The current clique being built.
        P: Candidate nodes that can be added to R.
        X: Nodes already processed, used to avoid duplicate cliques.
        cliques: A list to store all found maximal cliques.
    """
    if not P and not X:
        # When P and X are empty, R is a maximal clique.
        cliques.append(R)
        return

    if not P:
        return

    # Choose a pivot node u from P union X to reduce recursive calls.
    try:
        pivot = next(iter(P | X))
        P_without_neighbors_of_pivot = P - set(graph.neighbors(pivot))
    except (StopIteration, nx.NetworkXError):
        # Handle cases where P | X is empty or pivot has no neighbors
        P_without_neighbors_of_pivot = P

    for v in list(P_without_neighbors_of_pivot):
        neighbors_of_v = set(graph.neighbors(v))
        _bron_kerbosch_pivot(graph, R | {v}, P & neighbors_of_v, X & neighbors_of_v, cliques)
        P.remove(v)
        X.add(v)

def partition_with_bk(graph: nx.Graph) -> List[Set[int]]:
    """
    Partitions a graph into disjoint maximal cliques using the Bron-Kerbosch algorithm.

    Args:
        graph: The graph to partition (e.g., a commuting graph).

    Returns:
        A list of disjoint sets, where each set is a maximal clique.
    """
    graph_copy = graph.copy()
    partitions = []
    
    while graph_copy.number_of_nodes() > 0:
        cliques = []
        _bron_kerbosch_pivot(
            graph_copy,
            set(),
            set(graph_copy.nodes()),
            set(),
            cliques
        )
        
        if not cliques:
            for node in list(graph_copy.nodes()):
                partitions.append({node})
            break

        # Find the largest maximal clique
        largest_clique = max(cliques, key=len)
        partitions.append(largest_clique)
        
        # Remove the nodes of the found clique from the graph
        graph_copy.remove_nodes_from(largest_clique)
        
    return partitions

def _generate_random_pauli_string(N: int, k: int) -> np.ndarray:
    """
    Generates a random k-local Pauli string on N qubits.
    Internal Representation: I=0, X=1, Y=2, Z=3.
    """
    if not 0 <= k <= N:
        raise ValueError("k (locality) must be between 0 and N, inclusive.")
    
    # Start with an identity string (all zeros)
    pauli_array = np.zeros(N, dtype=int)
    
    # Choose k unique locations for non-identity Paulis
    qubit_indices = random.sample(list(range(N)), k)
    
    # Assign a random Pauli (X, Y, or Z) to the chosen locations
    pauli_operators = [0, 1, 2, 3] # Corresponding to I, X, Y, Z
    for idx in qubit_indices:
        pauli_array[idx] = random.choice(pauli_operators)
        
    return pauli_array

def _pauli_array_to_string(pauli_array: np.ndarray) -> str:
    """Converts the numerical Pauli array to its string representation."""
    mapping = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}
    return ''.join([mapping[p] for p in pauli_array])

def _do_pauli_strings_commute(p1: np.ndarray, p2: np.ndarray) -> bool:
    """
    Checks if two Pauli strings commute.
    
    They commute if the number of positions where their single-qubit
    operators anticommute (e.g., X and Y) is even.
    """
    mismatches = 0
    for op1, op2 in zip(p1, p2):
        # Operators anticommute if they are different and not Identity
        if op1 != 0 and op2 != 0 and op1 != op2:
            mismatches += 1
            
    return mismatches % 2 == 0

def generate_hamiltonian_graph(N: int, k: int, num_terms: int | None = None) -> Tuple[nx.Graph, Hamiltonian]:
    """
    Generates a graph representing the commutation relations of a random Hamiltonian.

    Nodes are k-local Pauli strings, and an edge exists between two nodes
    if the corresponding Pauli strings commute.

    Args:
        N: The number of qubits.
        k: The locality of the Pauli terms (the number of non-identity operators).
        num_terms: The number of Pauli strings to generate for the Hamiltonian.
                   Defaults to 1.5*N if not provided.

    Returns:
        A tuple of (NetworkX Graph, Hamiltonian object) representing the commutation relations.
    """
    if num_terms is None:
        num_terms = int(1.5 * N)

    # Generate unique Pauli strings
    pauli_strings = []
    pauli_strings_set = set()
    
    while len(pauli_strings) < num_terms:
        p_string = PauliString.random(N, k)
        p_str_repr = str(p_string)
        
        if p_str_repr not in pauli_strings_set:
            pauli_strings_set.add(p_str_repr)
            pauli_strings.append(p_string)
    
    # Create Hamiltonian with random coefficients
    hamiltonian = Hamiltonian()
    for p_string in pauli_strings:
        # Generate non-zero coefficient
        coeff = 0.0
        while abs(coeff) < 1e-14:
            coeff = random.gauss(0, 1)
        hamiltonian.add_term(p_string, coeff)

    # Create the graph and add all Pauli strings as nodes
    G = nx.Graph()
    G.add_nodes_from([str(p) for p in pauli_strings])
    
    # Iterate through all pairs of nodes to check for commutation
    for i in range(len(pauli_strings)):
        for j in range(i + 1, len(pauli_strings)):
            p1 = pauli_strings[i]
            p2 = pauli_strings[j]
            
            # If they commute, add an edge to the graph
            if p1.commutes_with(p2):
                G.add_edge(str(p1), str(p2))
                
    return nx.complement(G), hamiltonian