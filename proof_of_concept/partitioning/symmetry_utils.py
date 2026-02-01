"""
Symmetry utilities for S3-REBROKE algorithm.

This module provides functions for permutation application and canonical fingerprinting
of Pauli string cliques for use in the symmetry scavenging algorithm.

This module re-exports and wraps utilities from the permutation module for convenience.
"""

import hashlib
from typing import Set, Tuple, List
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hamiltonian_utils import PauliString

# Import permutation utilities from the existing permutation module
from permutation.permutation_utils import (
    _cycles_to_map as cycles_to_map_internal,
    _compose_maps as compose_maps_internal,
    _apply_permutation_int,
    _get_permutation_group,
)


def apply_permutation_to_pauli(
    pauli_string: PauliString, 
    perm_map: Tuple[int, ...]
) -> PauliString:
    """
    Apply a permutation to a Pauli string's qubit indices.
    
    The operator at old position 'i' moves to new position 'j = perm_map[i]'.
    Operates directly on the internal numpy array for O(n) performance.
    
    Args:
        pauli_string: The PauliString object to permute.
        perm_map: A tuple where perm_map[i] gives the new position for qubit i.
        
    Returns:
        A new PauliString with permuted qubit indices.
        
    Example:
        For perm_map = (1, 0, 2), qubit 0 moves to position 1, 
        qubit 1 moves to position 0, and qubit 2 stays at position 2.
    """
    return pauli_string.apply_permutation(perm_map)


def apply_permutation_to_clique(
    clique: Set[PauliString], 
    perm_map: Tuple[int, ...]
) -> Set[PauliString]:
    """
    Apply a permutation to all Pauli strings in a clique.
    
    Args:
        clique: A set of PauliString objects.
        perm_map: A tuple where perm_map[i] gives the new position for qubit i.
        
    Returns:
        A new set containing the permuted PauliStrings.
    """
    return {apply_permutation_to_pauli(p, perm_map) for p in clique}


def get_clique_fingerprint(clique: Set[PauliString]) -> str:
    """
    Compute a unique canonical fingerprint for a set of Pauli strings.
    
    This is used to prevent infinite loops in cyclic symmetry groups by 
    identifying when we've already processed a particular clique orbit.
    
    The fingerprint is computed by:
    1. Converting each PauliString to its string representation.
    2. Sorting the representations lexicographically.
    3. Joining and hashing the sorted tuple.
    
    Args:
        clique: A set of PauliString objects.
        
    Returns:
        A hexadecimal hash string uniquely identifying this clique.
        
    Note:
        Two cliques with the same set of Pauli strings (regardless of order)
        will produce the same fingerprint.
    """
    # Convert each Pauli string to its canonical string representation
    sorted_strings = sorted(str(p) for p in clique)
    
    # Create a deterministic hash from the sorted strings
    combined = '|'.join(sorted_strings)
    fingerprint = hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    return fingerprint


def get_identity_permutation(n_qubits: int) -> Tuple[int, ...]:
    """
    Return the identity permutation for n qubits.
    
    Args:
        n_qubits: Number of qubits.
        
    Returns:
        A tuple (0, 1, 2, ..., n_qubits-1).
    """
    return tuple(range(n_qubits))


def is_identity_permutation(perm_map: Tuple[int, ...]) -> bool:
    """
    Check if a permutation is the identity.
    
    Args:
        perm_map: The permutation map to check.
        
    Returns:
        True if perm_map is the identity, False otherwise.
    """
    return perm_map == tuple(range(len(perm_map)))


def compose_permutations(
    perm1: Tuple[int, ...], 
    perm2: Tuple[int, ...]
) -> Tuple[int, ...]:
    """
    Compose two permutations: result[i] = perm1[perm2[i]].
    
    This represents applying perm2 first, then perm1.
    
    Args:
        perm1: First permutation (applied second).
        perm2: Second permutation (applied first).
        
    Returns:
        The composed permutation.
        
    Raises:
        ValueError: If permutations have different lengths.
    """
    if len(perm1) != len(perm2):
        raise ValueError("Permutations must have the same length")
    
    return compose_maps_internal(perm1, perm2)


def invert_permutation(perm_map: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Compute the inverse of a permutation.
    
    Args:
        perm_map: The permutation to invert.
        
    Returns:
        The inverse permutation such that compose(perm, inverse) = identity.
    """
    n = len(perm_map)
    inverse = [0] * n
    for i, j in enumerate(perm_map):
        inverse[j] = i
    return tuple(inverse)


def cycles_to_permutation(cycles: List[List[int]], n_qubits: int) -> Tuple[int, ...]:
    """
    Convert cycle notation to a permutation map.
    
    Wraps the internal permutation utility for public use.
    
    Args:
        cycles: List of disjoint cycles, e.g., [[0, 1], [2, 3, 4]].
        n_qubits: Total number of qubits.
        
    Returns:
        A permutation tuple where perm[i] is the image of i.
        
    Example:
        cycles_to_permutation([[0, 1, 2]], 4) returns (1, 2, 0, 3)
        meaning: 0->1, 1->2, 2->0, 3->3
    """
    return cycles_to_map_internal(cycles, n_qubits)


def get_permutation_group(
    generator_cycles_list: List[List[List[int]]], 
    n_qubits: int
) -> Set[Tuple[int, ...]]:
    """
    Generate the full permutation group from a set of generator permutations.
    
    Uses BFS to compute the closure of the generators under composition.
    
    Args:
        generator_cycles_list: List of generators, where each generator is
                               in cycle format (e.g., [[0, 1], [2, 3]]).
        n_qubits: Total number of qubits.
        
    Returns:
        Set of all permutation maps in the generated group.
    """
    return _get_permutation_group(generator_cycles_list, n_qubits)


if __name__ == "__main__":
    # Simple self-test
    print("Testing symmetry utilities...")
    
    # Test permutation application
    p = PauliString.from_string("XYZII")
    perm = (1, 0, 2, 4, 3)  # Swap 0<->1 and 3<->4
    p_permuted = apply_permutation_to_pauli(p, perm)
    print(f"Original: {p}, Permuted: {p_permuted}")
    
    # Test fingerprinting
    clique = {
        PauliString.from_string("XYZII"),
        PauliString.from_string("IXYZI"),
        PauliString.from_string("IIXYZ"),
    }
    fp = get_clique_fingerprint(clique)
    print(f"Clique fingerprint: {fp}")
    
    # Verify same clique gives same fingerprint
    clique2 = {
        PauliString.from_string("IIXYZ"),
        PauliString.from_string("XYZII"),
        PauliString.from_string("IXYZI"),
    }
    fp2 = get_clique_fingerprint(clique2)
    assert fp == fp2, "Fingerprints should match for same clique"
    print("All tests passed!")
