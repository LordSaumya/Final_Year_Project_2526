"""
SymmetryScavenger: BFS-based symmetry orbit exploration for S3-REBROKE.

This module implements the core scavenging engine that uses symmetry generators
to find orbiting cliques cheaply after expensive heuristics discover unique seeds.
"""

from collections import deque
from typing import Set, Optional, List, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hamiltonian_utils import PauliString
from partitioning.symmetry_utils import (
    apply_permutation_to_clique,
    get_clique_fingerprint,
    is_identity_permutation,
)


class SymmetryScavenger:
    """
    Explores symmetry orbits of cliques using BFS to find related partitions.
    
    The scavenger uses permutation generators to discover "orbiting" cliques
    that are related by symmetry to seed cliques found by expensive heuristics.
    This allows finding multiple valid partitions with minimal computational cost.
    
    Attributes:
        generators: List of permutation mappings (tuples).
        tau: Strictness threshold (0.0 to 1.0) for accepting partial cliques.
        queue: BFS queue storing candidate cliques.
        visited_fingerprints: Set of fingerprints for already-processed clique orbits.
    """
    
    def __init__(
        self, 
        generators: List[Tuple[int, ...]], 
        tau: float = 0.9
    ):
        """
        Initialise the SymmetryScavenger.
        
        Args:
            generators: List of permutation maps. Each map is a tuple where
                        perm_map[i] gives the new position for qubit i.
            tau: Strictness threshold (default: 0.9). A candidate clique is
                 accepted only if at least tau fraction of its terms remain
                 unassigned. Higher values enforce stricter partition quality.
                 
        Raises:
            ValueError: If tau is not in the range [0.0, 1.0].
        """
        if not 0.0 <= tau <= 1.0:
            raise ValueError(f"tau must be in [0.0, 1.0], got {tau}")
        
        # Filter out identity permutations as they don't generate new cliques
        self.generators = [
            g for g in generators 
            if not is_identity_permutation(g)
        ]
        self.tau = tau
        self.queue: deque[Set[PauliString]] = deque()
        self.visited_fingerprints: Set[str] = set()
        
        # Statistics for analysis
        self._stats = {
            'seeds_pushed': 0,
            'candidates_generated': 0,
            'candidates_accepted': 0,
            'candidates_rejected': 0,
            'duplicates_avoided': 0,
        }
    
    def reset(self) -> None:
        """
        Reset the scavenger state for a new partitioning run.
        
        Clears the queue and visited fingerprints while keeping generators
        and threshold settings.
        """
        self.queue.clear()
        self.visited_fingerprints.clear()
        self._stats = {
            'seeds_pushed': 0,
            'candidates_generated': 0,
            'candidates_accepted': 0,
            'candidates_rejected': 0,
            'duplicates_avoided': 0,
        }
    
    def push_seed(self, clique: Set[PauliString]) -> None:
        """
        Entry point for cliques found by standard ERLF (expensive discovery).
        
        Registers the seed clique and enqueues its symmetric neighbours
        for later scavenging. This should be called whenever the main
        algorithm discovers a new clique using expensive heuristics.
        
        Args:
            clique: A set of PauliStrings forming a commuting group (clique).
        """
        if not clique:
            return
        
        self._stats['seeds_pushed'] += 1
        
        # Mark the seed itself as visited
        seed_fingerprint = get_clique_fingerprint(clique)
        self.visited_fingerprints.add(seed_fingerprint)
        
        # Enqueue all symmetric neighbours of this seed
        self._enqueue_neighbors(clique)
    
    def _enqueue_neighbors(self, clique: Set[PauliString]) -> None:
        """
        Enqueue symmetric transformations of a clique.
        
        Applies all generators to the clique and adds the resulting
        symmetric cliques to the queue if they haven't been visited.
        
        Args:
            clique: The clique to generate symmetric neighbours from.
        """
        for generator in self.generators:
            # Apply permutation to get symmetric clique
            sym_clique = apply_permutation_to_clique(clique, generator)
            
            # Calculate fingerprint for deduplication
            fingerprint = get_clique_fingerprint(sym_clique)
            
            # Only enqueue if we haven't seen this clique orbit before
            if fingerprint not in self.visited_fingerprints:
                self.visited_fingerprints.add(fingerprint)
                self.queue.append(sym_clique)
                self._stats['candidates_generated'] += 1
            else:
                self._stats['duplicates_avoided'] += 1
    
    def pop_candidate(
        self, 
        unassigned_set: Set[PauliString]
    ) -> Optional[Set[PauliString]]:
        """
        Pop and return the next valid candidate clique from the queue.
        
        Iterates through the queue, checking each candidate against the
        strictness threshold. A candidate is accepted if at least tau
        fraction of its terms remain in the unassigned set.
        
        When a valid candidate is found:
        1. The full candidate is used to enqueue further symmetric neighbours
           (even if some terms are already assigned).
        2. Only the remnant (intersection with unassigned) is returned.
        
        Args:
            unassigned_set: The set of PauliStrings not yet assigned to any partition.
            
        Returns:
            The remnant of a valid candidate (intersection with unassigned),
            or None if the queue is empty or no valid candidates remain.
        """
        while self.queue:
            candidate = self.queue.popleft()
            
            # Compute the remnant: terms still available for assignment
            remnant = candidate.intersection(unassigned_set)
            
            # Strict acceptance check: remnant must be at least tau fraction
            if len(candidate) > 0:
                acceptance_ratio = len(remnant) / len(candidate)
                
                if acceptance_ratio >= self.tau:
                    self._stats['candidates_accepted'] += 1
                    
                    # Recursive step: use the FULL candidate to find further orbits
                    # This ensures we explore the complete symmetry group even if
                    # parts of the candidate are already assigned
                    self._enqueue_neighbors(candidate)
                    
                    # Return only the remnant for actual partition assignment
                    return remnant
                else:
                    self._stats['candidates_rejected'] += 1
        
        # Queue is empty, no valid candidates found
        return None
    
    def has_candidates(self) -> bool:
        """Check if there are candidates waiting in the queue."""
        return len(self.queue) > 0
    
    def queue_size(self) -> int:
        """Return the current number of candidates in the queue."""
        return len(self.queue)
    
    def get_stats(self) -> dict:
        """
        Return statistics about the scavenger's operation.
        
        Returns:
            Dictionary containing:
            - seeds_pushed: Number of seed cliques from expensive discovery
            - candidates_generated: Number of symmetric candidates created
            - candidates_accepted: Number of candidates passing threshold
            - candidates_rejected: Number of candidates failing threshold
            - duplicates_avoided: Number of duplicate fingerprints caught
        """
        return self._stats.copy()
    
    def __repr__(self) -> str:
        return (
            f"SymmetryScavenger("
            f"generators={len(self.generators)}, "
            f"tau={self.tau}, "
            f"queue_size={len(self.queue)}, "
            f"visited={len(self.visited_fingerprints)})"
        )


if __name__ == "__main__":
    # Simple self-test
    print("Testing SymmetryScavenger...")
    
    # Create a simple test case
    # 4-qubit system with translation symmetry (0->1->2->3->0)
    translation_gen = (1, 2, 3, 0)
    scavenger = SymmetryScavenger([translation_gen], tau=0.8)
    
    # Create a seed clique
    seed = {
        PauliString.from_string("ZZII"),
        PauliString.from_string("IZZI"),
    }
    
    print(f"Initial: {scavenger}")
    
    # Push the seed
    scavenger.push_seed(seed)
    print(f"After push_seed: {scavenger}")
    
    # Create unassigned set (all terms)
    all_terms = set()
    for s in ["ZZII", "IZZI", "IIZZ", "ZIIZ"]:
        all_terms.add(PauliString.from_string(s))
    
    # Pop candidates
    print("\nPopping candidates:")
    unassigned = all_terms.copy()
    while scavenger.has_candidates():
        candidate = scavenger.pop_candidate(unassigned)
        if candidate:
            print(f"  Got candidate: {[str(p) for p in candidate]}")
            unassigned -= candidate
        else:
            print("  No valid candidate")
            break
    
    print(f"\nFinal stats: {scavenger.get_stats()}")
    print("Test complete!")
