from enum import IntEnum
import numpy as np
from typing import Dict, List, Tuple, Optional
import random


class PauliOperator(IntEnum):
    """Enumeration for Pauli operators."""
    I = 0
    X = 1
    Y = 2
    Z = 3


class PauliString:
    """
    Represents a Pauli string of length n_qubits.
    
    Stores the string as a numpy array with elements from {I, X, Y, Z}.
    """
    
    def __init__(self, operators: np.ndarray):
        """
        Initialise a Pauli string.
        
        Args:
            operators: Array of integers representing Pauli operators (0=I, 1=X, 2=Y, 3=Z)
        """
        self.operators = np.array(operators, dtype=np.int8)
        self.n_qubits = len(self.operators)
        self._hash = None
    
    @classmethod
    def from_list(cls, operators: List[int]) -> 'PauliString':
        """Create a PauliString from a list of integers."""
        return cls(np.array(operators, dtype=np.int8))
    
    @classmethod
    def from_string(cls, pauli_str: str) -> 'PauliString':
        """Create a PauliString from a string like 'IXYZ'."""
        mapping = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
        operators = [mapping[char] for char in pauli_str]
        return cls.from_list(operators)
    
    @classmethod
    def identity(cls, n_qubits: int) -> 'PauliString':
        """Create an identity Pauli string."""
        return cls(np.zeros(n_qubits, dtype=np.int8))
    
    @classmethod
    def random(cls, n_qubits: int, locality: int) -> 'PauliString':
        """
        Generate a random k-local Pauli string.
        
        Args:
            n_qubits: Number of qubits
            locality: Number of non-identity operators
        """
        if not 0 <= locality <= n_qubits:
            raise ValueError("Locality must be between 0 and n_qubits")
        
        operators = np.zeros(n_qubits, dtype=np.int8)
        indices = random.sample(range(n_qubits), locality)
        
        for idx in indices:
            operators[idx] = random.choice([1, 2, 3])  # X, Y, or Z
        
        return cls(operators)
    
    def __str__(self) -> str:
        """Convert to string representation."""
        mapping = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}
        return ''.join(mapping[op] for op in self.operators)
    
    def __repr__(self) -> str:
        return f"PauliString('{str(self)}')"
    
    def __hash__(self) -> int:
        """Hash based on the tuple of operators."""
        if self._hash is None:
            self._hash = hash(tuple(self.operators))
        return self._hash
    
    def __eq__(self, other) -> bool:
        """Check equality with another PauliString."""
        if not isinstance(other, PauliString):
            return False
        return np.array_equal(self.operators, other.operators)
    
    def __getitem__(self, index: int) -> int:
        """Get operator at a specific qubit index."""
        return int(self.operators[index])
    
    def commutes_with(self, other: 'PauliString') -> bool:
        """
        Check if this Pauli string commutes with another.
        
        They commute if the number of positions where their single-qubit
        operators anticommute is even.
        """
        if self.n_qubits != other.n_qubits:
            raise ValueError("Pauli strings must have same length")
        
        mismatches = 0
        for op1, op2 in zip(self.operators, other.operators):
            # Operators anticommute if they are different and not Identity
            if op1 != 0 and op2 != 0 and op1 != op2:
                mismatches += 1
        
        return mismatches % 2 == 0
    
    def apply_permutation(self, perm_map: Tuple[int, ...]) -> 'PauliString':
        """
        Apply a permutation to this Pauli string.
        
        Args:
            perm_map: Tuple where perm_map[i] is the new position for qubit i
        """
        new_operators = np.zeros(self.n_qubits, dtype=np.int8)
        for i in range(self.n_qubits):
            j = perm_map[i]
            new_operators[j] = self.operators[i]
        return PauliString(new_operators)
    
    def to_tuple(self) -> Tuple[int, ...]:
        """Convert to tuple representation."""
        return tuple(self.operators)


class Hamiltonian:
    """
    Represents a Hamiltonian as a dictionary of Pauli strings and their coefficients.
    """
    
    def __init__(self, terms: Optional[Dict[PauliString, complex]] = None):
        """
        Initialise a Hamiltonian.
        
        Args:
            terms: Dictionary mapping PauliString to complex coefficients
        """
        self.terms: Dict[PauliString, complex] = terms if terms is not None else {}
        self.n_qubits = None
        if self.terms:
            self.n_qubits = next(iter(self.terms)).n_qubits
    
    @classmethod
    def from_dict(cls, terms_dict: Dict[Tuple[int, ...], complex]) -> 'Hamiltonian':
        """
        Create a Hamiltonian from a dictionary with tuple keys.
        
        Args:
            terms_dict: Dictionary mapping tuples to coefficients
        """
        terms = {PauliString.from_list(list(key)): coeff 
                for key, coeff in terms_dict.items()}
        return cls(terms)
    
    def add_term(self, pauli_string: PauliString, coefficient: complex) -> None:
        """Add or update a term in the Hamiltonian."""
        if self.n_qubits is None:
            self.n_qubits = pauli_string.n_qubits
        elif self.n_qubits != pauli_string.n_qubits:
            raise ValueError("All terms must have same number of qubits")
        
        if pauli_string in self.terms:
            self.terms[pauli_string] += coefficient
        else:
            self.terms[pauli_string] = coefficient
    
    def __len__(self) -> int:
        """Return the number of terms."""
        return len(self.terms)
    
    def __iter__(self):
        """Iterate over (PauliString, coefficient) pairs."""
        return iter(self.terms.items())
    
    def __getitem__(self, pauli_string: PauliString) -> complex:
        """Get coefficient for a given Pauli string."""
        return self.terms[pauli_string]
    
    def __contains__(self, pauli_string: PauliString) -> bool:
        """Check if a Pauli string is in the Hamiltonian."""
        return pauli_string in self.terms
    
    def __str__(self) -> str:
        """Convert to human-readable string."""
        terms = []
        for pauli, coeff in self.terms.items():
            if isinstance(coeff, complex):
                if coeff.imag == 0:
                    terms.append(f"{coeff.real:.3f} * {pauli}")
                else:
                    terms.append(f"({coeff.real:.3f}+{coeff.imag:.3f}j) * {pauli}")
            else:
                terms.append(f"{coeff:.3f} * {pauli}")
        return " + ".join(terms)
    
    def apply_permutation(self, perm_map: Tuple[int, ...]) -> 'Hamiltonian':
        """
        Apply a permutation to all terms in the Hamiltonian.
        
        Args:
            perm_map: Tuple where perm_map[i] is the new position for qubit i
        """
        new_terms = {}
        for pauli, coeff in self.terms.items():
            new_pauli = pauli.apply_permutation(perm_map)
            new_terms[new_pauli] = coeff
        return Hamiltonian(new_terms)
    
    def to_dict(self) -> Dict[Tuple[int, ...], complex]:
        """Convert to dictionary with tuple keys."""
        return {pauli.to_tuple(): coeff for pauli, coeff in self.terms.items()}