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
        indices = random.sample(list(range(n_qubits)), locality)
        
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
    
    def weight(self) -> int:
        """Return the weight (number of non-identity operators)."""
        return int(np.count_nonzero(self.operators))


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
        self.terms: Dict[PauliString, complex] = {}
        self.n_qubits: int | None = None
        
        if terms is not None:
            for pauli_string, coeff in terms.items():
                self.add_term(pauli_string, coeff)
    
    @classmethod
    def from_dict(cls, terms_dict: Dict[Tuple[int, ...], complex]) -> 'Hamiltonian':
        """
        Create a Hamiltonian from a dictionary with tuple keys.
        
        Args:
            terms_dict: Dictionary mapping tuples to coefficients
        """
        hamiltonian = cls()
        for key, coeff in terms_dict.items():
            pauli_string = PauliString.from_list(list(key))
            hamiltonian.add_term(pauli_string, coeff)
        return hamiltonian
    
    def add_term(self, pauli_string: PauliString, coefficient: complex) -> None:
        """
        Add or update a term in the Hamiltonian.
        
        Args:
            pauli_string: The Pauli string for this term
            coefficient: The coefficient (must be non-zero)
            
        Raises:
            ValueError: If coefficient is zero or close to zero
        """
        if self.n_qubits is None:
            self.n_qubits = pauli_string.n_qubits
        elif self.n_qubits != pauli_string.n_qubits:
            raise ValueError("All terms must have same number of qubits")
        
        # Check for zero coefficient
        if isinstance(coefficient, complex):
            if abs(coefficient) < 1e-14:
                raise ValueError(f"Coefficient must be non-zero, got {coefficient}")
        else:
            if abs(coefficient) < 1e-14:
                raise ValueError(f"Coefficient must be non-zero, got {coefficient}")
        
        if pauli_string in self.terms:
            new_coeff = self.terms[pauli_string] + coefficient
            # Check if adding results in zero
            if abs(new_coeff) < 1e-14:
                raise ValueError(
                    f"Adding coefficient {coefficient} to existing term results in zero"
                )
            self.terms[pauli_string] = new_coeff
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
    
    @staticmethod
    def tfi_1d(n_qubits: int, h: float = 1.0, J: float = 1.0, periodic: bool = False) -> 'Hamiltonian':
        """
        Generate the 1D Transverse Field Ising Model Hamiltonian with open boundary conditions.
        
        H = -J sum Z_i Z_{i+1} - h sum X_i
        
        Args:
            n_qubits: Number of qubits
            h: Transverse field strength
            J: Interaction strength
        """
        terms = {}
        if not periodic:
            # Interaction terms -J Z_i Z_{i+1}
            for i in range(n_qubits - 1):
                pauli_ops = [0] * n_qubits
                pauli_ops[i] = 3  # Z
                pauli_ops[i + 1] = 3  # Z
                pauli_string = PauliString.from_list(pauli_ops)
                terms[pauli_string] = -J
            
            # Transverse field terms -h X_i
            for i in range(n_qubits):
                pauli_ops = [0] * n_qubits
                pauli_ops[i] = 1  # X
                pauli_string = PauliString.from_list(pauli_ops)
                terms[pauli_string] = -h
        else:
            # Interaction terms -J Z_i Z_{i+1} with periodic boundary
            for i in range(n_qubits):
                pauli_ops = [0] * n_qubits
                pauli_ops[i] = 3  # Z
                pauli_ops[(i + 1) % n_qubits] = 3  # Z
                pauli_string = PauliString.from_list(pauli_ops)
                terms[pauli_string] = -J
            
            # Transverse field terms -h X_i
            for i in range(n_qubits):
                pauli_ops = [0] * n_qubits
                pauli_ops[i] = 1  # X
                pauli_string = PauliString.from_list(pauli_ops)
                terms[pauli_string] = -h
        
        return Hamiltonian(terms)
    
    @staticmethod
    def heisenberg_1d(n_qubits: int, Jx: float = 1.0, Jy: float = 1.0, Jz: float = 1.0, periodic = False, fc = False) -> 'Hamiltonian':
        """
        Generate the 1D Heisenberg Model Hamiltonian with open boundary conditions.
        
        H = sum (Jx X_i X_{i+1} + Jy Y_i Y_{i+1} + Jz Z_i Z_{i+1})
        
        Args:
            n_qubits: Number of qubits
            Jx: Interaction strength for X terms
            Jy: Interaction strength for Y terms
            Jz: Interaction strength for Z terms
        """
        terms = {}
        if not fc:
            if not periodic:
                for i in range(n_qubits - 1):
                    if Jx != 0.0:
                        # X_i X_{i+1}
                        pauli_ops_x = [0] * n_qubits
                        pauli_ops_x[i] = 1  # X
                        pauli_ops_x[i + 1] = 1  # X
                        pauli_string_x = PauliString.from_list(pauli_ops_x)
                        terms[pauli_string_x] = Jx
                    
                    if Jy != 0.0:
                        # Y_i Y_{i+1}
                        pauli_ops_y = [0] * n_qubits
                        pauli_ops_y[i] = 2  # Y
                        pauli_ops_y[i + 1] = 2  # Y
                        pauli_string_y = PauliString.from_list(pauli_ops_y)
                        terms[pauli_string_y] = Jy
                    
                    if Jz != 0.0:
                        # Z_i Z_{i+1}
                        pauli_ops_z = [0] * n_qubits
                        pauli_ops_z[i] = 3  # Z
                        pauli_ops_z[i + 1] = 3  # Z
                        pauli_string_z = PauliString.from_list(pauli_ops_z)
                        terms[pauli_string_z] = Jz
                
                return Hamiltonian(terms)
            else:
                for i in range(n_qubits):
                    if Jx != 0.0:
                        # X_i X_{i+1}
                        pauli_ops_x = [0] * n_qubits
                        pauli_ops_x[i] = 1  # X
                        pauli_ops_x[(i + 1) % n_qubits] = 1  # X
                        pauli_string_x = PauliString.from_list(pauli_ops_x)
                        terms[pauli_string_x] = Jx
                    
                    if Jy != 0.0:
                        # Y_i Y_{i+1}
                        pauli_ops_y = [0] * n_qubits
                        pauli_ops_y[i] = 2  # Y
                        pauli_ops_y[(i + 1) % n_qubits] = 2  # Y
                        pauli_string_y = PauliString.from_list(pauli_ops_y)
                        terms[pauli_string_y] = Jy
                    
                    if Jz != 0.0:
                        # Z_i Z_{i+1}
                        pauli_ops_z = [0] * n_qubits
                        pauli_ops_z[i] = 3  # Z
                        pauli_ops_z[(i + 1) % n_qubits] = 3  # Z
                        pauli_string_z = PauliString.from_list(pauli_ops_z)
                        terms[pauli_string_z] = Jz
                
                return Hamiltonian(terms)
        else:
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    if Jx != 0.0:
                        # X_i X_j
                        pauli_ops_x = [0] * n_qubits
                        pauli_ops_x[i] = 1  # X
                        pauli_ops_x[j] = 1  # X
                        pauli_string_x = PauliString.from_list(pauli_ops_x)
                        terms[pauli_string_x] = Jx
                    
                    if Jy != 0.0:
                        # Y_i Y_j
                        pauli_ops_y = [0] * n_qubits
                        pauli_ops_y[i] = 2  # Y
                        pauli_ops_y[j] = 2  # Y
                        pauli_string_y = PauliString.from_list(pauli_ops_y)
                        terms[pauli_string_y] = Jy
                    
                    if Jz != 0.0:
                        # Z_i Z_j
                        pauli_ops_z = [0] * n_qubits
                        pauli_ops_z[i] = 3  # Z
                        pauli_ops_z[j] = 3  # Z
                        pauli_string_z = PauliString.from_list(pauli_ops_z)
                        terms[pauli_string_z] = Jz

            return Hamiltonian(terms)