import sys
import os
from typing import Dict, Tuple, List, Set
from functools import partial
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hamiltonian_utils import Hamiltonian, PauliString

def check_trs_symmetry(hamiltonian: Hamiltonian) -> bool:
    """
    Check if the Hamiltonian has time-reversal symmetry (TRS).
    
    A Hamiltonian has TRS if:
    - Even weight strings have purely real coefficients.
    - Odd weight strings have purely imaginary coefficients.
    - If an even-weight Pauli string has a non-real coefficient, or if an odd-weight Pauli string does not 
    have a purely imaginary coefficient, then H must contain both the original term and its time-reversed
    counterpart (i.e. mapped by complex conjugation and possible sign-flip).

    Note that the third case is impossible for this implementation since Pauli strings are represented using unique keys.
    If a term exists, it may only have one coefficient.
    
    Args:
        hamiltonian: A Hamiltonian object.
    """

    if not hamiltonian.n_qubits:
        return True  # Empty Hamiltonian trivially has TRS
    
    for pauli_string, coeff in hamiltonian.terms.items():
        weight: int = pauli_string.weight()
        is_even_weight: bool = (weight % 2 == 0)
        
        if is_even_weight:
            # Even weight: coefficient must be real
            if not np.isclose(coeff.imag, 0.0):
                # If not real, check for time-reversed counterpart
                conj_coeff = np.conj(coeff)
                if pauli_string.to_tuple() not in hamiltonian.terms or not np.isclose(hamiltonian.terms[pauli_string.to_tuple()], conj_coeff):
                    return False
        else:
            # Odd weight: coefficient must be purely imaginary
            if not np.isclose(coeff.real, 0.0):
                # If not purely imaginary, check for time-reversed counterpart
                conj_coeff = np.conj(coeff)
                if pauli_string.to_tuple() not in hamiltonian.terms or not np.isclose(hamiltonian.terms[pauli_string.to_tuple()], conj_coeff):
                    return False
    
    return True

if __name__ == "__main__":

    # All real coefficients, All even weight
    terms = {
        PauliString.from_list([1, 1, 0]): 5.0,
        PauliString.from_list([2, 2, 0]): 2.0,
        PauliString.from_list([3, 3, 0]): -4.0,
    }

    hamiltonian = Hamiltonian(terms)
    has_trs = check_trs_symmetry(hamiltonian)
    print(f"Hamiltonian 1 has TRS (true expected): {has_trs}")

    # All imaginary coefficients, All odd weight
    terms = {
        PauliString.from_list([1, 0, 0]): 1.0j,
        PauliString.from_list([0, 2, 0]): 3.0j,
        PauliString.from_list([0, 0, 3]): -2.5j,
    }
    hamiltonian = Hamiltonian(terms)
    has_trs = check_trs_symmetry(hamiltonian)
    print(f"Hamiltonian 2 has TRS (true expected): {has_trs}")

    # Imaginary coefficient with even weight
    terms = {
        PauliString.from_list([1, 1, 0]): 5.0,
        PauliString.from_list([2, 2, 0]): 2.0 + 1.0j,
        PauliString.from_list([3, 3, 0]): -4.0,
    }

    hamiltonian = Hamiltonian(terms)
    has_trs = check_trs_symmetry(hamiltonian)
    print(f"Hamiltonian 3 has TRS (false expected): {has_trs}")

    # Real coefficient with odd weight
    terms = {
        PauliString.from_list([1, 0, 0]): 1.0 + 1.0j,
        PauliString.from_list([0, 2, 0]): 3.0j,
        PauliString.from_list([0, 0, 3]): -2.5j,
    }

    hamiltonian = Hamiltonian(terms)
    has_trs = check_trs_symmetry(hamiltonian)
    print(f"Hamiltonian 4 has TRS (false expected): {has_trs}")

