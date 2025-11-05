import numpy as np
from typing import List
import sys
import os

from z2_sym_utils import PAULI_TO_BINARY_MAP, BINARY_TO_PAULI_MAP, gaussian_elimination_gf2, find_kernel_gf2 # ty: ignore[unresolved-import]

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hamiltonian_utils import PauliString, Hamiltonian, PauliOperator

def find_z2_symmetries(hamiltonian: Hamiltonian) -> List[PauliString]:
    """
    Identifies the Z2 Pauli symmetries of a given Hamiltonian.
    
    This function implements the algorithm from Section 4.1 of the
    provided paper [cite: 280-289].
    
    Args:
        hamiltonian: A Hamiltonian object.
    
    Returns:
        A list of PauliString objects, each representing a generator
        of the Z2 symmetry group.
    """
    
    if hamiltonian.n_qubits is None:
        return []
        
    n_qubits = hamiltonian.n_qubits
    assert n_qubits is not None
    
    # The algorithm finds symmetries of the support of H.
    pauli_terms = [
        p for p, c in hamiltonian.terms.items() 
        if not np.isclose(c, 0.0)
    ]
    
    if not pauli_terms:
        # No non-zero terms, any Pauli string is a "symmetry"
        # but returning an empty list is more pragmatic.
        return []
        
    m_terms = len(pauli_terms)

    # Construct the G_x and G_z matrices
    # G_x and G_z will be (n_qubits, m_terms)
    G_x = np.zeros((n_qubits, m_terms), dtype=np.int8)
    G_z = np.zeros((n_qubits, m_terms), dtype=np.int8)

    for j, pauli_string in enumerate(pauli_terms):
        for i in range(n_qubits):
            op = pauli_string[i]
            # Use the map from our utils file
            ax, az = PAULI_TO_BINARY_MAP[op]
            G_x[i, j] = ax
            G_z[i, j] = az
            
    # "The Pauli strings in a Hamiltonian can be represented as G(H) = [Gx/Gz]" [cite: 287]
    # "From this matrix, we can construct the check matrix E = [G_z^T | G_x^T]" 
    E = np.hstack([G_z.T, G_x.T])
    
    # "The kernel ker(E) thus gives us the generators..." 
    # Use the kernel-finding function from our utils file
    kernel_basis_vectors = find_kernel_gf2(E)
    
    # Convert kernel basis vectors back to PauliStrings
    symmetries: List[PauliString] = []
    for v in kernel_basis_vectors:
        # Each vector v is a 2n-length binary vector (ax | az)
        ax_vec = v[:n_qubits]
        az_vec = v[n_qubits:]
        
        pauli_ops = []
        for i in range(n_qubits):
            # Use the reverse map from our utils file
            op = BINARY_TO_PAULI_MAP[(ax_vec[i], az_vec[i])]
            pauli_ops.append(op)
            
        symmetry_string = PauliString.from_list(pauli_ops)
        symmetries.append(symmetry_string)
        
    return symmetries

if __name__ == "__main__":
    n_qubits = 4
    pauli_terms = {
        PauliString.from_list([PauliOperator.X, PauliOperator.Y]): 1.0,
        PauliString.from_list([PauliOperator.Y, PauliOperator.Z]): 1.0,
        PauliString.from_list([PauliOperator.X, PauliOperator.X]): 1.0,
    }
    hamiltonian = Hamiltonian(pauli_terms)
    
    symmetries = find_z2_symmetries(hamiltonian)
    print("Found Z2 Symmetries:")
    for sym in symmetries:
        print(sym)