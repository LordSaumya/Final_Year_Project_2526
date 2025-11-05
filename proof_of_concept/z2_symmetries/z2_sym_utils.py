import numpy as np
from typing import List, Tuple, Dict
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hamiltonian_utils import PauliString, Hamiltonian, PauliOperator

# Map PauliOperator enum to its (ax, az) binary representation
PAULI_TO_BINARY_MAP: Dict[int, Tuple[int, int]] = {
    PauliOperator.I: (0, 0),
    PauliOperator.X: (1, 0),
    PauliOperator.Y: (1, 1),
    PauliOperator.Z: (0, 1),
}

# Map (ax, az) binary representation back to PauliOperator enum
BINARY_TO_PAULI_MAP: Dict[Tuple[int, int], int] = {
    (0, 0): PauliOperator.I,
    (1, 0): PauliOperator.X,
    (1, 1): PauliOperator.Y,
    (0, 1): PauliOperator.Z,
}

# --- GF(2) Linear Algebra ---
def gaussian_elimination_gf2(matrix: np.ndarray) -> np.ndarray:
    """
    Performs Gaussian elimination on a matrix over the finite field GF(2).
    
    Args:
        matrix: A 2D numpy array with entries 0 or 1.
    
    Returns:
        The row-reduced echelon form (RREF) of the matrix.
    """
    mat = matrix.copy()
    n_rows, n_cols = mat.shape
    pivot_row = 0
    
    for j in range(n_cols):
        if pivot_row >= n_rows:
            break
        
        # Find a pivot (a '1') in the current column
        pivot = np.where(mat[pivot_row:, j] == 1)[0]
        
        if pivot.size > 0:
            # Pivot found at relative index pivot[0]
            pivot_idx = pivot_row + pivot[0]
            
            # Swap rows to bring the pivot to the current pivot_row
            mat[[pivot_row, pivot_idx], :] = mat[[pivot_idx, pivot_row], :]
            
            # Eliminate other '1's in this column
            for i in range(n_rows):
                if i != pivot_row and mat[i, j] == 1:
                    # XOR the pivot row with the current row
                    mat[i, :] = (mat[i, :] + mat[pivot_row, :]) % 2
            
            pivot_row += 1
            
    return mat


def find_kernel_gf2(matrix: np.ndarray) -> List[np.ndarray]:
    """
    Finds a basis for the kernel (null space) of a matrix over GF(2).
    
    Args:
        matrix: A 2D numpy array (m x n) with entries 0 or 1.
    
    Returns:
        A list of 1D numpy arrays, where each array is a basis vector
        for the kernel.
    """
    # "The kernel ker(E) ... gives us the generators" 
    rref_matrix = gaussian_elimination_gf2(matrix)
    n_rows, n_cols = rref_matrix.shape
    
    pivot_cols = []
    pivot_row = 0
    for j in range(n_cols):
        if pivot_row < n_rows and rref_matrix[pivot_row, j] == 1:
            pivot_cols.append(j)
            pivot_row += 1
            
    free_cols = [j for j in range(n_cols) if j not in pivot_cols]
    
    kernel_basis = []
    
    for free_col in free_cols:
        # Create a new basis vector
        basis_vector = np.zeros(n_cols, dtype=np.int8)
        
        # Set the free variable component to 1
        basis_vector[free_col] = 1
        
        # Solve for the pivot variables
        for i, pivot_col in enumerate(pivot_cols):
            if rref_matrix[i, free_col] == 1:
                basis_vector[pivot_col] = 1
                
        kernel_basis.append(basis_vector)
        
    return kernel_basis