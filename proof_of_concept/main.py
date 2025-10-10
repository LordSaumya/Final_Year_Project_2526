from cmath import pi
import pennylane as qml
from typing import List

# --- 2 qubit prototype ---

# Maps from symmetries to operator sets
phi = pi / 2 # Example angle
theta = pi / 4 # Example angle

operator_pool = {
    "u1": [qml.PauliZ(0),
           qml.S(0), qml.T(0), qml.adjoint(qml.S(0)), qml.adjoint(qml.T(0)), qml.PhaseShift(0), qml.RZ(0),
           qml.PauliZ(1), qml.S(1), qml.T(1), qml.adjoint(qml.S(1)), qml.adjoint(qml.T(1)), qml.PhaseShift(phi, 1), qml.RZ(phi, 1)],
    "z2": [qml.PauliX(0), qml.RX(phi, 0), qml.MultiRZ(theta, 0), qml.PauliX(1), qml.RX(theta, 1), qml.MultiRZ(theta, 1)],
    "su2": [qml.Identity(0), qml.Identity(1), qml.exp(qml.sum(qml.prod(qml.PauliX(0), qml.PauliX(1)), qml.prod(qml.PauliY(0), qml.PauliY(1)), qml.prod(qml.PauliZ(0), qml.PauliZ(1))))],
    "time_reversal": [qml.PauliX(0), qml.PauliZ(0), qml.Hadamard(0), qml.RX(phi, 0), qml.RY(phi, 0), qml.PauliX(1), qml.PauliZ(1), qml.Hadamard(1), qml.RX(phi, 1), qml.RY(phi, 1)],
    # Add other symmetries and their corresponding operator sets
}

# Maps from symmetries to their generators
# A Hamiltonian must commute with all generators of a symmetry to exhibit that symmetry
symmetry_generators = {
    "u1": [qml.PauliZ(0) * qml.PauliZ(1)], 
    "z2_x": [qml.PauliX(0) * qml.PauliX(1)],
    "z2_y": [qml.PauliY(0) * qml.PauliY(1)],
    "z2_z": [qml.PauliZ(0) * qml.PauliZ(1)],
    "su2": [qml.Hamiltonian([0.5, 0.5], [qml.PauliX(0), qml.PauliX(1)]), qml.Hamiltonian([0.5, 0.5], [qml.PauliY(0), qml.PauliY(1)]), qml.Hamiltonian([0.5, 0.5], [qml.PauliZ(0), qml.PauliZ(1)])],
}

def commutes(op1, op2) -> bool:
    """Check if two operators commute."""
    return all((op1 @ op2 - op2 @ op1).simplify() == 0)

def find_symmetries(hamiltonian: qml.Hamiltonian) -> List[str]:
    """Identify symmetries present in the Hamiltonian.

    Args:
        hamiltonian (qml.Hamiltonian): The Hamiltonian to analyze.

    Returns:
        List[str]: A list of identified symmetries.
    """

    symmetries = []
    for symmetry, generators in symmetry_generators.items():
        if all(commutes(hamiltonian, gen) for gen in generators):
            symmetries.append(symmetry)
    return symmetries


def main():
    hamiltonian = qml.Hamiltonian(
        [0.5, -1.0, 0.8],
        [qml.PauliZ(0), qml.PauliX(1), qml.PauliZ(0) @ qml.PauliZ(1)]
    )

    symmetries = find_symmetries(hamiltonian)

    print("Identified Symmetries:", symmetries)

if __name__ == "__main__":
    main()
