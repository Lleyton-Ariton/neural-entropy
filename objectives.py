import math
import numpy as np

from scipy.stats import entropy

from typing import Optional, List, Dict, Tuple


def construct_transition_matrix(
        states: np.ndarray,
        transitions: Optional[List[Tuple[int, int]]] = None,
        indices: Optional[Dict[str, int]] = None,
        n: int = 0
) -> Tuple[np.ndarray, Tuple[List[Tuple[int, int]], Dict[str, int]], int]:
    if transitions is None:
        transitions = []

    if indices is None:
        indices = {}

    for t in range(0, len(states) - 1):
        state_t = ''.join(map(lambda element: str(element), states[t]))
        state_t_plus_one = ''.join(
            map(lambda element: str(element), states[t + 1]))

        if indices.get(state_t, None) is None:
            indices[state_t] = n

            n += 1

        if indices.get(state_t_plus_one, None) is None:
            indices[state_t_plus_one] = n

            n += 1

        transitions.append((indices[state_t], indices[state_t_plus_one]))

    transition_matrix = np.zeros((n, n), dtype=int)
    for transition in transitions:
        transition_matrix[transition[0], transition[1]] += 1

    return transition_matrix, (transitions, indices), n


def topological_entropy_objective(transition_matrix: np.ndarray) -> float:
    if transition_matrix.size == 0 or np.all(transition_matrix == 0):
        return 0.0

    transition_matrix = (transition_matrix > 0).astype(np.float32)

    eigenvalues = np.linalg.eigvals(transition_matrix)
    spectral_radius = np.abs(eigenvalues).max()

    if spectral_radius == 0.0:
        return 0.0

    return math.log(spectral_radius)


def shannon_entropy_objective(transition_matrix: np.ndarray) -> float:
    if transition_matrix.size == 0 or np.all(transition_matrix == 0):
        return 0.0

    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1

    p_matrix = transition_matrix / row_sums

    eigenvalues, eigenvectors = np.linalg.eig(p_matrix.T)

    stationary_distribution = eigenvectors[:, np.isclose(eigenvalues, 1)][:, 0]
    stationary_distribution = stationary_distribution.real

    stationary_distribution[stationary_distribution < 0] = 0
    stationary_distribution /= np.sum(stationary_distribution)

    return entropy(stationary_distribution)


def entropy_rate_objective(transition_matrix: np.ndarray) -> float:
    if transition_matrix.size == 0 or np.all(transition_matrix == 0):
        return 0.0

    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1

    p_matrix = transition_matrix / row_sums

    eigenvalues, eigenvectors = np.linalg.eig(p_matrix.T)

    stationary_distribution = eigenvectors[:, np.isclose(eigenvalues, 1)][:, 0]
    stationary_distribution = stationary_distribution.real

    stationary_distribution[stationary_distribution < 0] = 0
    stationary_distribution /= np.sum(stationary_distribution)

    log_p_matrix = np.where(p_matrix != 0, np.log(p_matrix), 0)

    return -np.sum(np.dot(stationary_distribution, p_matrix) * log_p_matrix)
