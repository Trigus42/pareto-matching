from typing import Dict, List, TypeAlias, Any, Tuple
import numpy as np
import random
from collections.abc import Sequence
from dataclasses import dataclass

Person: TypeAlias = str
PreferenceList: TypeAlias = List[Person]
Matching: TypeAlias = Dict[Person, Person]

@dataclass(frozen=True)
class MatchingResult:
    """Result of the matching algorithm with metrics."""
    matching: Matching
    avg_satisfaction: float
    min_satisfaction: float
    max_satisfaction: float
    is_pareto_efficient: bool

def find_pareto_efficient_matching(
    preferences: Dict[Person, PreferenceList], 
    population_size: int = 1000,
    iterations: int = 100,
    seed: int | None = None
) -> MatchingResult:
    """
    Find a Pareto-efficient matching for the stable roommates problem.
    
    Args:
        preferences: Dictionary mapping each person to their ordered list of preferences
        population_size: Number of matchings to generate in the population
        iterations: Number of improvement iterations
        seed: Random seed for reproducibility
        
    Returns:
        MatchingResult with the Pareto-efficient matching and metrics
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    people = list(preferences.keys())
    n = len(people)
    
    if n % 2 != 0:
        raise ValueError("Number of people must be even for complete matching")
    
    # Create a mapping from person to index
    person_to_idx: Dict[Person, int] = {p: i for i, p in enumerate(people)}
    idx_to_person: Dict[int, Person] = {i: p for i, p in enumerate(people)}
    
    # Convert preferences to utility matrices (lower rank = higher utility)
    # utility[i, j] = utility person i gets from being matched with person j
    preference_matrix = np.zeros((n, n), dtype=np.float64)
    
    for person, prefs in preferences.items():
        i = person_to_idx[person]
        for rank, preferred in enumerate(prefs):
            j = person_to_idx[preferred]
            # Normalize utility to [0, 1] range (reversed order: higher rank = lower utility)
            preference_matrix[i, j] = 1.0 - (rank / (n - 1))
    
    # Set diagonal to -inf (can't match with self)
    np.fill_diagonal(preference_matrix, -np.inf)
    
    # Generate initial population of feasible matchings
    population = [_generate_random_matching(n) for _ in range(population_size)]
    
    # Improvement phase - local search to find better matchings
    for _ in range(iterations):
        # Generate some new matchings through local perturbations
        new_matchings = []
        for matching in population[:population_size//5]:  # Focus on the top 20%
            for _ in range(3):  # Generate 3 variations per matching
                new_matching = _perturb_matching(matching.copy())
                new_matchings.append(new_matching)
        
        population.extend(new_matchings)
        
        # Evaluate and trim population
        population = _evaluate_and_select(population, preference_matrix, population_size)
    
    # Calculate utility vectors for final population
    utility_vectors = np.zeros((len(population), n))
    for i, matching in enumerate(population):
        utility_vectors[i] = _calculate_utilities(matching, preference_matrix)
    
    # Find Pareto-efficient matchings using our own implementation
    pareto_indices = _find_pareto_efficient_indices(utility_vectors)
    
    # Choose the best Pareto-efficient matching (maximize minimum utility)
    best_idx = _select_best_matching(utility_vectors, pareto_indices)
    best_matching = population[best_idx]
    
    # Convert to result format
    result_matching: Matching = {}
    for i, j in enumerate(best_matching):
        result_matching[idx_to_person[i]] = idx_to_person[j]
    
    # Calculate metrics
    best_utils = utility_vectors[best_idx]
    avg_satisfaction = float(np.mean(best_utils))
    min_satisfaction = float(np.min(best_utils))
    max_satisfaction = float(np.max(best_utils))
    
    # Verify Pareto efficiency
    is_pareto_efficient = verify_pareto_efficiency(preferences, result_matching)
    
    return MatchingResult(
        matching=result_matching,
        avg_satisfaction=avg_satisfaction,
        min_satisfaction=min_satisfaction,
        max_satisfaction=max_satisfaction,
        is_pareto_efficient=is_pareto_efficient
    )

def verify_pareto_efficiency(
    preferences: Dict[Person, PreferenceList], 
    matching: Matching
) -> bool:
    """
    Verify that a matching is Pareto-efficient by checking for trading cycles.
    
    A matching is Pareto-efficient if no alternative matching makes at least one person
    better off without making anyone worse off.
    
    Args:
        preferences: Dictionary mapping each person to their ordered list of preferences
        matching: Dictionary mapping each person to their partner
        
    Returns:
        True if the matching is Pareto-efficient, False otherwise
    """
    # Create preference rankings (lower rank = higher preference)
    rankings: Dict[Person, Dict[Person, int]] = {}
    for person, prefs in preferences.items():
        rankings[person] = {prefs[i]: i for i in range(len(prefs))}
        # Add any missing people with lowest preference
        lowest_rank = len(prefs)
        for other in preferences.keys():
            if other != person and other not in rankings[person]:
                rankings[person][other] = lowest_rank
    
    people = list(matching.keys())
    n = len(people)
    
    # Check for pairwise swaps (cycle length 2)
    for i in range(n):
        for j in range(i+1, n):
            a = people[i]
            b = people[j]
            
            # Skip if they're already paired
            if matching[a] == b:
                continue
                
            c = matching[a]  # a's partner
            d = matching[b]  # b's partner
            
            # Check if this swap is a Pareto improvement
            # Current happiness (lower rank = happier)
            current_happiness = [
                rankings[a][c],
                rankings[b][d],
                rankings[c][a],
                rankings[d][b]
            ]
            
            # New happiness after the swap
            new_happiness = [
                rankings[a][b],
                rankings[b][a],
                rankings[c][d],
                rankings[d][c]
            ]
            
            # If everyone is at least as happy and someone is happier, this is a Pareto improvement
            all_at_least_as_happy = all(new_happiness[i] <= current_happiness[i] for i in range(4))
            someone_happier = any(new_happiness[i] < current_happiness[i] for i in range(4))
            
            if all_at_least_as_happy and someone_happier:
                return False  # Found a Pareto improvement, so not Pareto-efficient
    
    # Check for 3-cycles
    for i in range(n):
        a = people[i]
        a_partner = matching[a]
        
        for j in range(n):
            if j == i or people[j] == a_partner:
                continue
                
            b = people[j]
            b_partner = matching[b]
            
            if b_partner == a:
                continue
            
            for k in range(n):
                if k == i or k == j or people[k] == a_partner or people[k] == b_partner:
                    continue
                    
                c = people[k]
                c_partner = matching[c]
                
                if c_partner == a or c_partner == b:
                    continue
                
                # Check if the cycle a -> b_partner -> c_partner -> a is a Pareto improvement
                # Current happiness
                current_happiness = {
                    a: rankings[a][a_partner],
                    b: rankings[b][b_partner],
                    c: rankings[c][c_partner],
                    a_partner: rankings[a_partner][a],
                    b_partner: rankings[b_partner][b],
                    c_partner: rankings[c_partner][c]
                }
                
                # New happiness after the 3-cycle trade
                new_happiness = {
                    a: rankings[a][b_partner],
                    b: rankings[b][c_partner],
                    c: rankings[c][a_partner],
                    a_partner: rankings[a_partner][c],
                    b_partner: rankings[b_partner][a],
                    c_partner: rankings[c_partner][b]
                }
                
                # Check if this is a Pareto improvement
                all_at_least_as_happy = all(
                    new_happiness[p] <= current_happiness[p] 
                    for p in current_happiness
                )
                
                someone_happier = any(
                    new_happiness[p] < current_happiness[p] 
                    for p in current_happiness
                )
                
                if all_at_least_as_happy and someone_happier:
                    return False  # Found a Pareto improvement, so not Pareto-efficient
    
    # No Pareto improvements found
    return True

def _find_pareto_efficient_indices(vectors: np.ndarray) -> List[int]:
    """
    Find the indices of Pareto-efficient points in a set of utility vectors.
    
    A point is Pareto-efficient if no other point dominates it
    (i.e., is at least as good in all dimensions and strictly better in at least one).
    
    Args:
        vectors: Matrix of utility vectors (rows=solutions, columns=person utilities)
        
    Returns:
        List of indices of Pareto-efficient points
    """
    n_points = vectors.shape[0]
    is_efficient = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        if is_efficient[i]:
            # Find points that dominate this point
            dominated_mask = np.all(vectors >= vectors[i, :], axis=1) & np.any(vectors > vectors[i, :], axis=1)
            
            # If any such points exist, this point is not Pareto-efficient
            if np.any(dominated_mask):
                is_efficient[i] = False
            else:
                # Find points dominated by this point (efficient point)
                dominated_by_i = np.all(vectors[i, :] >= vectors, axis=1) & np.any(vectors[i, :] > vectors, axis=1)
                is_efficient[dominated_by_i] = False
    
    return list(np.where(is_efficient)[0])

def _generate_random_matching(n: int) -> List[int]:
    """Generate a random valid matching where each person is paired with exactly one other person."""
    # A matching is represented as a list where matching[i] = j means i is matched with j
    people = list(range(n))
    random.shuffle(people)
    
    matching = [-1] * n
    for i in range(0, n, 2):
        p1 = people[i]
        p2 = people[i + 1]
        matching[p1] = p2
        matching[p2] = p1
    
    return matching

def _perturb_matching(matching: List[int]) -> List[int]:
    """Create a new matching by swapping two pairs."""
    n = len(matching)
    # Randomly select two people who are not currently matched
    while True:
        i = random.randrange(n)
        j = random.randrange(n)
        if i != j and matching[i] != j:
            break
    
    # Get their current partners
    i_partner = matching[i]
    j_partner = matching[j]
    
    # Swap partners
    matching[i] = j_partner
    matching[j_partner] = i
    matching[j] = i_partner
    matching[i_partner] = j
    
    return matching

def _calculate_utilities(matching: List[int], preference_matrix: np.ndarray) -> np.ndarray:
    """Calculate utility for each person based on their preferences."""
    n = len(matching)
    utilities = np.zeros(n)
    
    for i in range(n):
        partner = matching[i]
        utilities[i] = preference_matrix[i, partner]
    
    return utilities

def _evaluate_and_select(
    population: List[List[int]], 
    preference_matrix: np.ndarray, 
    max_size: int
) -> List[List[int]]:
    """Evaluate matchings and select the best ones."""
    # Calculate utilities for each matching
    utilities = []
    for matching in population:
        match_utils = _calculate_utilities(matching, preference_matrix)
        # Use different measures for ranking
        min_util = np.min(match_utils)
        avg_util = np.mean(match_utils)
        sum_util = np.sum(match_utils)
        
        utilities.append((min_util, avg_util, sum_util, matching))
    
    # Sort by min utility (reversed for maximin), then by average, then by sum
    utilities.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    
    # Return the top matchings
    return [u[3] for u in utilities[:max_size]]

def _select_best_matching(
    utility_vectors: np.ndarray, 
    pareto_indices: Sequence[int]
) -> int:
    """
    Select the best matching among Pareto-efficient matchings.
    Uses the leximin principle: maximize the minimum utility,
    then the second minimum, and so on.
    """
    best_idx = pareto_indices[0]
    best_sorted_utilities = sorted(utility_vectors[best_idx])
    
    for idx in pareto_indices[1:]:
        current_sorted = sorted(utility_vectors[idx])
        
        # Compare lexicographically
        for i in range(len(current_sorted)):
            if current_sorted[i] > best_sorted_utilities[i]:
                best_sorted_utilities = current_sorted
                best_idx = idx
                break
            elif current_sorted[i] < best_sorted_utilities[i]:
                break
    
    return best_idx