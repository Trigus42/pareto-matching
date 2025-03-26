from typing import Dict, List, Optional
from pareto_matching import find_pareto_efficient_matching, MatchingResult, Person, PreferenceList, verify_pareto_efficiency

def find_best_matching(
    preferences: Dict[Person, PreferenceList], 
    runs: int = 10
) -> MatchingResult:
    """
    Run the matching algorithm multiple times with different seeds
    and select the best result based on minimum satisfaction.
    
    Args:
        preferences: Dictionary mapping each person to their ordered list of preferences
        runs: Number of times to run the algorithm with different seeds
        
    Returns:
        The MatchingResult with the highest minimum satisfaction
    """
    best_result: Optional[MatchingResult] = None
    best_min_satisfaction: float = -float('inf')
    
    for i in range(runs):
        # Use different seeds for each run
        result = find_pareto_efficient_matching(
            preferences,
            population_size=2000,
            iterations=1000,
            seed=i
        )
        
        # Use minimum satisfaction as the primary selection criterion
        if best_result is None or result.min_satisfaction > best_min_satisfaction:
            best_min_satisfaction = result.min_satisfaction
            best_result = result
        elif result.min_satisfaction == best_min_satisfaction:
            # If tied on minimum satisfaction, prefer higher average
            if best_result is not None and result.avg_satisfaction > best_result.avg_satisfaction:
                best_result = result
                
    # This should never happen since we initialize with at least one run
    assert best_result is not None, "No matching found after multiple runs"
    
    return best_result


def main() -> None:
    """Example usage of the pareto_matching library."""
    # Example preferences
    preferences = {
        "Alice": ["Bob", "Charlie", "David"],
        "Bob": ["Charlie", "Alice", "David"],
        "Charlie": ["Alice", "Bob", "David"],
        "David": ["Alice", "Bob", "Charlie"]
    }
    
    # Find best Pareto-efficient matching
    result = find_best_matching(preferences, runs=5)
    
    # Print the results
    print("\nBest matching after multiple runs:")
    for person, partner in result.matching.items():
        print(f"{person} is matched with {partner}")
    
    print(f"\nAverage satisfaction: {result.avg_satisfaction:.4f}")
    print(f"Minimum satisfaction: {result.min_satisfaction:.4f}")
    print(f"Maximum satisfaction: {result.max_satisfaction:.4f}")
    print(f"Is Pareto-efficient: {result.is_pareto_efficient}")
    
    # You can also verify Pareto efficiency separately
    is_efficient = verify_pareto_efficiency(preferences, result.matching)
    print(f"Verified Pareto efficiency: {is_efficient}")

if __name__ == "__main__":
    main()