from pareto_matching import find_pareto_efficient_matching, verify_pareto_efficiency

def main() -> None:
    """Example usage of the pareto_matching library."""
    # Example preferences
    preferences = {
        "Alice": ["Bob", "Charlie", "David"],
        "Bob": ["Charlie", "Alice", "David"],
        "Charlie": ["Alice", "Bob", "David"],
        "David": ["Alice", "Bob", "Charlie"]
    }
    
    # Find a Pareto-efficient matching
    result = find_pareto_efficient_matching(preferences, seed=42)
    
    # Print the results
    print("Matching:")
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