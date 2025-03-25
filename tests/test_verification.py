import unittest
from pareto_matching import find_pareto_efficient_matching, verify_pareto_efficiency

class TestParetoVerification(unittest.TestCase):
    def test_verify_pareto_efficiency(self):
        # Example with an optimal matching
        preferences = {
            "A": ["B", "D", "C"],
            "B": ["A", "C", "D"],
            "C": ["D", "A", "B"],
            "D": ["C", "B", "A"]
        }
        
        # The optimal matching where everyone gets their first choice
        optimal_matching = {
            "A": "B",
            "B": "A",
            "C": "D",
            "D": "C"
        }
        
        # Verify that the optimal matching is Pareto-efficient
        self.assertTrue(verify_pareto_efficiency(preferences, optimal_matching))
        
        # A non-optimal matching that is not Pareto-efficient
        non_optimal_matching = {
            "A": "C",
            "C": "A",
            "B": "D",
            "D": "B"
        }
        
        # Verify that the non-optimal matching is not Pareto-efficient
        # In this case, A and B would both prefer to be matched with each other
        # and C and D would both prefer to be matched with each other
        self.assertFalse(verify_pareto_efficiency(preferences, non_optimal_matching))
    
    def test_algorithm_produces_pareto_efficient_matchings(self):
        # Test that our algorithm produces Pareto-efficient matchings
        preferences = {
            "Alice": ["Bob", "Charlie", "David"],
            "Bob": ["Alice", "Charlie", "David"],
            "Charlie": ["Alice", "Bob", "David"],
            "David": ["Alice", "Bob", "Charlie"]
        }
        
        result = find_pareto_efficient_matching(preferences, seed=42)
        
        # The result should be Pareto-efficient
        self.assertTrue(result.is_pareto_efficient)
        
        # Double-check with our verification function
        self.assertTrue(verify_pareto_efficiency(preferences, result.matching))

if __name__ == "__main__":
    unittest.main()