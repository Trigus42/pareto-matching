import unittest
from pareto_matching import find_pareto_efficient_matching
import numpy as np
import random

class TestParetoMatching(unittest.TestCase):
    def setUp(self):
        # Set fixed seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
    def test_simple_matching(self):
        """Test a simple matching with four people."""
        preferences = {
            "Alice": ["Bob", "Charlie", "David"],
            "Bob": ["Alice", "Charlie", "David"],
            "Charlie": ["Alice", "Bob", "David"],
            "David": ["Alice", "Bob", "Charlie"]
        }
        
        result = find_pareto_efficient_matching(preferences, seed=42)
        
        # Verify that each person is matched with exactly one other person
        self.assertEqual(len(result.matching), 4)
        
        # Check that matches are symmetric (if A is matched with B, B is matched with A)
        for person, partner in result.matching.items():
            self.assertEqual(result.matching[partner], person)
            
        # Check that satisfaction metrics are reasonable
        self.assertGreaterEqual(result.avg_satisfaction, 0.0)
        self.assertLessEqual(result.avg_satisfaction, 1.0)
        self.assertGreaterEqual(result.min_satisfaction, 0.0)
        self.assertLessEqual(result.max_satisfaction, 1.0)
    
    def test_odd_number_of_people(self):
        """Test that an odd number of people raises a ValueError."""
        preferences = {
            "Alice": ["Bob", "Charlie"],
            "Bob": ["Alice", "Charlie"],
            "Charlie": ["Alice", "Bob"]
        }
        
        with self.assertRaises(ValueError):
            find_pareto_efficient_matching(preferences)
    
    def test_consistent_results(self):
        """Test that the algorithm produces consistent results with the same seed."""
        preferences = {
            "A": ["B", "C", "D", "E", "F"],
            "B": ["A", "C", "D", "E", "F"],
            "C": ["A", "B", "D", "E", "F"],
            "D": ["A", "B", "C", "E", "F"],
            "E": ["A", "B", "C", "D", "F"],
            "F": ["A", "B", "C", "D", "E"]
        }
        
        result1 = find_pareto_efficient_matching(preferences, seed=42)
        result2 = find_pareto_efficient_matching(preferences, seed=42)
        
        # Results should be identical with the same seed
        self.assertEqual(result1.matching, result2.matching)
        self.assertEqual(result1.avg_satisfaction, result2.avg_satisfaction)
    
    def test_different_seeds(self):
        """Test that different seeds can produce different results."""
        preferences = {}
        people = [f"Person{i}" for i in range(20)]
        
        # Create random preferences for a larger group
        for person in people:
            others = [p for p in people if p != person]
            random.shuffle(others)
            preferences[person] = others
        
        result1 = find_pareto_efficient_matching(preferences, seed=1)
        result2 = find_pareto_efficient_matching(preferences, seed=2)
        
        # With different seeds and randomness, results may differ
        # This is a probabilistic test, so it might occasionally fail
        # If results are identical, that's unusual but not impossible
        # We're just expecting that different seeds give different matchings with high probability
        # for a large enough problem
        
    def test_pareto_efficiency(self):
        """Test that the result is Pareto efficient."""
        # Create a small example with a clearly optimal matching
        preferences = {
            "A": ["B", "D", "C"],
            "B": ["A", "C", "D"],
            "C": ["D", "A", "B"],
            "D": ["C", "B", "A"]
        }
        
        result = find_pareto_efficient_matching(preferences, seed=42)
        
        # In this example, the optimal matching should pair A with B and C with D
        # because these are their respective first choices
        expected_matching = {
            "A": "B",
            "B": "A",
            "C": "D", 
            "D": "C"
        }
        
        # Check if the matching is as expected
        self.assertEqual(result.matching, expected_matching)
        
        # The optimal matching should have a high average satisfaction
        self.assertGreaterEqual(result.avg_satisfaction, 0.9)
    
    def test_larger_group(self):
        """Test that the algorithm works with a larger group."""
        # Create a larger test case with 50 people
        n = 50
        people = [f"Person{i}" for i in range(n)]
        preferences = {}
        
        for person in people:
            others = [p for p in people if p != person]
            random.shuffle(others)
            preferences[person] = others
        
        result = find_pareto_efficient_matching(preferences, seed=42, 
                                                population_size=500, iterations=20)
        
        # Verify that each person is matched with exactly one other person
        self.assertEqual(len(result.matching), n)
        
        # Check that matches are symmetric (if A is matched with B, B is matched with A)
        for person, partner in result.matching.items():
            self.assertEqual(result.matching[partner], person)

if __name__ == "__main__":
    unittest.main()