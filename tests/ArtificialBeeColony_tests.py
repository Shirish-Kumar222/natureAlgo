import unittest
import pandas as pd
import numpy as np

import sys
import os
myDir = os.getcwd()
sys.path.append(myDir)

from pathlib import Path
path = Path(myDir)
a=str(path.parent.absolute())

sys.path.append(a)

from natureAlgo.bee_colony import ArtificialBeeColony

class TestArtificialBeeColony(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        self.data = pd.DataFrame(np.random.rand(10, 2), columns=['Feature1', 'Feature2'])
        
        # Define the objective function (sum of squares)
        def objective_function(x):
            return sum([x_i ** 2 for x_i in x])
        
        # Initialize the ABC algorithm with sample parameters
        self.abc = ArtificialBeeColony(self.data, objective_function, max_iterations=100, num_employed=10, num_onlookers=10)
    
    def test_best_solution(self):
        # Run the ABC algorithm
        self.abc.run()
        
        # Check if the best solution is within the search space
        self.assertTrue(all(self.data[col].min() <= val <= self.data[col].max() for col, val in zip(self.data.columns, self.abc.best_solution)))
        
    def test_best_fitness(self):
        # Run the ABC algorithm
        self.abc.run()
        
        # Calculate the fitness of the best solution
        best_fitness_calculated = sum([val ** 2 for val in self.abc.best_solution])
        
        # Check if the calculated fitness matches the reported best fitness
        self.assertEqual(self.abc.best_fitness, best_fitness_calculated)

if __name__ == '__main__':
    unittest.main()