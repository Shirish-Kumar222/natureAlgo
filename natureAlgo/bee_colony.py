import random
import math

class ArtificialBeeColony:
    def __init__(self, data, objective_function, max_iterations, num_employed, num_onlookers):
        self.data = data
        self.objective_function = objective_function
        self.max_iterations = max_iterations
        self.num_employed = num_employed
        self.num_onlookers = num_onlookers
        self.num_dimensions = len(data.columns)
        self.search_space = [(data[col].min(), data[col].max()) for col in data.columns]
        self.best_solution = None
        self.best_fitness = float('inf')
    
    def run(self):
        # Initialize the population of employed bees with random solutions
        employed_bees = [{'solution': [random.uniform(self.search_space[i][0], self.search_space[i][1]) for i in range(self.num_dimensions)],
                          'fitness': None,
                          'trials': 0}
                         for _ in range(self.num_employed)]
        
        # Evaluate the fitness of employed bees' solutions using the user-defined objective function
        for bee in employed_bees:
            bee['fitness'] = self.objective_function(bee['solution'])
        
        # Main loop
        for iteration in range(self.max_iterations):
            # Employed bees phase
            for bee in employed_bees:
                neighbor_solution = bee['solution'][:]
                dimension_to_change = random.randint(0, self.num_dimensions - 1)
                neighbor_solution[dimension_to_change] = random.uniform(self.search_space[dimension_to_change][0],
                                                                        self.search_space[dimension_to_change][1])
                neighbor_fitness = self.objective_function(neighbor_solution)
                if neighbor_fitness < bee['fitness']:
                    bee['solution'] = neighbor_solution
                    bee['fitness'] = neighbor_fitness
                    bee['trials'] = 0
                else:
                    bee['trials'] += 1
            
            # Onlooker bees phase
            total_fitness = sum(bee['fitness'] for bee in employed_bees)
            for bee in employed_bees:
                probability = bee['fitness'] / total_fitness
                if random.random() < probability:
                    neighbor_solution = bee['solution'][:]
                    dimension_to_change = random.randint(0, self.num_dimensions - 1)
                    neighbor_solution[dimension_to_change] = random.uniform(self.search_space[dimension_to_change][0],
                                                                            self.search_space[dimension_to_change][1])
                    neighbor_fitness = self.objective_function(neighbor_solution)
                    if neighbor_fitness < bee['fitness']:
                        bee['solution'] = neighbor_solution
                        bee['fitness'] = neighbor_fitness
                        bee['trials'] = 0
                    else:
                        bee['trials'] += 1
            
            # Scout bees phase
            for bee in employed_bees:
                if bee['trials'] >= 3:
                    bee['solution'] = [random.uniform(self.search_space[i][0], self.search_space[i][1]) for i in range(self.num_dimensions)]
                    bee['fitness'] = self.objective_function(bee['solution'])
                    bee['trials'] = 0
        
        # Update the best solution found
        self.best_solution = min(employed_bees, key=lambda bee: bee['fitness'])['solution']
        self.best_fitness = min(employed_bees, key=lambda bee: bee['fitness'])['fitness']