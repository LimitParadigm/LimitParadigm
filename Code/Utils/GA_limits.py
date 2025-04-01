import pygad
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Any

class GA_limits:
    """Genetic Algorithm implementation for optimizing network limits.
    
    This class implements a GA to optimize network constraints, particularly
    for power systems with PV production and EV considerations.
    """

    def __init__(self, network_obj: Any, limit_obj: Any,
                 population_size: int = 30,
                 num_generations: int = 50,
                 mutation_rate: float = 0.4,
                 early_stopping_generations: int = 10,
                 early_stopping_threshold: float = 1e-6):
        """Initialize the GA optimizer.
        
        Args:
            network_obj: Network object containing system constraints
            limit_obj: Object defining limit verification methods
            population_size: Number of solutions in population
            num_generations: Maximum number of generations
            mutation_rate: Rate of mutation for genetic diversity
            early_stopping_generations: Generations to check for improvement
            early_stopping_threshold: Minimum improvement threshold
        """
        self.network_obj = network_obj
        self.limit_obj = limit_obj
        self.okay = 0
        self.total = 0
        self.mutation_rate = mutation_rate
        self.keep_parents = None
        self.num_generations = num_generations
        self.population_size = population_size
        self.number_genes = network_obj.number_customers
        self.best_solution_found = None
        self.mistakes: List[np.ndarray] = []
        self.fitnesses: List[float] = []
        
        self.early_stopping_gens = early_stopping_generations
        self.early_stopping_thresh = early_stopping_threshold
        
        self.feeder = None
        self.initial_solution = None

        self.scaling_factor = 1.0
        self.decrease_scaling_factor = 0.05
        self.c = 0

        self.n = self.network_obj.number_customers
        self.gene_space = [{} for _ in range(self.n * 2)]
        for i in range(self.n):
            #Export (production -> [PV peak production,0])
            self.gene_space[i] = {'low': float(self.network_obj.contractual_limits[i,0]), 'high':0}
            #Import [2, contractual power + EV])
            self.gene_space[self.n+i] = {'low': 1, 'high': float(self.network_obj.contractual_limits[i,1])}

    def generate_individual(self) -> np.ndarray:
        """Generate a random individual solution."""
        L = np.ones(self.network_obj.limits_shape)
        for i,c in enumerate(self.network_obj.contractual_limits):
            for j,l in enumerate(c):
                L[i,j] = np.random.random() * self.network_obj.contractual_limits[i,j] * self.scaling_factor
        return L.flatten('F')

    def fitness_func(self, ga_instance: pygad.GA, solution: np.ndarray, solution_idx: int) -> float:
        """Evaluate the fitness of a solution."""
        #TODO: this function can be improved by addining a factor for the deviation in limits between the customers
        self.total+=1
        if(self.limit_obj.SafetyVerification(solution, True) == False):
            self.mistakes.append(solution.copy())
            self.scaling_factor -= self.decrease_scaling_factor
            self.scaling_factor = self.scaling_factor if self.scaling_factor>0.1 else 0.1
            return 0
        self.scaling_factor += self.decrease_scaling_factor
        self.scaling_factor = self.scaling_factor if self.scaling_factor<0.9 else 0.9
        self.okay+=1
        sol = self.limit_obj.reshape_function(solution)
        fitness = self.limit_obj.objective_function(sol)
        deviation = np.std(sol[:,0]) + np.std(sol[:,1])
        fitness -= deviation * 5
        # fitness = 1.0 / (fitness + 1e-9)
        return fitness

    def on_mutation(self, ga_instance: pygad.GA, offspring_mutation: np.ndarray) -> None:
        """Apply mutation to offspring."""
        for i, gene in enumerate(offspring_mutation):
            for j, value in enumerate(gene):
                column = j // self.n
                if column == 0:
                    v = max(self.gene_space[j]['low'], min(self.gene_space[j]['high'], value))
                elif column == 1:
                    v = max(self.gene_space[j]['low'], min(self.gene_space[j]['high'], value))
                offspring_mutation[i, j] = v

    def on_generation(self, ga_instance: pygad.GA) -> None:
        """Called after each generation to update progress."""
        tmp_best_solution = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
        self.fitnesses.append(ga_instance.last_generation_fitness)
        
        diff = tmp_best_solution[1] if self.best_solution_found is None else tmp_best_solution[1] - self.best_solution_found[1]
        if ga_instance.generations_completed % 10 == 0 or diff > 0:
            print(f"Generation = {ga_instance.generations_completed}/{self.num_generations}")
        
        if diff>0:
            print(f"Fitness = {tmp_best_solution[1]} (diff: {diff})")
            self.best_solution_found = tmp_best_solution
        elif self.best_solution_found is None:
            self.best_solution_found = tmp_best_solution

        for i, gene in enumerate(ga_instance.population):
            for j, value in enumerate(gene):
                if i > self.keep_parents and np.random.rand() < self.mutation_rate:
                    customer = j % self.n
                    column = j // self.n
                    v = np.random.rand() * self.network_obj.contractual_limits[customer,column] * self.scaling_factor
                    ga_instance.population[i,j] = v

    def initialize_run(self) -> pygad.GA:
        """Initialize the GA run."""
        num_parents_mating = max(4, int(self.population_size*0.4)) # Number of solutions to be selected as parents in the mating pool.
        self.keep_parents = np.max([4,int(self.population_size*0.1)])
        
        # tested_combinations = self.num_generations*self.population_size
        # print(f"There are {possible_combinations:e} possible combinations to test (good luck with that!!). Solutions that will be tested: {tested_combinations} ({(tested_combinations/possible_combinations*100):.2f}%)")
        # print(f"Initial solution: {self.feederbalancing.B_init_nobinary}. Number customers: {len(self.feederbalancing.B_init_nobinary)}\n")

        initial_population = [self.generate_individual() for _ in range(self.population_size)]

        ga_instance = pygad.GA(num_generations=self.num_generations,
                            num_genes=self.number_genes,
                            gene_type=float,
                            sol_per_pop=self.population_size,
                            num_parents_mating=num_parents_mating,
                            keep_parents=self.keep_parents,
                            keep_elitism=self.keep_parents,
                            # gene_space=self.gene_space,
                            save_solutions=True,
                            save_best_solutions=True,
                            initial_population=initial_population,
                            fitness_func=self.fitness_func,
                            on_generation=self.on_generation,
                            random_seed=14,
                            suppress_warnings=True, 
                            # mutation_type=None,
                            on_mutation=self.on_mutation,
                            # crossover_type=None
                            )
        return ga_instance

    def runGA(self) -> 'GA_limits':
        """Run the genetic algorithm optimization process.
        
        Returns:
            self: Returns instance for method chaining
        """
        ga_instance = self.initialize_run()

        initial_fitness = self.limit_obj.objective_function(self.network_obj.contractual_limits, True)
        print(f'Initial fitness: {initial_fitness}. Feasible: {self.limit_obj.SafetyVerification(self.network_obj.contractual_limits)}')

        ga_instance.run()
        
        # Plot fitness history
        ga_instance.plot_fitness()
        
        # Return optimization results
        solution, solution_fitness, solution_idx = self.best_solution_found
        print(f"Parameters of the best solution: \n {self.limit_obj.reshape_function(solution)}")
        print(f"Fitness value of the best solution = {solution_fitness}")
        print(f"Best fitness value reached after {ga_instance.best_solution_generation} generations.")

        # Saving the GA instance.
        # filename = 'genetic' # The filename to which the instance is saved. The name is without extension.
        # ga_instance.save(filename=filename)

        # # Loading the saved GA instance.
        # loaded_ga_instance = pygad.load(filename=filename)
        # loaded_ga_instance.plot_fitness()

        print(self.okay, self.total, self.okay/self.total, self.scaling_factor)
        self.ga_instance = ga_instance

        self.limit_obj.limits = self.limit_obj.reshape_function(solution)
        return self