from genetic_clustering import ClusteringSolution, Point2D, random_index, crossover, mutation
from data import small_dataset_0, generate_dataset, plot_clusters
from itertools import permutations

def parallel_islands_execution_main(
    dataset: list[tuple], number_of_solutions: int, generations: int, 
    mutation_rate: int, number_of_clusters: int,  number_of_islands: int, migration_rate: float
):
    
    import ray
    import numpy as np
    from random import randint

    @ray.remote
    class Island:
        def __init__(self, island_id, number_of_solutions, number_of_clusters, mutation_rate, clusters_permutations, points, generations, migration_rate):
            self.island_id = island_id
            self.number_of_clusters = number_of_clusters
            self.population = sorted(
                [ClusteringSolution.create_random(points, number_of_clusters) for _ in range(number_of_solutions)],
                key=lambda s: s.quality, reverse=True
            )
            self.mutation_rate = mutation_rate
            self.clusters_permutations = clusters_permutations
            self.points = points
            self.generations = generations
            self.migration_rate = migration_rate
            self.other_islands = []

        def run_all_generations(self):
            for _ in range(self.generations):
                self.run_generation()

        def run_generation(self):
            for _ in range(len(self.population)):
                good_solution, bad_solution = self.select_solutions()
                child_solution = self.genetic_process(good_solution, bad_solution)
                self.update_population(child_solution)
            self.perform_migration()

        def genetic_process(self, good_solution, bad_solution):
            child_solution = crossover(
                good_solution, bad_solution, self.clusters_permutations, self.points
            )
            return mutation(
                child_solution, self.mutation_rate, self.number_of_clusters, self.points
            )

        def update_population(self, child_solution):
            random_bad_solution_idx = randint(0, len(self.population) - 1)
            self.population[random_bad_solution_idx] = child_solution
            self.population.sort(key=lambda s: s.quality, reverse=True)

        def get_other_islands(self):
            if not self.other_islands:
                self.other_islands = [ray.get_actor(f"island_{i}") for i in range(number_of_islands) if i != self.island_id]
            return self.other_islands

        def perform_migration(self):
            if np.random.rand() < self.migration_rate and self.other_islands:
                other_island = np.random.choice(self.other_islands)
                # Initiate migration without waiting for it to complete
                self.migrate_solutions.remote(other_island)

        def migrate_solutions(self, other_island):
            if self.population:
                # Send the solution to the other island asynchronously
                outgoing_solution = self.population[-1]
                other_island.receive_solution.remote(outgoing_solution)
                # Optionally, you can handle the incoming solution asynchronously in another method

        def receive_solution(self, incoming_solution):
            # This method can be called remotely by another island to send a solution
            # Integrate the incoming solution into your population
            self.population.append(incoming_solution)  # Example integration method
            self.population.sort(key=lambda s: s.quality, reverse=True)


        def select_solutions(self):
            good_solution = self.population[randint(0, len(self.population) // 2)]
            bad_solution = self.population[randint(len(self.population) // 2, len(self.population) - 1)]
            return good_solution, bad_solution

        def get_best_solution(self):
            return self.population[0] if self.population else None

        
    ray.init()

    points = [Point2D(x, y) for x, y in dataset]
    clusters_permutations: list[tuple] = list(permutations(list(range(number_of_clusters))))

    # Create islands (actors)
    islands = [Island.options(name=f"island_{i}").remote(
                i, number_of_solutions, number_of_clusters, mutation_rate, 
                clusters_permutations, points, generations, migration_rate)
               for i in range(number_of_islands)]

    # Start all generations on each island
    all_generation_tasks = [island.run_all_generations.remote() for island in islands]
    ray.get(all_generation_tasks)

    # Retrieve the best solution from all islands
    best_solutions = ray.get([island.get_best_solution.remote() for island in islands])
    best_solution = min(best_solutions, key=lambda s: s.quality)

    print("Final best solution quality : ", best_solution.quality)
    ray.shutdown()

    return points, best_solution, number_of_clusters


if __name__ == "__main__":
    import sys
    # Check if at least one additional argument is provided
    if len(sys.argv) > 1:
        # Access the first argument (after the script name)
        first_arg = sys.argv[1]

        # Perform different actions based on the argument
        if first_arg == "test":
            # Parameters
            number_of_solutions = 50
            generations = 80
            mutation_rate = 5
            number_of_clusters = 4
            number_of_tasks = 8

            
            points, solution, number_of_clusters = parallel_islands_execution_main(
                small_dataset_0, number_of_solutions, generations, mutation_rate, number_of_clusters, number_of_tasks, 0.1
            )
            plot_clusters(points, solution.assignments, number_of_clusters)
        elif first_arg == "random":
            number_of_clusters = 5

            large_dataset = generate_dataset(num_centers=number_of_clusters, min_points=2, max_points=10)

            # print size of dataset
            print('dataset size', len(large_dataset))


            number_of_solutions = 50
            generations = 200
            mutation_rate = 5

            number_of_tasks = 8

            points, solution, number_of_clusters = parallel_islands_execution_main(
                large_dataset, number_of_solutions, generations, mutation_rate, number_of_clusters, number_of_tasks, 0.1
            )
            plot_clusters(points, solution.assignments, number_of_clusters)
        else:
            print("Invalid argument")
    else:
        print("No arguments provided")