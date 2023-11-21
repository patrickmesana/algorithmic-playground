from genetic_clustering import ClusteringSolution, Point2D, random_index, crossover, mutation
from data import small_dataset_0, generate_large_dataset, plot_clusters
from itertools import permutations

def parallel_actors_execution_main(
    dataset: list[tuple], number_of_solutions: int, generations: int, 
    mutation_rate: int, number_of_clusters: int, number_of_workers: int
):
    import ray
    import time

    @ray.remote
    class Coordinator:
        def __init__(self, number_of_solutions, number_of_clusters, mutation_rate, clusters_permutations, points, generations):
            self.population = sorted(
                [ClusteringSolution.create_random(points, number_of_clusters) for _ in range(number_of_solutions)],
                key=lambda s: s.quality, reverse=True
            )
            self.mutation_rate = mutation_rate
            self.clusters_permutations = clusters_permutations
            self.points = points
            self.generations = generations
            self.current_generation = 0
            self.work_completed = 0

        def get_work(self):
            if self.current_generation < self.generations:
                good_solution, bad_solution = self.select_solutions()
                self.current_generation += 1
                return good_solution, bad_solution
            else:
                return None, None

        def update_population(self, child_solution):
            random_bad_solution_idx = random_index(len(self.population), len(self.population) // 2, top=False)
            self.population[random_bad_solution_idx] = child_solution
            self.population.sort(key=lambda s: s.quality, reverse=True)
            self.work_completed += 1

        def select_solutions(self):
            good_solution = self.population[random_index(len(self.population), len(self.population) // 2, top=True)]
            bad_solution = self.population[random_index(len(self.population), len(self.population) // 2, top=False)]
            return good_solution, bad_solution

        def get_best_solution(self):
            return self.population[0]

        def is_work_done(self):
            return self.work_completed >= self.generations
        
    @ray.remote
    class Worker:
        def __init__(self, coordinator):
            self.coordinator = coordinator

        def work(self):
            while True:
                good_solution, bad_solution = ray.get(self.coordinator.get_work.remote())
                if good_solution is None:
                    break
                child_solution = self.genetic_process(good_solution, bad_solution)
                self.coordinator.update_population.remote(child_solution)

        def genetic_process(self, good_solution, bad_solution):
            child_solution = crossover(
                good_solution, bad_solution, clusters_permutations, points
            )
            return mutation(
                child_solution, mutation_rate, number_of_clusters, points
            )

    

    points = [Point2D(x, y) for x, y in dataset]
    clusters_permutations: list[tuple] = list(permutations(list(range(number_of_clusters))))

    coordinator = Coordinator.remote(
        number_of_solutions, number_of_clusters, mutation_rate, clusters_permutations, points, generations
    )

    workers = [Worker.remote(coordinator) for _ in range(number_of_workers)]
    [worker.work.remote() for worker in workers]

    # Wait for the Coordinator to indicate that all work is done
    while not ray.get(coordinator.is_work_done.remote()):
        print("Solution value : {}".format(
                ray.get(coordinator.get_best_solution.remote()).quality
            )
        )
        time.sleep(0.1)
    
    print("Final solution value : ", ray.get(coordinator.get_best_solution.remote()).quality)

    best_solution = ray.get(coordinator.get_best_solution.remote())
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

            
            points, solution, number_of_clusters = parallel_actors_execution_main(
                small_dataset_0, number_of_solutions, generations * number_of_tasks, mutation_rate, number_of_clusters, number_of_tasks
            )
            plot_clusters(points, solution.assignments, number_of_clusters)
        elif first_arg == "random":
            number_of_clusters = 5

            large_dataset = generate_large_dataset(num_centers=number_of_clusters, min_points=2, max_points=10)

            # print size of dataset
            print('dataset size', len(large_dataset))


            number_of_solutions = 50
            generations = 200
            mutation_rate = 5

            number_of_tasks = 8

            points, solution, number_of_clusters = parallel_actors_execution_main(
                large_dataset, number_of_solutions, generations * number_of_tasks, mutation_rate, number_of_clusters, number_of_tasks
            )
            plot_clusters(points, solution.assignments, number_of_clusters)
        else:
            print("Invalid argument")
    else:
        print("No arguments provided")