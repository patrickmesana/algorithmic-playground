from genetic_clustering import ClusteringSolution, Point2D, random_index, crossover, mutation
from data import small_dataset_0, generate_large_dataset, plot_clusters
from itertools import permutations

def parallel_tasks_2_execution_main(
    dataset: list[tuple], number_of_solutions: int, generations: int, 
    mutation_rate: int, number_of_clusters: int, concurrent_tasks: int
):
    """
    Runs the genetic algorithm in parallel with Ray, managing tasks dynamically for asynchronous processing.

    Args:
    dataset (list[tuple]): Dataset to use, each tuple representing a point (x, y).
    number_of_solutions (int): Number of solutions in the population.
    generations (int): Number of generations for the genetic algorithm.
    mutation_rate (int): Mutation rate for the genetic algorithm.
    number_of_clusters (int): Number of clusters to form.
    concurrent_tasks (int): Number of tasks to run in parallel at any given time.
    """
    import ray
    ray.init()

    def start_new_task(population, clusters_permutations, points, mutation_rate, number_of_clusters):
        good_solution, bad_solution = select_solutions(population)
        return process_solution.remote(
            good_solution, bad_solution, clusters_permutations, points, mutation_rate, number_of_clusters
        )

    def select_solutions(population):
        good_solution = population[random_index(len(population), len(population) // 2, top=True)]
        bad_solution = population[random_index(len(population), len(population) // 2, top=False)]
        return good_solution, bad_solution

    def update_population(population, child_solution):
        random_bad_solution_idx = random_index(len(population), len(population) // 2, top=False)
        population[random_bad_solution_idx] = child_solution
        population.sort(key=lambda s: s.quality, reverse=True)
    


    @ray.remote
    def process_solution(good_solution, bad_solution, clusters_permutations, points, mutation_rate, number_of_clusters):
        child_solution = crossover(
            good_solution, bad_solution, clusters_permutations, points
        )
        return mutation(
            child_solution, mutation_rate, number_of_clusters, points
        )

    points = [Point2D(x, y) for x, y in dataset]
    clusters_permutations: list[tuple] = list(permutations(list(range(number_of_clusters))))

    population: list[ClusteringSolution] = sorted(
        [ClusteringSolution.create_random(points, number_of_clusters) for _ in range(number_of_solutions)],
        key=lambda s: s.quality,
        reverse=True,
    )

    tasks = []
    generations_processed = 0

    # Start initial tasks
    for _ in range(concurrent_tasks):
        tasks.append(start_new_task(population, clusters_permutations, points, mutation_rate, number_of_clusters))

    while generations_processed < generations:
        done_ids, tasks = ray.wait(tasks, num_returns=1)
        completed_task = ray.get(done_ids[0])
        update_population(population, completed_task)
        generations_processed += 1

        if generations_processed < generations:
            tasks.append(start_new_task(population, clusters_permutations, points, mutation_rate, number_of_clusters))

        print("Solution value for iteration {}: {}".format(generations_processed, population[0].quality))

    print("Final solution value : ", population[0].quality)
    ray.shutdown()

    return points, population[0], number_of_clusters


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

            
            points, solution, number_of_clusters = parallel_tasks_1_execution_main(
                small_dataset_0, number_of_solutions, generations, mutation_rate, number_of_clusters, number_of_tasks
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

            points, solution, number_of_clusters = parallel_tasks_1_execution_main(
                large_dataset, number_of_solutions, generations, mutation_rate, number_of_clusters, number_of_tasks
            )
            plot_clusters(points, solution.assignments, number_of_clusters)
        else:
            print("Invalid argument")
    else:
        print("No arguments provided")
