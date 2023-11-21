import math
from random import randint, choice
from dataclasses import dataclass

@dataclass
class Point2D:
    x: float
    y: float

    def distance(self, other: "Point2D") -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


def intra_clusters_distance(
    points: list[Point2D], cluster_assignments: list[int]
) -> float:
    """
    Calculate the cumulative intra-cluster distance for a list of points given their cluster assignments.

    Args:
    points (list[Point2D]): A list of Point2D objects representing the points.
    cluster_assignments (list[int]): A list of integers where each integer represents the cluster assignment for the corresponding point in 'points'.

    Returns:
    float: The cumulative distance between each pair of points within the same cluster.
    """
    cummulative_intra_clusters_distance = 0.0

    # go through all pairs of points
    for i in range(len(cluster_assignments)):
        for j in range(i + 1, len(cluster_assignments)):

            # if they are in the same cluster, add their distance to the cummulative distance
            if cluster_assignments[i] == cluster_assignments[j]:
                cummulative_intra_clusters_distance += points[i].distance(points[j])

    return cummulative_intra_clusters_distance


@dataclass
class ClusteringSolution:
    """
    A data class representing a clustering solution.

    Attributes:
    assignments (list[int]): A list of integers where each integer represents the cluster assignment of a point.
    quality (float): A float value representing the quality of the clustering solution, typically a measure of intra-cluster distance.

    Methods:
    copy: Returns a deep copy of the ClusteringSolution instance.
    create: Creates a new ClusteringSolution instance given point assignments and points.
    create_random: Generates a ClusteringSolution with random cluster assignments.
    clone_with_change: Creates a new ClusteringSolution instance by altering a single assignment in the existing solution.
    """

    assignments: list[int]
    quality: float

    def copy(self) -> "ClusteringSolution":
        return ClusteringSolution(self.assignments.copy(), self.quality)

    @classmethod
    def create(
        cls, assignments: list[int], points: list[Point2D]
    ) -> "ClusteringSolution":
        return ClusteringSolution(
            assignments.copy(), -intra_clusters_distance(points, assignments)
        )

    @classmethod
    def create_random(
        cls, points: list[Point2D], number_of_clusters: int
    ) -> "ClusteringSolution":
        if number_of_clusters < 1:
            raise ValueError("Number of clusters must be at least 1")
        return ClusteringSolution.create(
            [randint(0, number_of_clusters - 1) for _ in range(len(points))], points
        )

    @classmethod
    def clone_with_change(
        cls,
        solution: "ClusteringSolution",
        assignment_i: int,
        cluster: int,
        points: list[Point2D],
    ) -> "ClusteringSolution":
        assignments_ = solution.assignments.copy()
        assignments_[assignment_i] = cluster
        return ClusteringSolution.create(assignments_, points)


def random_index(list_length: int, split_index: int, top: bool = True) -> int:
    """
    Generate a random index within a specified range of a list.

    Args:
    list_length (int): The total length of the list.
    split_index (int): The index that divides the list into two parts.
    top (bool): If True, generate index in the top part (before split_index). If False, generate index in the bottom part (after split_index).

    Returns:
    int: A random index within the specified range.

    Raises:
    ValueError: If split_index is out of the valid range.
    """
    if split_index < 1 or split_index + 1 > list_length - 2:
        raise ValueError("Split index is out of bounds")

    if top:
        return randint(0, split_index)
    else:
        return randint(split_index + 1, list_length - 1)


def crossover(
    solution1: ClusteringSolution,
    solution2: ClusteringSolution,
    cluster_index_permutations: list[tuple],
    points: list[Point2D],
) -> ClusteringSolution:
    """
    Perform a crossover operation between two clustering solutions.

    Args:
    solution1 (ClusteringSolution): The first clustering solution.
    solution2 (ClusteringSolution): The second clustering solution.
    cluster_index_permutations (list[tuple]): A list of tuples representing all possible permutations of cluster indices.
    points (list[Point2D]): A list of Point2D objects representing the points.

    Returns:
    ClusteringSolution: A new ClusteringSolution that is a result of the crossover between the two input solutions.
    """
    # Map solution2 with each possible permutation and compute similarity with solution1
    solution2_representations = [
        (
            i,
            sum(
                1
                for j, solution1_assignment in enumerate(solution1.assignments)
                if solution1_assignment
                == [
                    permutation[solution2_assignment]
                    for solution2_assignment in solution2.assignments
                ][j]
            ),
        )
        for i, permutation in enumerate(cluster_index_permutations)
    ]

    # Find the the most similar representation
    best_i_, _ = max(solution2_representations, key=lambda x: x[1])

    # Create a new solution from solution1 and the permuted solution2 with a random choice for each assignment
    assignments = [
        cluster_index_permutations[best_i_][solution2_assignment]
        if choice([0, 1]) == 0
        else solution1.assignments[i]
        for i, solution2_assignment in enumerate(solution2.assignments)
    ]

    return ClusteringSolution.create(assignments, points)


def mutation(
    solution: ClusteringSolution,
    mutation_rate: int,
    number_of_clusters: int,
    points: list[Point2D],
) -> ClusteringSolution:
    """
    Apply mutation to a clustering solution.

    Args:
    solution (ClusteringSolution): The clustering solution to mutate.
    mutation_rate (int): The number of mutations to apply.
    number_of_clusters (int): The total number of clusters.
    points (list[Point2D]): A list of Point2D objects representing the points.

    Returns:
    ClusteringSolution: The mutated clustering solution.
    """
    solution_ = solution.copy()

    for _ in range(mutation_rate):

        # Choose a random point assignment
        assignment_i = choice(range(len(solution_.assignments)))

        # Test all possible cluster assignments for the point
        for cluster in range(number_of_clusters):

            # Try to assign a point to another cluster
            mutated_solution = ClusteringSolution.clone_with_change(
                solution_, assignment_i, cluster, points
            )

            if mutated_solution.quality > solution_.quality:
                solution_ = mutated_solution

    return solution_