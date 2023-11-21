import matplotlib.pyplot as plt

def plot_clusters(points, cluster_assignments, number_of_clusters):
    """
    Plot the points and color them based on their cluster assignments.

    Args:
    points (list[Point2D]): List of Point2D objects representing the points.
    cluster_assignments (list[int]): List of cluster assignments for each point.
    number_of_clusters (int): Total number of clusters.
    """
    colors = plt.cm.rainbow(np.linspace(0, 1, number_of_clusters))
    
    for cluster_index in range(number_of_clusters):
        cluster_points = [point for point, assignment in zip(points, cluster_assignments) if assignment == cluster_index]
        plt.scatter([p.x for p in cluster_points], [p.y for p in cluster_points], color=colors[cluster_index], label=f'Cluster {cluster_index}')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Cluster Visualization')
    plt.legend()
    plt.show()

import numpy as np

def generate_large_dataset(num_centers=10, min_points=50, max_points=100):
    """
    Generate a larger dataset with specified number of centers and points per center.

    Args:
    num_centers (int): Number of centers to generate.
    min_points (int): Minimum number of points per center.
    max_points (int): Maximum number of points per center.

    Returns:
    list[tuple]: A list of tuples representing points.
    """
    # Generate random centers
    centers = [tuple(np.random.uniform(0, 100, 2)) for _ in range(num_centers)]

    # Generate points for each center
    return [
        tuple(np.random.normal(center, scale=1.0, size=2))
        for center in centers
        for _ in range(np.random.randint(min_points, max_points))
    ]

small_dataset_0 = [
    (5.0, 3.0),
    (2.0, 4.0),
    (0.0, 1.0),
    (8.0, 6.0),
    (7.0, 3.0),
    (2.0, 8.0),
    (1.0, 6.0),
    (2.0, 12.0),
    (4.0, 10.0),
    (5.0, 13.0),
    (7.0, 2.0),
    (15.0, 3.0),
    (18.0, 20.0),
    (20.0, 16.0),
    (13.0, 6.0),
    (9.0, 13.0),
    (4.0, 9.0),
    (1.0, 4.0),
    (0.0, 7.0),
    (6.0, 3.0),
]