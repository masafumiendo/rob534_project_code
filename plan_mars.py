import numpy as np
import matplotlib.pyplot as plt

from rover_map import MapGenerator
from RRG import Graph
from astar import RoverAStar

if __name__ == '__main__':

    graph = Graph()

    world_map = MapGenerator()
    world_map.get_mars_map()

    plt.imshow(world_map.map)
    plt.show()

    num_vertices = 250
    radius = 10
    path_lengths = [[], []]
    error_covs = [[], []]

    graph.generate_RRG(0, world_map.x_limit, 0, world_map.y_limit, num_vertices, radius)

    start = tuple(np.random.uniform(low=0, high=world_map.x_limit * 0.1, size=2))
    end = tuple(np.random.uniform(low=world_map.y_limit * 0.9, high=world_map.y_limit, size=2))

    astar_error = RoverAStar(graph, world_map, start=start, goal=end, cost_type="error")
    astar_dist  = RoverAStar(graph, world_map, start=start, goal=end, cost_type="distance")
    error_node = astar_error.run()
    dist_node = astar_dist.run()
    nodes = [error_node, dist_node]
    colors = ["r", "b"]

    plt.imshow(world_map.map)
    for node, color in zip(nodes, colors):
        if color == "r":
            index = 0
        else:
            index = 1

        error_covs[index].append(np.trace(node.P))

        path = []
        while node != None:
            P = node.P
            path.append(node.vertex)
            node = node.parent

        dist = 0
        for i in range(len(path) - 1):

            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            dist += np.linalg.norm((dx, dy))

            plt.plot((path[i][0], path[i+1][0]), (path[i][1], path[i+1][1]), '%so-' % color)

        path_lengths[index].append(dist)

    plt.show()
