import numpy as np
import matplotlib.pyplot as plt

from rover_map import MapGenerator
from RRG import Graph
from astar import RoverAStar

if __name__ == '__main__':

    graph = Graph()

    world_map = MapGenerator()
    world_map.get_mars_map()
    world_map.select_location((590, 690), (100, 200))

    plt.imshow(world_map.map)
    plt.show()

    num_vertices = 1000
    radius = 10
    path_lengths = [[], []]
    error_covs = [[], []]

    graph.generate_RRG(0, world_map.x_limit, 0, world_map.y_limit, num_vertices, radius)

    start = tuple(np.random.uniform(low=0, high=world_map.x_limit * 0.1, size=2))
    end = tuple(np.random.uniform(low=world_map.y_limit * 0.9, high=world_map.y_limit, size=2))

    astar_error = RoverAStar(graph, world_map, start=start, goal=end, cost_type="error", alpha=10)
    astar_dist  = RoverAStar(graph, world_map, start=start, goal=end, cost_type="distance", alpha=None)
    error_node = astar_error.run()
    dist_node = astar_dist.run()
    nodes = [error_node, dist_node]
    colors = ["r", "b"]

    plt.imshow(world_map.map)
    for index, (node, color) in enumerate(zip(nodes, colors)):

        error_covs[index].append(np.trace(node.P))

        path = []
        while node != None:
            P = node.P
            path.append(node.vertex)
            node = node.parent

        dist = 0
        path_x = []
        path_y = []
        for i in range(len(path) - 1):

            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            dist += np.linalg.norm((dx, dy))
            path_x.append(path[i][0])
            path_y.append(path[i][1])

        path_x.append(path[-1][0])
        path_y.append(path[-1][1])
        if index == 0:
            plt.plot(path_x, path_y, 'ro-', label='EPA*')
        elif index == 1:
            plt.plot(path_x, path_y, 'bo-', label='A*')

        path_lengths[index].append(dist)

    print(error_covs, path_lengths)

    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend()

    plt.show()
