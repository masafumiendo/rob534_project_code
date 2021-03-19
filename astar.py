import math
import priority_queue
import numpy as np
import error_propagation

def euclidianDist(a, b, unused1, unused2):
    
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def errorProp(a, b, P, alpha):

    dist = euclidianDist(a, b, None, None)
    return dist + alpha * np.trace(P)

def errorPropMin(a, b, dsig_min):
    
    dist = euclidianDist(a, b, None, None)
    return dist * (1 + np.trace(dsig_min))

class Node:
    
    def __init__(self, vertex, cost, parent, P):
        
        self.vertex = vertex
        self.cost = cost
        self.parent = parent
        self.P = P
        
class RoverAStar:
    
    def __init__(self, graph, world_map, start, goal, cost_type, alpha):
        
        self.heuristic = None
        self.error_calc = None
        self.using_error = cost_type == "error"
        if not self.using_error:
            self.cost = euclidianDist
        else:
            self.cost = errorProp
        self.alpha = alpha
        
        self.error_calc = error_propagation.ErrorCalculator(world_map, np.diag([4e-4, 4e-4, 3e-7]), np.diag([1e-2, 1e-2, 3e-7]))
            
        self.graph = graph
        if start not in graph.vertices:
            self.graph.add_vertex(start, graph.radius)
        if goal not in graph.vertices:
            self.graph.add_vertex(goal, graph.radius)
        
        self.error_calc.calculate_matrices(self.graph.edges)
        self.start = start
        self.goal = goal
        self.pq = None
        self.open = None
        self.closed = None
        
    def reset(self):
        self.pq = priority_queue.PriorityQueue()
        self.open = dict()
        self.closed = dict()
        
        starting_P = np.zeros((3, 3))
        self.open[self.start] = Node(self.start, 0, None, starting_P)
        self.pq.insert(self.start, self.getHeuristic(self.start))
                       
    def getHeuristic(self, location):

        if not self.using_error:
            return euclidianDist(self.goal, location, None, None)
        else:
            return errorPropMin(self.goal, location, self.error_calc.vo_sig)
        
    def run(self):
        
        self.reset()
    
        while len(self.pq) > 0:
            
            vertex = self.pq.pop()
            
            node = self.open[vertex]
            
            del self.open[vertex]
            self.closed[vertex] = node
            
            if node.vertex[0] == self.goal[0] and node.vertex[1] == self.goal[1]:
                return node
            
            edges = self.graph.get_edges(vertex)

            for edge in edges:
                if edge.vertex_a == vertex:
                    neighbor_loc = edge.vertex_b
                    A = edge.data.A_ij
                    B_dSIG_BT = edge.data.B_dSIG_BT_ij
                else:
                    neighbor_loc = edge.vertex_a
                    A = edge.data.A_ji
                    B_dSIG_BT = edge.data.B_dSIG_BT_ji
                    
                if neighbor_loc not in self.closed:

                    P = A * node.P * A.T + B_dSIG_BT
                        
                    if not self.pq.test(neighbor_loc):
                        neighbor_node = Node(neighbor_loc, node.cost + self.cost(vertex, neighbor_loc, P, self.alpha), node, P)
                        self.pq.insert(neighbor_loc, neighbor_node.cost + self.getHeuristic(neighbor_loc))
                        self.open[neighbor_loc] = neighbor_node
                    else:
                        neighbor_node = self.open[neighbor_loc]
                        new_cost = node.cost + self.cost(vertex, neighbor_loc, P, self.alpha)
                        if new_cost  < neighbor_node.cost:
                            neighbor_node.P = P
                            neighbor_node.cost = new_cost
                            neighbor_node.parent = node
                            self.pq.insert(neighbor_loc, neighbor_node.cost + self.getHeuristic(neighbor_loc))
                            
        print("PQ is empty")
        return None
    
if __name__ == "__main__":
    
    import rover_map
    world_map = rover_map.MapGenerator()
    world_map.get_random_map()
    
    from RRG import Graph
    graph = Graph()
    
    import time
    import matplotlib.pyplot as plt
    
    display = True
    num_vertices = 1000
    radius = 10
    num_iterations = 1
    path_lengths = [[], [], [], []]
    error_covs = [[], [], [], []]
    for iteration in range(num_iterations):
        print("Iteration", iteration)
        
        world_map = rover_map.MapGenerator()
        world_map.get_random_map()
        
        graph.generate_RRG(0, world_map.x_limit, 0, world_map.y_limit, num_vertices, radius)
        
        start = tuple(np.random.uniform(low=0, high=world_map.x_limit * 0.1, size=2))
        end = tuple(np.random.uniform(low=world_map.y_limit * 0.9, high=world_map.y_limit, size=2))
        
        astar_error1 = RoverAStar(graph, world_map, start=start, goal=end, cost_type="error", alpha=10)
        # astar_error2 = RoverAStar(graph, world_map, start=start, goal=end, cost_type="error", alpha=100)
        # astar_error3 = RoverAStar(graph, world_map, start=start, goal=end, cost_type="error", alpha=1000)
        astar_dist   = RoverAStar(graph, world_map, start=start, goal=end, cost_type="distance", alpha=None)
        error_node1 = astar_error1.run()
        # error_node2 = astar_error2.run()
        # error_node3 = astar_error3.run()
        dist_node = astar_dist.run()
        nodes = [error_node1, dist_node]
        # nodes = [error_node1, error_node2, error_node3, dist_node]
        colors = ["r", "r", "r", "b"]
        
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
            elif index == 3:
                plt.plot(path_x, path_y, 'ro--', label='alpha=100')
            elif index == 2:
                plt.plot(path_x, path_y, 'ro:', label='alpha=1000')
            elif index == 1:
                plt.plot(path_x, path_y, 'bo-', label='A*')
                
            path_lengths[index].append(dist)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.legend()
        
        if display:
            plt.show()
            
        # plt.savefig("paths_%d.png" % iteration)
        # plt.clf()

        # # print(error_covs, path_lengths)
        # import pickle
        # with open("sample_data_alpha_%dvertices_%.2fradius.pickle" % (num_vertices, radius), "wb") as f:
        #     pickle.dump({"path_lengths" : path_lengths, "error_covs" : error_covs}, f)
        
    