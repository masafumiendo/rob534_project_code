import math
import priority_queue
import numpy as np
import error_propagation

def euclidianDist(a, b, unused):
    
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def errorProp(a, b, P, alpha=10):
    
    dist = euclidianDist(a, b, None)
    return dist + alpha * np.trace(P)

def errorPropMin(a, b, dsig_min):
    
    dist = euclidianDist(a, b, None)
    return dist * (1 + np.trace(dsig_min))

class Node:
    
    def __init__(self, vertex, cost, parent, P):
        
        self.vertex = vertex
        self.cost = cost
        self.parent = parent
        self.P = P
        
class RoverAStar:
    
    def __init__(self, graph, world_map, start, goal, cost_type):
        
        self.heuristic = None
        self.error_calc = None
        self.using_error = cost_type == "error"
        if not self.using_error:
            self.cost = euclidianDist
        else:
            self.cost = errorProp
            self.error_calc = error_propagation.ErrorCalculator(world_map, np.diag([4e-4, 4e-4, 3e-7]), np.diag([1e-2, 1e-2, 3e-7]))
            
        self.graph = graph
        if start not in graph.vertices:
            self.graph.add_vertex(start, graph.radius)
        if goal not in graph.vertices:
            self.graph.add_vertex(goal, graph.radius)
        if self.using_error: 
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
            return euclidianDist(self.goal, location, None)
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
                    if self.using_error:
                        A = edge.data.A_ij
                        B_dSIG_BT = edge.data.B_dSIG_BT_ij
                else:
                    neighbor_loc = edge.vertex_a
                    if self.using_error:
                        A = edge.data.A_ji
                        B_dSIG_BT = edge.data.B_dSIG_BT_ji
                    
                if neighbor_loc not in self.closed:
                    if not self.using_error:
                        P = None
                    else:
                        P = A * node.P * A.T + B_dSIG_BT
                        
                    if not self.pq.test(neighbor_loc):
                        neighbor_node = Node(neighbor_loc, node.cost + self.cost(vertex, neighbor_loc, P), node, P)
                        self.pq.insert(neighbor_loc, neighbor_node.cost + self.getHeuristic(neighbor_loc))
                        self.open[neighbor_loc] = neighbor_node
                    else:
                        neighbor_node = self.open[neighbor_loc]
                        new_cost = node.cost + self.cost(vertex, neighbor_loc, P)
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
    # world_map.get_mars_map()
    world_map.get_random_map()
    
    from RRG import Graph
    graph = Graph()
    graph.generate_RRG(0, world_map.x_limit, 0, world_map.y_limit, 2000, 10)
    
    astar_error = RoverAStar(graph, world_map, start=(1, 1), goal=(99, 99), cost_type="error")
    astar_dist  = RoverAStar(graph, world_map, start=(1, 1), goal=(99, 99), cost_type="distance")
    error_node = astar_error.run()
    dist_node = astar_dist.run()
    nodes = [error_node, dist_node]
    colors = ["r", "b"]
    # graph.plot(display=False)
    for node, color in zip(nodes, colors):
        path = []
        while node != None:
            
            path.append(node.vertex)
            node = node.parent
        
        import matplotlib.pyplot as plt
        
        for i in range(len(path) - 1):
            
            plt.plot((path[i][0], path[i+1][0]), (path[i][1], path[i+1][1]), '%so-' % color)
    
    plt.imshow(world_map.map)
    plt.show()