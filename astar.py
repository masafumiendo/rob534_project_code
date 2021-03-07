import math
import priority_queue

def euclidianDist(a, b):
    
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def errorProp(a, b):
    
    return 0

def errorPropMin(a, b):
    
    return 0

class Node:
    
    def __init__(self, vertex, cost, parent):
        
        self.vertex = vertex
        self.cost = cost
        self.parent = parent
        
class RoverAStar:
    
    def __init__(self, graph, start, goal, cost_type):
        
        self.heuristic = None
        if cost_type == "distance":
            self.heuristic = euclidianDist
            self.cost = euclidianDist
        elif cost_type == "error":
            self.heuristic = errorPropMin
            self.cost = errorProp
        else:
            print('Invalid cost type. Must be "distance" or "error". Got %s' % cost_type)
            exit(-1)
            
        self.graph = graph
        self.graph.add_vertex(start, graph.radius)
        self.graph.add_vertex(goal, graph.radius)
        self.start = start
        self.goal = goal
        self.pq = None
        self.open = None
        self.closed = None
        
    def reset(self):
        self.pq = priority_queue.PriorityQueue()
        self.open = dict()
        self.closed = dict()
        
        self.open[self.start] = Node(self.start, 0, None)
        self.pq.insert(self.start, self.getHeuristic(self.start))
                       
    def getHeuristic(self, location):

        return self.heuristic(self.goal, location)
        
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
                else:
                    neighbor_loc = edge.vertex_a
                    
                if neighbor_loc not in self.closed:
                    if not self.pq.test(neighbor_loc):
                        neighbor_node = Node(neighbor_loc, node.cost + self.cost(vertex, neighbor_loc), node)
                        self.pq.insert(neighbor_loc, neighbor_node.cost + self.getHeuristic(neighbor_loc))
                        self.open[neighbor_loc] = neighbor_node
                    else:
                        neighbor_node = self.open[neighbor_loc]
                        if node.cost + self.cost(vertex, neighbor_loc) < neighbor_node.cost:
                            neighbor_node.cost = node.cost + self.cost(vertex, neighbor_loc)
                            neighbor_node.parent = node
                            self.pq.insert(neighbor_loc, neighbor_node.cost + self.getHeuristic(neighbor_loc))
                            
        print("PQ is empty")
        return None
    
if __name__ == "__main__":
    
    from RRG import Graph
    graph = Graph()
    graph.generate_RRG(0, 1, 0, 1, 500, 0.1)
    
    astar = RoverAStar(graph, start=(0, 0), goal=(1, 1), cost_type="distance")
    
    node = astar.run()
    
    path = []
    while node != None:
        
        path.append(node.vertex)
        node = node.parent
        
    import matplotlib.pyplot as plt
    graph.plot(display=False)
    for i in range(len(path) - 1):
        
        plt.plot((path[i][0], path[i+1][0]), (path[i][1], path[i+1][1]), 'ro-')
        
    plt.show()