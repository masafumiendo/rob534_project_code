import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

class Edge:
    
    def __init__(self, vertex_a, vertex_b, data=None):
        
        self.vertex_a = vertex_a
        self.vertex_b = vertex_b
        
        self.data = data

class Graph:
    
    def __init__(self):
        
        self.vertices = []
        self.edges = []
        
def RRG(x_min, x_max, y_min, y_max, total_points, radius, starting_point = None):
    
    if starting_point is None:
        starting_point = ((x_min + x_max) / 2, (y_min + y_max / 2))
    
    graph = Graph()
    graph.vertices.append(starting_point)
    current_points = 1
    
    while current_points < total_points:
        # make updated KD Tree
        kdtree = KDTree(graph.vertices)
        
        # generate new point
        new_x = np.random.uniform(low=x_min, high=x_max)
        new_y = np.random.uniform(low=y_min, high=y_max)
        
        new_point = (new_x, new_y)
        
        # find closest point to new point
        _, nearest_index = kdtree.query(new_point)
        
        nearest = graph.vertices[nearest_index]
        
        # add new point and edge to graph
        graph.vertices.append(new_point)
        graph.edges.append(Edge(new_point, nearest))
        current_points += 1
        
        # see if any extra connections are within distance
        close_point_indices = kdtree.query_ball_point(new_point, radius)
        
        # avoid double adding of connection to nearest
        try:
            close_point_indices.remove(nearest_index)
        except ValueError:
            pass
        
        # add near connections
        for close_point_index in close_point_indices:
            graph.edges.append(Edge(new_point, graph.vertices[close_point_index]))
            
    return graph
        
    
if __name__ == "__main__":
    graph = RRG(0, 1, 0, 1, 250, 0.15)
    
    for i in range(len(graph.edges)):
        x = [graph.edges[i].vertex_a[0], graph.edges[i].vertex_b[0]]
        y = [graph.edges[i].vertex_a[1], graph.edges[i].vertex_b[1]]
        
        plt.plot(x, y, 'ko-')
        
    plt.show()