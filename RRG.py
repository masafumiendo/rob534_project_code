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
        
        self.radius = None
        
    def generate_RRG(self, x_min, x_max, y_min, y_max, total_points, radius, starting_point = None):
        
        self.radius = radius
        
        if starting_point is None:
            starting_point = ((x_min + x_max) / 2, (y_min + y_max / 2))
        
        self.vertices.append(starting_point)
        current_points = 1
        
        while current_points < total_points:
            
            # generate new point
            new_x = np.random.uniform(low=x_min, high=x_max)
            new_y = np.random.uniform(low=y_min, high=y_max)
            
            new_point = (new_x, new_y)
            
            self.add_vertex(new_point, self.radius)
            current_points += 1
        
    def add_vertex(self, new_point, radius):
        
        # make updated KD Tree
        kdtree = KDTree(self.vertices)
        
        # find closest point to new point
        _, nearest_index = kdtree.query(new_point)
            
        nearest = self.vertices[nearest_index]
        
        # add new point and edge to graph
        self.vertices.append(new_point)
        self.edges.append(Edge(new_point, nearest))
        
        # see if any extra connections are within distance
        close_point_indices = kdtree.query_ball_point(new_point, radius)
        
        # avoid double adding of connection to nearest
        try:
            close_point_indices.remove(nearest_index)
        except ValueError:
            pass
        
        # add near connections
        for close_point_index in close_point_indices:
            self.edges.append(Edge(new_point, self.vertices[close_point_index]))
            
    def get_edges(self, vertex):
        
        edges = []
        for edge in self.edges:
            
            if edge.vertex_a == vertex or edge.vertex_b == vertex:
                edges.append(edge)
                
        return edges
    
    def plot(self, display=True):
        
        for i in range(len(self.edges)):
            x = [self.edges[i].vertex_a[0], self.edges[i].vertex_b[0]]
            y = [self.edges[i].vertex_a[1], self.edges[i].vertex_b[1]]
            
            plt.plot(x, y, 'ko-')
            
        if display:
            plt.show()
    
if __name__ == "__main__":
    graph = Graph()
    
    graph.generate_RRG(0, 1, 0, 1, 250, 0.12)
    
    graph.plot()