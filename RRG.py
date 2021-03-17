import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import math

class Edge:
    
    def __init__(self, vertex_a, vertex_b, data=None):
        
        self.vertex_a = vertex_a
        self.vertex_b = vertex_b
        
        self.data = data

class Graph:
    
    def __init__(self):
        
        self.reset()
        
    def reset(self):
        self.vertices = []
        self.edges = []
        
        self.radius = None
        
    def generate_RRG(self, x_min, x_max, y_min, y_max, total_points, radius, starting_point = None):
        
        self.reset()
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        area = x_range * y_range
        gamma = 2 * (1 + 1/2) ** (1/2) * (area / math.pi) ** (1/2)
        
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
            
            self.radius = min(radius, gamma * (math.log(current_points)/current_points) ** (1/2))
        
    def add_vertex(self, new_point, radius):
               
        # make updated KD Tree
        kdtree = KDTree(self.vertices)
        
        # find closest point to new point
        nearest_point_dist, nearest_index = kdtree.query(new_point)
        
        nearest_edge_dist = 0
        nearest_edge = None
        for edge in self.edges:
            
            ab = np.array(edge.vertex_b) - np.array(edge.vertex_a)
            ap = np.array(new_point) - np.array(edge.vertex_a)
            cp = ap - (np.dot(ap, ab) / (np.linalg.norm(ab) ** 2)) * ab
            ac = ap - cp
            bc = -ab + ac
            
            if np.linalg.norm(bc) >= np.linalg.norm(ab) or np.linalg.norm(ac) >= np.linalg.norm(ab):
                continue
            
            if np.linalg.norm(cp) < nearest_edge_dist or nearest_edge is None:
                nearest_edge_dist = np.linalg.norm(cp)
                nearest_edge = edge
        
        if nearest_point_dist <= nearest_edge_dist or nearest_edge is None:
            nearest = self.vertices[nearest_index]
        else:
            ab = np.array(nearest_edge.vertex_b) - np.array(nearest_edge.vertex_a)
            ap = np.array(new_point) - np.array(nearest_edge.vertex_a)
            cp = ap - (np.dot(ap, ab) / (np.linalg.norm(ab) ** 2)) * ab
            ac = ap - cp
            
            vertex_c = (nearest_edge.vertex_a[0] + ac[0], nearest_edge.vertex_a[1] + ac[1])
            self.vertices.append(vertex_c)
            nearest = vertex_c
            
            self.edges.append(Edge(nearest_edge.vertex_b, vertex_c))
            nearest_edge.vertex_b = vertex_c
        
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
    
    graph.generate_RRG(0, 1, 0, 1, 500, 0.12)
    
    graph.plot()