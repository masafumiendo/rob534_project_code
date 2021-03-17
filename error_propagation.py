import numpy as np
import RRG
import math


class ErrorMatrices:
    def __init__(self, A_ij, B_dSIG_BT_ij, A_ji, B_dSIG_BT_ji):
        self.A_ij = A_ij
        self.B_dSIG_BT_ij = B_dSIG_BT_ij
        self.A_ji = A_ji
        self.B_dSIG_BT_ji = B_dSIG_BT_ji


class ErrorCalculator:

    def __init__(self, world_map, vo_sig=None, wo_sig=None):

        self.map = world_map
            
        if vo_sig is None:
            self.vo_sig = np.diag([4e-4, 4e-4, 3e-7])
        else:
            self.vo_sig = vo_sig
        
        if wo_sig is None:
            self.wo_sig = np.diag([1e-2, 1e-2, 3e-7])
        else:
            self.wo_sig = wo_sig

    # The coefficient matrices
    def calculate_matrices(self, edges):
        for edge in edges:
            dist = np.linalg.norm(np.array(edge.vertex_a) - np.array(edge.vertex_b))
            delta_x_ij = dist
            # delta_y_ij = 0
            delta_x_ji = dist
            # delta_y_ji = 0

            theta_ij = math.atan2(edge.vertex_b[1] - edge.vertex_a[1], edge.vertex_b[0] - edge.vertex_a[0])
            theta_ji = math.atan2(edge.vertex_a[1] - edge.vertex_b[1], edge.vertex_a[0] - edge.vertex_b[0])
            
            if self.map.is_feature_rich(edge.vertex_a) and self.map.is_feature_rich(edge.vertex_b):
                dsig = self.vo_sig
            else:
                dsig = self.wo_sig
                
            dSIG = dist * dsig

            A_ij = np.array([[1, 0, -delta_x_ij * np.sin(theta_ij)],
                             [0, 1, (delta_x_ij * np.cos(theta_ij))],
                             [0, 0, 1]])
            B_ij = np.array([[np.cos(theta_ij), -np.sin(theta_ij), 0],
                             [np.sin(theta_ij), np.cos(theta_ij), 0],
                             [0, 0, 1]])
            
            B_dSIG_BT_ij = B_ij * dSIG * B_ij.T

            A_ji = np.array([[1, 0, -delta_x_ji * np.sin(theta_ji)],
                             [0, 1, delta_x_ji * np.cos(theta_ji)],
                             [0, 0, 1]])
            B_ji = np.array([[np.cos(theta_ji), -np.sin(theta_ji), 0],
                             [np.sin(theta_ji), np.cos(theta_ji), 0],
                             [0, 0, 1]])
            
            B_dSIG_BT_ji = B_ji * dSIG * B_ji.T

            edge.data = ErrorMatrices(A_ij, B_dSIG_BT_ij, A_ji, B_dSIG_BT_ji)


if __name__ == '__main__':
    import rover_map
    
    feature_map = rover_map.MapGenerator()
    feature_map.get_mars_map()

    graph = RRG.Graph()
    graph.generate_RRG(0, feature_map.x_limit, 0, feature_map.y_limit, 250, 0.15)
    
    ep = ErrorCalculator(feature_map)
    ep.calculate_matrices(graph.edges)

    print(graph.edges[0].vertex_a, graph.edges[0].vertex_b)
    print(graph.edges[0].data.A_ij)