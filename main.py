import numpy as np


class error_propagation:

    def __init__(self, vertex, edges):
        self.vertex = vertex
        self.edges = edges

        # define the true state
        self.x_hat = []
        self.y_hat = []
        self.theta_hat = []

        # define the estimate state
        self.x = []
        self.y = []
        self.theta = []

    # The estimate error propagation
    def eep(self, j, i):
        """
        Calculate the estimate error propagation
        :param j: to_vertex index
        :param i: from_vertex index
        :return:
        """

    # The state estimate error
    def see(self):
        # The state estimation error is the difference between the true state and the estimate state:
        # error = [[e_x],[e_y],[e_theta]] = [[x_hat],[y_hat],[theta_hat]] + [[x],[y],[theta]]

        self.error_from_vertex = self.edges.from_vertex + self.edges.hat.from_vertex
        self.error_to_vertex = self.edges.to_vertex + self.edges.hat.to_vertex
        return self.error_from_vertex, self.error_to_vertex

    # The state estimation error propagation
    def seep(self):
        # d_error = [[d_e_x],[d_e_y],[d_e_theta]] = [[d_x_hat],[d_y_hat],[d_theta_hat]] - [[d_x],[d_y],[d_theta]]
        # self.edges.cost.hat = [[d_x_hat],[d_y_hat],[d_theta_hat]]
        # self.edges.cost = [[d_x],[d_y],[d_theta]]

        self.d_error = self.edges.hat.cost - self.edges.cost

        # Use this to check the value of error
        if self.error_to_vertex == self.error_from_vertex + self.d_error:
            return True
        else:
            return False

    # Transform from Body frame to Global frame
    def transform(self, theta):
        theta = theta * np.pi /180
        # create the rotation matrix 3x3,type = array
        r_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0],[0, 0, 1]])
        return r_matrix

    # The coefficient matrices
    def coefficient_matrix(self):
        self.A_ij = np.array([[1, 0, (-self.edges.cost.hat.x*np.sin(self.edges.to_vertex.hat.theta*np.pi/180) - self.edges.cost.hat.y*np.cos(self.edges.to_vertex.hat.theta*np.pi/180))],
                [0, 1, (self.edges.cost.hat.x*np.cos(self.edges.to_vertex.hat.theta*np.pi/180) - self.edges.cost.hat.y*np.sin(self.edges.to_vertex.hat.theta*np.pi/180))],
                [0, 0, 1]])
        self.B_ij = np.array([[np.cos(self.edges.to_vertex.hat.theta*np.pi/180), -np.sin(self.edges.to_vertex.hat.theta*np.pi/180), 0],
                [np.sin(self.edges.to_vertex.hat.theta*np.pi/180), np.cos(self.edges.to_vertex.hat.theta*np.pi/180), 0],
                [0, 0, 1]])
        return self.A_ij, self.B_ij

    #The covariance matrix
    def cov_matrix(self, initial_cov_g, initial_cov_b):
        # initial_cov_g = P_i in paper
        # P_i = [self.error_from_vertex.x.g, self.error_from_vertex.y.g, self.error_from_vertex.theta].T
        # initial_cov_b = Epsilon in paper
        # Epsilon = [self.d_error.x, self.d_error.y, self.d_error.theta].T
        # return the P_j
        # 3x3 * 3x1 * 3x3
        P_j = self.A_ij * np.cov(initial_cov_g) * self.A_ij.T + self.B_ij * initial_cov_b * self.B_ij.T
        return P_j




if __name__ == '__main__':

