"""
author: Masafumi Endo
objective: implement map class
"""

import random
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

class MapGenerator:

    def __init__(self, x_limit=100, y_limit=100, num_circles=3, rad_circles=15):
        self.x_limit = x_limit
        self.y_limit = y_limit
        self.num_circles = num_circles
        self.rad_circles = rad_circles
        # init map w/ all feature-poor region (expressed as "0")
        self.map = np.zeros((self.y_limit, self.x_limit))
        self.fig = None

    def get_map(self):

        # put feature-rich circles randomly
        for i in range(self.num_circles):
            # generate center of circle
            xc = random.randint(self.rad_circles, self.x_limit - self.rad_circles)
            yc = random.randint(self.rad_circles, self.y_limit - self.rad_circles)
            x, y = np.meshgrid(np.arange(self.x_limit), np.arange(self.y_limit))
            d2 = (x - xc)**2 + (y - yc)**2
            mask = d2 < self.rad_circles**2
            self.map[mask] = 1

        return self.map

    def reset_map(self):
        self.map = np.zeros((self.y_limit, self.x_limit))
        return self.get_map()

    def show_map(self, path=None):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.map)
        if path is not None:
            self.ax.plot(path[:, 0], path[:, 1], color='red')
        plt.show(block=False)
        plt.pause(0.005)

if __name__ == '__main__':
    # test each method
    mg = MapGenerator()
    map = mg.get_map()
    mg.show_map()

    # sample to visualize randomly generate path
    path = np.array([[0, 0]])
    for i in range(20):
        x = random.randint(0, 100)
        y = random.randint(0, 100)
        print('x: {}, y: {}'.format(x, y))
        vertex = np.array([[x, y]])
        path = np.append(path, vertex, axis=0)
        mg.show_map(path)