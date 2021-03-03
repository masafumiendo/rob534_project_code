"""
author: Masafumi Endo
objective: implement map class
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class MapGenerator:

    def __init__(self, x_limit=100, y_limit=100, num_circles=3, rad_circles=15):

        self.x_limit = x_limit
        self.y_limit = y_limit
        self.num_circles = num_circles
        self.rad_circles = rad_circles
        # init map w/ all feature-poor region (expressed as "0")
        self.map = np.zeros((self.y_limit, self.x_limit))

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

    def show_map(self):
        plt.imshow(self.map)
        plt.show()

if __name__ == '__main__':
    # test each method
    mg = MapGenerator()
    map = mg.get_map()
    mg.show_map()
    map = mg.reset_map()
    mg.show_map()

