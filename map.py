"""
author: Masafumi Endo
objective: implement map class
"""

import random
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import cv2

class MapGenerator:

    def __init__(self, x_limit=100, y_limit=100, num_circles=3, rad_circles=15):
        self.x_limit = x_limit
        self.y_limit = y_limit
        self.num_circles = num_circles
        self.rad_circles = rad_circles
        # init map w/ all feature-poor region (expressed as "0")
        self.map = np.zeros((self.y_limit, self.x_limit))
        self.fig = None

    def get_random_map(self, x_limit=100, y_limit=100):
        # init limits
        self.x_limit = x_limit
        self.y_limit = y_limit
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

    def get_mars_map(self, show=False):

        # obtain mars map
        map_raw = cv2.imread('fig/hirise_image_raw.png')
        map_raw = cv2.cvtColor(map_raw, cv2.COLOR_BGR2RGB)
        map_gray = cv2.cvtColor(map_raw, cv2.COLOR_RGB2GRAY)
        # binary process
        ret, self.map = cv2.threshold(map_gray, 0, 1, cv2.THRESH_OTSU)
        if show:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.imshow(map_raw)
            ax2.imshow(self.map)
            plt.show()

        # reset limits
        self.x_limit = self.map.shape[1]
        self.y_limit = self.map.shape[0]
        return self.map

    def reset_random_map(self):
        self.map = np.zeros((self.y_limit, self.x_limit))
        return self.get_random_map()

    def show_map(self, path=None):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            self.fig_list = []
        self.ax.imshow(self.map)
        if path is not None:
            fig = self.ax.plot(path[:, 0], path[:, 1], color='red')
            self.fig_list.append(fig)
        plt.show(block=False)
        plt.pause(0.005)
        
    def generate_animation(self, name):
        ani = ArtistAnimation(self.fig, self.fig_list)
        ani.save(name, writer='pillow')

if __name__ == '__main__':
    # test each method
    mg = MapGenerator()
    map = mg.get_mars_map()
    mg.show_map()

    # sample to visualize randomly generate path
    path = np.array([[0, 0]])
    for i in range(20):
        x = random.randint(0, mg.x_limit)
        y = random.randint(0, mg.y_limit)
        print('x: {}, y: {}'.format(x, y))
        vertex = np.array([[x, y]])
        path = np.append(path, vertex, axis=0)
        mg.show_map(path)
    mg.generate_animation('fig/random_planner.gif')