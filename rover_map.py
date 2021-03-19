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
import math

class MapGenerator:

    def __init__(self, x_limit=100, y_limit=100, x_pixels=None, y_pixels=None, num_circles=5, rad_circles=15):

        self.x_limit = x_limit
        self.y_limit = y_limit
        # if no specific description, one pixel corresponds one meter
        if x_pixels == None and y_pixels == None:
            self.x_pixels = self.x_limit
            self.y_pixels = self.y_limit
        else:
            self.x_pixels = x_pixels
            self.y_pixels = y_pixels
        self.x_scale = self.x_pixels / self.x_limit
        self.y_scale = self.y_pixels / self.y_limit

        self.num_circles = num_circles
        self.rad_circles = rad_circles
        # init map w/ all feature-poor region (expressed as "0")
        self.map = np.zeros((self.y_pixels, self.x_pixels))
        self.fig = None

    def get_random_map(self, x_pixels=100, y_pixels=100):
        # init number of pixels for each axis
        self.x_pixels = x_pixels
        self.y_pixels = y_pixels
        # put feature-rich circles randomly
        for i in range(self.num_circles):
            # generate center of circle
            xc = random.randint(self.rad_circles, self.x_pixels - self.rad_circles)
            yc = random.randint(self.rad_circles, self.y_pixels - self.rad_circles)
            x, y = np.meshgrid(np.arange(self.x_pixels), np.arange(self.y_pixels))
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

        # reset map information
        self.x_limit = self.map.shape[1]
        self.y_limit = self.map.shape[0]
        self.x_pixels = self.x_limit
        self.y_pixels = self.y_limit
        self.x_scale = 1
        self.y_scale = 1
        return self.map

    def select_location(self, x_range, y_range):
        self.map = self.map[y_range[0]:y_range[1], x_range[0]:x_range[1]]
        # reset map information
        self.x_limit = self.map.shape[1]
        self.y_limit = self.map.shape[0]
        self.x_pixels = self.x_limit
        self.y_pixels = self.y_limit
        self.x_scale = 1
        self.y_scale = 1
        return self.map

    def is_feature_rich(self, location):
        x_index = location[0] * self.x_scale // 1
        y_index = location[1] * self.y_scale // 1
        return self.map[int(y_index), int(x_index)]

    def reset_random_map(self):
        self.map = np.zeros((self.y_pixels, self.x_pixels))
        return self.get_random_map()

    def show_map(self, path=None):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            self.fig_list = []
        self.ax.imshow(self.map)
        if path is not None:
            fig = self.ax.plot(path[:, 0] * self.x_scale, path[:, 1] * self.y_scale, color='red')
            self.fig_list.append(fig)
        plt.show()

    def generate_animation(self, name):
        ani = ArtistAnimation(self.fig, self.fig_list)
        ani.save(name, writer='pillow')

if __name__ == '__main__':
    # test each method
    mg = MapGenerator(x_limit=100, y_limit=100, x_pixels=100, y_pixels=100)
    rover_map = mg.get_random_map()
    plt.imshow(mg.map)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.savefig('fig/binary_mars.png')
    plt.show()

