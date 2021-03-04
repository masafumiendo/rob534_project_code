# Planning for Active Perception for Planetary Exploration Rovers

---

## Overview
This project implements an active perception method proposed in "[Active localization for planetary rovers](https://ieeexplore.ieee.org/abstract/document/7500599)" 
that generates paths while leveraging path efficiency and localization error. We compare Error Propagation A* (EPA*) and A* in terms of planned path cost, localization 
uncertainty, and generalization ability against several maps generated randomly.

## Prerequisites
Python 3.6 or later. Also, you need to use OpenCV package to use **get_mars_map** function.

## Code
### map.py
Implement MapGenerator class. It takes x- and y-axis limits, number of feature-rich regions, and radius of them.
Methods are as follows:
* **get_random_map** generates random map w/ given parameters.
* **get_mars_map** takes mars orbital image to generate map.
* **reset_random_map** resets random map.
* **show_map** visualizes map w/ generated paths. 
* **generate_animation** generates .gif and save it.