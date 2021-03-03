# Planning for Active Perception for Planetary Exploration Rovers

---

## Overview
This project implements an active perception method proposed in "[Active localization for planetary rovers](https://ieeexplore.ieee.org/abstract/document/7500599)" 
that generates paths while leveraging path efficiency and localization error. We compare Error Propagation A* (EPA*) and A* in terms of planned path cost, localization 
uncertainty, and generalization ability against several maps generated randomly.

## Prerequisites
Python 3.6 or later 

## Code
### map.py
Implement MapGenerator class. It takes x- and y-axis limits, number of feature-rich regions, and radius of them.
Methods are as follows:
* **get_map** generates map w/ given parameters.
* **reset_map** resets map.
* **show_map** visualizes map w/ generated paths. 