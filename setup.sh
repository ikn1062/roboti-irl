#!/bin/bash

source /opt/ros/humble/setup.bash

# rm -rf build/ install/ log/
colcon build
colcon test

source install/setup.bash
