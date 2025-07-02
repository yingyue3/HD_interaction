#!/bin/bash
source /environment.sh
dt-launchfile-init
rosrun my_package lane_following.py
dt-launchfile-join