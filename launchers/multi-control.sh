#!/bin/bash
source /environment.sh
dt-launchfile-init
rosrun my_package multi_control.py
dt-launchfile-join