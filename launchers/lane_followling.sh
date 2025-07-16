#!/bin/bash
source /environment.sh
dt-launchfile-init
ros my_package lane_following.py
dt-launchfile-join