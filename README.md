# Template: template-ros

This template provides a boilerplate repository
for developing ROS-based software in Duckietown.

**NOTE:** If you want to develop software that does not use
ROS, check out [this template](https://github.com/duckietown/template-basic).


## How to use it

### 1. Make the script executable:
chmod +x ./packages/my_package/src/camera_reader_node.py

### 2. rebuild the image:
dts devel build -f

### 3. launch the node
dts devel run -R ROBOT_NAME -L camera-reader


### run the docker
docker exec -it 70dc0ea2c534 /bin/sh

### copy from docker
docker cp de01ba5ba0e5:/code/catkin_ws/src/HD_interaction/dependencies-apt.txt ./data