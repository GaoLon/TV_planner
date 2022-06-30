<h1>Tensor Voting Planner</h1>

<h2>Prerequsite</h2>

* Platform: ROS noetic
* Nvidia Cuda10.1

<h2>Run the repo</h2>

1. Clone the repo

   ```
   git clone https://github.com/GaoLon/TV_planner.git
   ```
2. Build the workspace

   ```
   cd ~/TV_planner
   catkin_make
   ```
3. Run the demo

   ```
   cd ~/TV_planner
   source devel/setup.bash
   roslaunch tensorvoting demo_test_plan.launch
   ```

   Then you can use `/move_base_simple/goal` plugin in rviz to set goal and see results of different planner:
   
   * RED: K-NN
   * GREEN: Shortest path
   * YELLOW: Tensor Voting 

<h2>Note</h2>

1. There are some TODO list:
   * do experiments in different point cloud maps
   * there are some bugs about `std::cout` or `printf` function.
   * benchmark
1. Thanks to open-source work done by Liu Ming:
   * Liu, Ming, et al. "Normal estimation for pointcloud using GPU based sparse tensor voting." *2012 IEEE International Conference on Robotics and Biomimetics (ROBIO)*. IEEE, 2012.
   * Liu, Ming. "Robotic online path planning on point cloud." *IEEE transactions on cybernetics* 46.5 (2015): 1217-1228.
   * https://ram-lab.com/downloads/
