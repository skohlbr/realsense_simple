# realsense_simple
Unofficial ros realsense driver with some internal outlier rejection.

For most purposes you probably want the official ros driver https://github.com/intel-ros/realsense/tree/indigo-devel/realsense_camera

Disadvantages of this driver
- less features (cannot set any of realsenses internal parameters, no ir images, etc)
- less efficient

Advantages
- some outlier filtering and downsampling
- can be easier to get up and running with on some systems

The ros node depends on [librealsense](https://github.com/IntelRealSense/librealsense) and was based on the cpp_pointcloud example.
